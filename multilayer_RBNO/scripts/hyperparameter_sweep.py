#!/usr/bin/env python3
"""
Hyperparameter Sweep for MRBNO on Nonlinear Diffusion Dataset

This script performs a comprehensive hyperparameter search to find optimal
MRBNO configurations for the nonlinear PDE: -∇·(exp(m)∇u) = f
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import itertools
import json
import os
from datetime import datetime
import argparse
from typing import Dict, List, Tuple, Any
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings

warnings.filterwarnings('ignore')

# Add models directory to path in a way that is independent of CWD
THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR / "models"
sys.path.append(str(MODELS_DIR))

from MRBNO_simple import MRBNOSimple, PCAReducerMultiChannel
from MRBNO_lift_v2 import MRBNOLiftV2
from MRBNO_lift_v3 import MRBNOLiftV3

class HyperparameterSweep:
    def __init__(self, data_path: str = 'data/nonlinear_diffusion_dataset.npz',
                 output_dir: str = None, device: str = 'auto',
                 model_type: str = 'mrbno', seed: int = 42,
                 k_in: int = None, k_out: int = None, auto_k: bool = True):
        """
        Initialize hyperparameter sweep.

        Args:
            data_path: Path to the nonlinear diffusion dataset
            output_dir: Directory to save sweep results
            device: Device to use ('auto', 'cuda', 'cpu')
            model_type: Model type ('mrbno', 'mrbno_lift_v2', 'mrbno_lift_v3')
            seed: Random seed for reproducibility (None = random)
            k_in: Fixed number of PCA components for inputs (None if auto_k=True)
            k_out: Fixed number of PCA components for outputs (None if auto_k=True)
            auto_k: If True, automatically select k_in and k_out using k_from_cutoff
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.device = self._setup_device(device)
        self.model_type = model_type
        self.seed = seed
        self.auto_k = auto_k
        self.fixed_k_in = k_in
        self.fixed_k_out = k_out
        
        # Set output directory to SCRATCH/logs if not specified
        if output_dir is None:
            scratch_dir = os.environ.get('SCRATCH', '/tmp')
            output_dir = os.path.join(scratch_dir, 'logs')
        
        # Set random seed if provided
        if seed is not None:
            self._set_seed(seed)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and prepare data
        self._load_data()
        
        # Define hyperparameter search space
        self._define_search_space()
        
        # Create PCA reducer once and reuse it (since boundary params are fixed)
        # This ensures consistent data preprocessing across all runs
        self.rb, self.k_in, self.k_out = self._create_pca_reducer_once()
        print(f"PCA reducer created: k_in={self.k_in}, k_out={self.k_out}")
        
        # Results storage
        self.results = []
        
        # Bayesian optimization setup
        self.bayesian_optimizer = None
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _load_data(self):
        """Load and preprocess the dataset with 70/20/10 train/val/test split."""
        print(f"Loading data from {self.data_path}...")
        
        # Load dataset
        data = np.load(self.data_path)
        f_all = data['f']
        m_all = data['m']
        u_all = data['u']
        
        # Stack f and m as 2-channel input
        X = np.stack([f_all, m_all], axis=1)
        Y = u_all
        
        # 70/20/10 train/val/test split FIRST (before normalization to avoid data leakage)
        n_total = len(X)
        n_train = int(0.7 * n_total)
        n_val = int(0.2 * n_total)
        n_test = n_total - n_train - n_val
        
        X_train_raw = X[:n_train]
        Y_train_raw = Y[:n_train]
        X_val_raw = X[n_train:n_train+n_val]
        Y_val_raw = Y[n_train:n_train+n_val]
        X_test_raw = X[n_train+n_val:]
        Y_test_raw = Y[n_train+n_val:]
        
        # Normalize using TRAIN statistics only (fixes data leakage)
        x_mean = X_train_raw.mean()
        x_std = X_train_raw.std()
        y_mean = Y_train_raw.mean()
        y_std = Y_train_raw.std()
        
        # Apply normalization to all splits
        self.X_train = (X_train_raw - x_mean) / (x_std + 1e-8)
        self.Y_train = (Y_train_raw - y_mean) / (y_std + 1e-8)
        self.X_val = (X_val_raw - x_mean) / (x_std + 1e-8)
        self.Y_val = (Y_val_raw - y_mean) / (y_std + 1e-8)
        self.X_test = (X_test_raw - x_mean) / (x_std + 1e-8)
        self.Y_test = (Y_test_raw - y_mean) / (y_std + 1e-8)
        
        # Store normalization stats for potential denormalization
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        
        # Convert to PyTorch tensors
        self.X_train_t = torch.from_numpy(self.X_train).float()
        self.Y_train_t = torch.from_numpy(self.Y_train).float().unsqueeze(1)
        self.X_val_t = torch.from_numpy(self.X_val).float()
        self.Y_val_t = torch.from_numpy(self.Y_val).float().unsqueeze(1)
        self.X_test_t = torch.from_numpy(self.X_test).float()
        self.Y_test_t = torch.from_numpy(self.Y_test).float().unsqueeze(1)
        
        print(f"Data loaded: {len(self.X_train)} train, {len(self.X_val)} val, {len(self.X_test)} test samples")
        print(f"Using device: {self.device}")
    
    def _define_search_space(self):
        """Define the hyperparameter search space."""
        if self.model_type == 'mrbno':
            # Original MRBNO Simple search space
            self.search_space = {
                'layers': [3, 4, 5, 6, 7, 8, 9, 10],
                'hidden': [256, 512, 768, 1024, 2048],
                'dropout': [0.0, 0.1, 0.2],
                'latent_batch_norm': [False, True],
                'batch_size': [64],
                'max_grad_norm': [5.0],
                'use_skip_connections': [True],
                'init_scale': [0.05, 0.1, 0.2],
                'use_boundary_weighting': [False],
                'bc_weight': [1.0],
                'boundary_threshold': [0.1],
                'early_stopping_patience': [50],
                'early_stopping_min_delta': [0.0],
            }
        elif self.model_type == 'mrbno_lift_v2':
            # Lifted MRBNO v2: for now sweep only layers and width, keep other hyperparameters fixed
            self.search_space = {
                'layers': [3, 4, 5, 6, 7, 8, 9, 10],
                'width': [8, 12, 16, 24, 32, 64],
                'dropout': [0.0],
                'batch_size': [32],
                'max_grad_norm': [3.0],
                'init_scale': [0.1],
                'use_skip_connections': [True],
                'use_boundary_weighting': [False],
                'bc_weight': [1.0],
                'boundary_threshold': [0.1],
                'early_stopping_patience': [50],
                'early_stopping_min_delta': [0.0],
            }
        elif self.model_type == 'mrbno_lift_v3':
            # Lifted MRBNO v3: sweep all hyperparameters
            self.search_space = {
                'layers': [3, 4, 5, 6, 7, 8, 9, 10],
                'width': [8, 12, 16, 24, 32, 64],
                'dropout': [0.0, 0.1, 0.2],
                'batch_size': [8, 64],
                'max_grad_norm': [3.0, 5.0, 20.0],
                'init_scale': [0.05, 0.1, 0.2],
                'use_skip_connections': [True],
                'use_batch_norm': [False, True],
                'use_coordinates': [False, True],
                'use_boundary_weighting': [False],
                'bc_weight': [1.0],
                'boundary_threshold': [0.1],
                'early_stopping_patience': [30],  # Stop if no improvement for N epochs
                'early_stopping_min_delta': [0.0],  # Minimum change to qualify as improvement
            }
        else:
            raise ValueError(f"Unsupported model_type '{self.model_type}'.")

        total_combinations = 1
        for values in self.search_space.values():
            total_combinations *= len(values)

        print(f"Search space defined for {self.model_type}: {total_combinations:,} total combinations")
        print("Key hyperparameters:")
        for key, values in self.search_space.items():
            print(f"  {key}: {values}")

    def _create_pca_reducer_once(self) -> Tuple[PCAReducerMultiChannel, int, int]:
        """Create PCA reducer once using fixed boundary parameters (reused for all configs)."""
        # Use fixed values from search space (they're the same for all configs)
        rb = PCAReducerMultiChannel(
            bc_weight=1.0,  # Fixed in all search spaces
            boundary_threshold=0.1,  # Fixed in all search spaces
            use_boundary_weighting=False,  # Fixed in all search spaces
        ).fit(self.X_train, self.Y_train)
        
        # Auto-select k_in and k_out if enabled
        if self.auto_k:
            epsilon = 1e-5
            k_in, k_out = rb.k_from_cutoff(j_in=1-1*epsilon, j_out=1-epsilon)
            print(f"  Auto-selected k_in={k_in}, k_out={k_out}")
        else:
            k_in = self.fixed_k_in
            k_out = self.fixed_k_out
            if k_in is None or k_out is None:
                raise ValueError("k_in and k_out must be provided when auto_k=False")
        
        return rb, k_in, k_out
    
    def _get_pca_reducer(self) -> Tuple[PCAReducerMultiChannel, int, int]:
        """Get the pre-computed PCA reducer (reused for all configs)."""
        return self.rb, self.k_in, self.k_out

    def _create_model(self, config: Dict[str, Any], rb: PCAReducerMultiChannel, k_in: int, k_out: int):
        """Create model with given configuration."""
        if self.model_type == 'mrbno':
            return MRBNOSimple(
                rb=rb,
                k_in=k_in,
                k_out=k_out,
                layers=config['layers'],
                hidden=config['hidden'],
                dropout=config['dropout'],
                use_skip_connections=config['use_skip_connections'],
                latent_batch_norm=config.get('latent_batch_norm', False),
                init_scale=config.get('init_scale', 0.1),
            )
        elif self.model_type == 'mrbno_lift_v2':
            return MRBNOLiftV2(
                width=config['width'],
                rb=rb,
                k_in=k_in,
                k_out=k_out,
                layers=config['layers'],
                dropout=config['dropout'],
                init_scale=config.get('init_scale', 0.1),
                use_skip_connections=config['use_skip_connections'],
            )
        elif self.model_type == 'mrbno_lift_v3':
            return MRBNOLiftV3(
                width=config['width'],
                rb=rb,
                k_in=k_in,
                k_out=k_out,
                layers=config['layers'],
                dropout=config['dropout'],
                init_scale=config.get('init_scale', 0.1),
                use_skip_connections=config['use_skip_connections'],
                use_batch_norm=config.get('use_batch_norm', False),
                use_coordinates=config.get('use_coordinates', False),
            )
        else:
            raise ValueError(f"Unsupported model_type '{self.model_type}'.")
    
    def _train_model(self, model, config: Dict[str, Any], epochs: int = 100) -> Dict[str, float]:
        """
        Train a single model configuration.
        
        Returns:
            Dictionary with best_train_loss, best_val_loss, final_val_loss, 
            test_loss, val_rel_error, test_rel_error
        """
        model.to(self.device)
        
        # Create data loaders
        batch_size = config['batch_size']
        train_loader = DataLoader(
            TensorDataset(self.X_train_t, self.Y_train_t), 
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(self.X_val_t, self.Y_val_t), 
            batch_size=batch_size, shuffle=False
        )
        
        # Optimizer and scheduler (cosine_lr_high protocol)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=2e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=0.0
        )
        
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        
        # Gradient clipping threshold from config (always use, but with different values)
        max_grad_norm = config.get('max_grad_norm', 5.0)
        
        # Early stopping setup
        early_stopping_patience = config.get('early_stopping_patience', 50)
        early_stopping_min_delta = config.get('early_stopping_min_delta', 0.0)
        epochs_without_improvement = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                
                try:
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    
                    # Check for NaN or inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"    Warning: NaN/Inf loss detected at epoch {epoch}, skipping...")
                        continue
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    
                    # Check for NaN gradients
                    has_nan_grad = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
                    if has_nan_grad:
                        print(f"    Warning: NaN gradients detected at epoch {epoch}, skipping step...")
                        optimizer.zero_grad()
                        continue
                    
                    optimizer.step()
                    train_loss += loss.item() * xb.size(0)
                    
                except Exception as e:
                    print(f"    Error during training at epoch {epoch}: {e}")
                    # Continue to next batch
                    continue
                    
            train_loss /= len(train_loader.dataset)
            
            # Clear GPU cache periodically
            if epoch % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)
            
            # Track best losses
            if train_loss < best_train_loss:
                best_train_loss = train_loss
            
            # Early stopping: check if validation loss improved
            if val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_epoch = epoch
            else:
                epochs_without_improvement += 1
            
            scheduler.step()
            
            # Early stopping check
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(f"    Early stopping at epoch {epoch+1}: no improvement for {early_stopping_patience} epochs (best val loss: {best_val_loss:.4e} at epoch {best_epoch+1})")
                break
        
        # Final validation loss
        final_val_loss = val_loss
        
        # Test evaluation
        test_loader = DataLoader(
            TensorDataset(self.X_test_t, self.Y_test_t), 
            batch_size=batch_size, shuffle=False
        )
        model.eval()
        test_loss = 0.0
        test_rel_error_sum = 0.0
        val_rel_error_sum = 0.0
        test_count = 0
        val_count = 0
        
        with torch.no_grad():
            # Test set
            for xb, yb in test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = model(xb)
                loss = criterion(pred, yb)
                test_loss += loss.item() * xb.size(0)
                
                # Relative error: ||u_pred - u_true|| / ||u_true||
                # Denormalize for relative error computation
                pred_denorm = pred * self.y_std + self.y_mean
                yb_denorm = yb * self.y_std + self.y_mean
                # Compute norm per sample: flatten spatial dimensions
                pred_flat = pred_denorm.view(pred_denorm.size(0), -1)
                yb_flat = yb_denorm.view(yb_denorm.size(0), -1)
                rel_error = torch.norm(pred_flat - yb_flat, dim=1) / (torch.norm(yb_flat, dim=1) + 1e-8)
                test_rel_error_sum += rel_error.sum().item()
                test_count += xb.size(0)
            
            # Validation set relative error
            for xb, yb in val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = model(xb)
                # Denormalize for relative error computation
                pred_denorm = pred * self.y_std + self.y_mean
                yb_denorm = yb * self.y_std + self.y_mean
                # Compute norm per sample: flatten spatial dimensions
                pred_flat = pred_denorm.view(pred_denorm.size(0), -1)
                yb_flat = yb_denorm.view(yb_denorm.size(0), -1)
                rel_error = torch.norm(pred_flat - yb_flat, dim=1) / (torch.norm(yb_flat, dim=1) + 1e-8)
                val_rel_error_sum += rel_error.sum().item()
                val_count += xb.size(0)
        
        test_loss /= len(test_loader.dataset)
        test_rel_error = test_rel_error_sum / test_count
        val_rel_error = val_rel_error_sum / val_count
        
        return {
            'best_train_loss': best_train_loss,
            'best_val_loss': best_val_loss,
            'final_val_loss': final_val_loss,
            'test_loss': test_loss,
            'val_rel_error': val_rel_error,
            'test_rel_error': test_rel_error,
        }
    
    def run_sweep(self, max_configs: int = 200, epochs_per_config: int = 100, 
                  search_strategy: str = 'bayesian', save_frequency: int = 10):
        """
        Run the hyperparameter sweep.
        
        Args:
            max_configs: Maximum number of configurations to try
            epochs_per_config: Number of epochs per configuration
            search_strategy: 'random', 'grid', or 'bayesian'
            save_frequency: Save results every N configurations
        """
        print(f"\nStarting hyperparameter sweep...")
        print(f"Model type: {self.model_type}")
        print(f"Max configurations: {max_configs}")
        print(f"Epochs per config: {epochs_per_config}")
        print(f"Search strategy: {search_strategy}")
        
        # Generate initial configurations
        if search_strategy == 'bayesian':
            # Start with random configurations, then use Bayesian optimization
            initial_configs = self._generate_random_configs(min(20, max_configs))
            configs = initial_configs
            remaining_configs = max_configs - len(initial_configs)
        elif search_strategy == 'random':
            configs = self._generate_random_configs(max_configs)
            remaining_configs = 0
        else:  # grid
            configs = self._generate_grid_configs(max_configs)
            remaining_configs = 0
        
        print(f"Generated {len(configs)} initial configurations")
        
        # Run sweep
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{max_configs}] Testing configuration...")
            print(f"Config: {config}")
            
            try:
                # Get pre-computed PCA reducer (reused for all configs)
                rb, k_in, k_out = self._get_pca_reducer()
                
                # Create model
                model = self._create_model(config, rb, k_in, k_out)
                
                # Count parameters
                n_params = sum(p.numel() for p in model.parameters())
                
                # Train model
                metrics = self._train_model(model, config, epochs_per_config)
                
                record_config = config.copy()
                record_config.update({'lr': 2e-3, 'weight_decay': 1e-4, 'scheduler': 'cosine_lr_high', 'k_in': k_in, 'k_out': k_out})

                result = {
                    'config': record_config,
                    'n_params': n_params,
                    'best_train_loss': metrics['best_train_loss'],
                    'best_val_loss': metrics['best_val_loss'],
                    'final_val_loss': metrics['final_val_loss'],
                    'test_loss': metrics['test_loss'],
                    'val_rel_error': metrics['val_rel_error'],
                    'test_rel_error': metrics['test_rel_error'],
                    'train_val_gap': metrics['best_val_loss'] - metrics['best_train_loss'],
                    'timestamp': datetime.now().isoformat()
                }

                self.results.append(result)
                
                print(f"Results: train={metrics['best_train_loss']:.4e}, val={metrics['best_val_loss']:.4e}, test={metrics['test_loss']:.4e}, params={n_params:,}")
                
                # Save intermediate results
                if (i + 1) % save_frequency == 0:
                    self._save_results(f"intermediate_results_{i+1}.json")
                
                # Clean up model to free GPU memory (keep rb, it's reused)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error with config {config}: {error_msg}")
                
                # Log error details to a file
                try:
                    with open(f"{self.output_dir}/errors.log", "a") as f:
                        f.write(f"\nConfig {i+1}: {config}\n")
                        f.write(f"Error: {error_msg}\n\n")
                except:
                    pass
                
                # Clear GPU memory on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                continue
        
        # Continue with Bayesian optimization if requested
        if search_strategy == 'bayesian' and remaining_configs > 0:
            print(f"\nStarting Bayesian optimization for {remaining_configs} more configurations...")
            
            for i in range(remaining_configs):
                config_idx = len(configs) + i + 1
                print(f"\n[{config_idx}/{max_configs}] Bayesian optimization step...")
                
                # Generate next configurations using Bayesian optimization
                next_configs = self._bayesian_optimization_step(n_candidates=1)
                config = next_configs[0]
                
                print(f"Config: {config}")
                
                try:
                    # Get pre-computed PCA reducer (reused for all configs)
                    rb, k_in, k_out = self._get_pca_reducer()
                    
                    # Create model
                    model = self._create_model(config, rb, k_in, k_out)
                    
                    # Count parameters
                    n_params = sum(p.numel() for p in model.parameters())
                    
                    # Train model
                    metrics = self._train_model(model, config, epochs_per_config)
                    
                    # Store results
                    record_config = config.copy()
                    record_config.update({'lr': 2e-3, 'weight_decay': 1e-4, 'scheduler': 'cosine_lr_high', 'k_in': k_in, 'k_out': k_out})

                    result = {
                        'config': record_config,
                        'n_params': n_params,
                        'best_train_loss': metrics['best_train_loss'],
                        'best_val_loss': metrics['best_val_loss'],
                        'final_val_loss': metrics['final_val_loss'],
                        'test_loss': metrics['test_loss'],
                        'val_rel_error': metrics['val_rel_error'],
                        'test_rel_error': metrics['test_rel_error'],
                        'train_val_gap': metrics['best_val_loss'] - metrics['best_train_loss'],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.results.append(result)
                    
                    # Clean up model to free GPU memory (keep rb, it's reused)
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    print(f"Results: train={metrics['best_train_loss']:.4e}, val={metrics['best_val_loss']:.4e}, test={metrics['test_loss']:.4e}, params={n_params:,}")
                    
                    # Save intermediate results
                    if (config_idx) % save_frequency == 0:
                        self._save_results(f"intermediate_results_{config_idx}.json")
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"Error with config {config}: {error_msg}")
                    
                    # Log error details
                    try:
                        with open(f"{self.output_dir}/errors.log", "a") as f:
                            f.write(f"\nConfig {config_idx}: {config}\n")
                            f.write(f"Error: {error_msg}\n\n")
                    except:
                        pass
                    
                    # Clear GPU memory on error
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    continue
        
        # Save final results (sorted by validation loss)
        sorted_results = sorted(self.results, key=lambda x: x['best_val_loss'])
        self.results = sorted_results
        self._save_results("final_results.json")
        self._create_plots()
        self._print_summary()
    
    def _generate_random_configs(self, max_configs: int) -> List[Dict[str, Any]]:
        """Generate random configurations from the search space."""
        configs = []
        attempts = 0
        max_attempts = max_configs * 10

        while len(configs) < max_configs and attempts < max_attempts:
            config = {key: np.random.choice(values) for key, values in self.search_space.items()}
            config = self._convert_numpy_types(config)
            configs.append(config)
            attempts += 1

        if len(configs) < max_configs:
            print(f"Warning: generated only {len(configs)} configs out of requested {max_configs}")
        return configs
    
    def _generate_grid_configs(self, max_configs: int) -> List[Dict[str, Any]]:
        """Generate configurations via grid search."""
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        configs = []
        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            configs.append(self._convert_numpy_types(config))
            if len(configs) >= max_configs:
                break
        return configs
    
    def _config_to_vector(self, config: Dict[str, Any]) -> np.ndarray:
        """Convert configuration to numerical vector for Bayesian optimization."""
        if self.model_type == 'mrbno':
            return np.array(
                [
                    config['layers'],
                    config['hidden'],
                    config['dropout'],
                    1 if config.get('latent_batch_norm', False) else 0,
                    config['batch_size'],
                    config.get('init_scale', 0.1),
                ],
                dtype=float,
            )
        elif self.model_type == 'mrbno_lift_v2':
            return np.array(
                [
                    config['layers'],
                    config['width'],
                    config['dropout'],
                    config['batch_size'],
                    config.get('init_scale', 0.1),
                ],
                dtype=float,
            )
        elif self.model_type == 'mrbno_lift_v3':
            return np.array(
                [
                    config['layers'],
                    config['width'],
                    config['dropout'],
                    1 if config.get('use_batch_norm', False) else 0,
                    1 if config.get('use_coordinates', False) else 0,
                    config['batch_size'],
                    config.get('init_scale', 0.1),
                ],
                dtype=float,
            )
        else:
            raise ValueError(f"Unsupported model_type '{self.model_type}'.")

    def _vector_to_config(self, vector: np.ndarray) -> Dict[str, Any]:
        """Convert numerical vector back to configuration."""
        config = {}
        
        if self.model_type == 'mrbno':
            layers_values = np.array(self.search_space['layers'])
            hidden_values = np.array(self.search_space['hidden'])
            dropout_values = np.array(self.search_space['dropout'])
            bn_values = self.search_space['latent_batch_norm']
            batch_values = np.array(self.search_space['batch_size'])
            init_values = np.array(self.search_space['init_scale'])

            config['layers'] = int(layers_values[np.argmin(np.abs(layers_values - vector[0]))])
            config['hidden'] = int(hidden_values[np.argmin(np.abs(hidden_values - vector[1]))])
            config['dropout'] = float(dropout_values[np.argmin(np.abs(dropout_values - vector[2]))])
            config['latent_batch_norm'] = bool(bn_values[int(round(vector[3])) % len(bn_values)])
            config['batch_size'] = int(batch_values[np.argmin(np.abs(batch_values - vector[4]))])
            config['init_scale'] = float(init_values[np.argmin(np.abs(init_values - vector[5]))])
        elif self.model_type == 'mrbno_lift_v2':
            layers_values = np.array(self.search_space['layers'])
            width_values = np.array(self.search_space['width'])
            dropout_values = np.array(self.search_space['dropout'])
            batch_values = np.array(self.search_space['batch_size'])
            init_values = np.array(self.search_space['init_scale'])

            config['layers'] = int(layers_values[np.argmin(np.abs(layers_values - vector[0]))])
            config['width'] = int(width_values[np.argmin(np.abs(width_values - vector[1]))])
            config['dropout'] = float(dropout_values[np.argmin(np.abs(dropout_values - vector[2]))])
            config['batch_size'] = int(batch_values[np.argmin(np.abs(batch_values - vector[3]))])
            config['init_scale'] = float(init_values[np.argmin(np.abs(init_values - vector[4]))])
        elif self.model_type == 'mrbno_lift_v3':
            layers_values = np.array(self.search_space['layers'])
            width_values = np.array(self.search_space['width'])
            dropout_values = np.array(self.search_space['dropout'])
            bn_values = self.search_space['use_batch_norm']
            coords_values = self.search_space['use_coordinates']
            batch_values = np.array(self.search_space['batch_size'])
            init_values = np.array(self.search_space['init_scale'])

            config['layers'] = int(layers_values[np.argmin(np.abs(layers_values - vector[0]))])
            config['width'] = int(width_values[np.argmin(np.abs(width_values - vector[1]))])
            config['dropout'] = float(dropout_values[np.argmin(np.abs(dropout_values - vector[2]))])
            config['use_batch_norm'] = bool(bn_values[int(round(vector[3])) % len(bn_values)])
            config['use_coordinates'] = bool(coords_values[int(round(vector[4])) % len(coords_values)])
            config['batch_size'] = int(batch_values[np.argmin(np.abs(batch_values - vector[5]))])
            config['init_scale'] = float(init_values[np.argmin(np.abs(init_values - vector[6]))])
        else:
            raise ValueError(f"Unsupported model_type '{self.model_type}'.")

        config['max_grad_norm'] = self.search_space['max_grad_norm'][0]
        config['use_skip_connections'] = self.search_space['use_skip_connections'][0]
        config['use_boundary_weighting'] = self.search_space['use_boundary_weighting'][0]
        config['bc_weight'] = self.search_space['bc_weight'][0]
        config['boundary_threshold'] = self.search_space['boundary_threshold'][0]
        config['early_stopping_patience'] = self.search_space['early_stopping_patience'][0]
        config['early_stopping_min_delta'] = self.search_space['early_stopping_min_delta'][0]

        return self._convert_numpy_types(config)
    
    def _acquisition_function(self, x: np.ndarray, gp: GaussianProcessRegressor, 
                            best_val_loss: float, exploration_weight: float = 0.1) -> float:
        """Expected Improvement acquisition function."""
        mean, std = gp.predict(x.reshape(1, -1), return_std=True)
        improvement = best_val_loss - mean[0]
        return improvement + exploration_weight * std[0]
    
    def _bayesian_optimization_step(self, n_candidates: int = 5) -> List[Dict[str, Any]]:
        """Generate next configurations using Bayesian optimization."""
        # Need at least 5 successful results for Bayesian optimization
        successful_results = [r for r in self.results if 'best_val_loss' in r and not np.isnan(r['best_val_loss']) and not np.isinf(r['best_val_loss'])]
        
        if len(successful_results) < 5:
            print(f"  Not enough successful results ({len(successful_results)} < 5), using random search")
            return self._generate_random_configs(n_candidates)

        try:
            X = np.array([self._config_to_vector(r['config']) for r in successful_results])
            y = np.array([r['best_val_loss'] for r in successful_results])

            # Check for constant y values (would cause GP fitting issues)
            if np.std(y) < 1e-10:
                print(f"  Validation losses are too similar (std={np.std(y):.2e}), using random search")
                return self._generate_random_configs(n_candidates)

            kernel = Matern(length_scale=1.0, nu=2.5)
            gp = GaussianProcessRegressor(kernel=kernel, random_state=42, alpha=1e-6)
            gp.fit(X, y)

            best_val_loss = y.min()
            candidates = []
            for _ in range(n_candidates * 10):
                candidate = self._config_to_vector(self._generate_random_configs(1)[0])
                acq_value = self._acquisition_function(candidate, gp, best_val_loss)
                candidates.append((acq_value, candidate))

            candidates.sort(key=lambda x: x[0], reverse=True)
            selected_configs = [self._vector_to_config(vec) for _, vec in candidates[:n_candidates]]
            return selected_configs
        except Exception as e:
            print(f"  Error in Bayesian optimization: {e}, falling back to random search")
            return self._generate_random_configs(n_candidates)
    
    def _convert_numpy_types(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy types to regular Python types."""
        converted = {}
        for key, value in config.items():
            if isinstance(value, np.integer):
                converted[key] = int(value)
            elif isinstance(value, np.floating):
                converted[key] = float(value)
            elif isinstance(value, np.bool_):
                converted[key] = bool(value)
            else:
                converted[key] = value
        return converted
    
    def _save_results(self, filename: str):
        """Save results to JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")
    
    def _create_plots(self):
        """Create plots summarising sweep results."""
        if not self.results:
            print("No results to plot")
            return

        print("Creating plots...")
        df = pd.DataFrame([
            {
                **result['config'],
                'val_loss': result['best_val_loss'],
                'train_loss': result['best_train_loss'],
                'train_val_gap': result['train_val_gap'],
                'n_params': result['n_params'],
            }
            for result in self.results
        ])

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        if self.model_type == 'mrbno_lift_v2':
            model_title = 'MRBNO Lift V2'
        elif self.model_type == 'mrbno_lift_v3':
            model_title = 'MRBNO Lift V3'
        else:
            model_title = 'MRBNO Simple'
        fig.suptitle(f'{model_title} Hyperparameter Sweep', fontsize=16)

        if self.model_type in ['mrbno_lift_v2', 'mrbno_lift_v3']:
            # Layers vs width
            axes[0, 0].scatter(df['layers'], df['width'], c=df['val_loss'], cmap='viridis', s=100, alpha=0.7)
            axes[0, 0].set_xlabel('Layers')
            axes[0, 0].set_ylabel('Width')
            axes[0, 0].set_title('Validation loss by architecture')

            # Dropout vs layers
            axes[0, 1].scatter(df['dropout'], df['layers'], c=df['val_loss'], cmap='viridis', s=100, alpha=0.7)
            axes[0, 1].set_xlabel('Dropout')
            axes[0, 1].set_ylabel('Layers')
            axes[0, 1].set_title('Validation loss vs regularisation')

            # Width vs batch size
            axes[0, 2].scatter(df['width'], df['batch_size'], c=df['val_loss'], cmap='viridis', s=100, alpha=0.7)
            axes[0, 2].set_xlabel('Width')
            axes[0, 2].set_ylabel('Batch size')
            axes[0, 2].set_title('Training hyperparameters')

            # Parameters vs train/val gap
            axes[1, 0].scatter(df['n_params'], df['train_val_gap'], c=df['val_loss'], cmap='viridis', s=100, alpha=0.7)
            axes[1, 0].set_xlabel('Number of parameters')
            axes[1, 0].set_ylabel('Train-val gap')
            axes[1, 0].set_title('Overfitting diagnostic')

            # Histogram of validation loss
            axes[1, 1].hist(df['val_loss'], bins=15, color='skyblue', edgecolor='black')
            axes[1, 1].axvline(df['val_loss'].min(), color='red', linestyle='--', label=f"Best: {df['val_loss'].min():.4e}")
            axes[1, 1].set_xlabel('Validation loss')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Validation loss distribution')
            axes[1, 1].legend()

            # Top configurations
            top_n = df.sort_values('val_loss').head(10)
            axes[1, 2].barh(range(len(top_n)), top_n['val_loss'], color='lightcoral')
            axes[1, 2].set_yticks(range(len(top_n)))
            if self.model_type == 'mrbno_lift_v3':
                axes[1, 2].set_yticklabels([
                    f"layers={row.layers}, width={row.width}, dropout={row.dropout}, bn={row.use_batch_norm}, coords={row.use_coordinates}" 
                    for row in top_n.itertuples()
                ])
            else:
                axes[1, 2].set_yticklabels([
                    f"layers={row.layers}, width={row.width}, dropout={row.dropout}" for row in top_n.itertuples()
                ])
            axes[1, 2].invert_yaxis()
            axes[1, 2].set_xlabel('Validation loss')
            axes[1, 2].set_title('Top 10 configurations')
        else:
            # Layers vs hidden size
            axes[0, 0].scatter(df['layers'], df['hidden'], c=df['val_loss'], cmap='viridis', s=100, alpha=0.7)
            axes[0, 0].set_xlabel('Layers')
            axes[0, 0].set_ylabel('Hidden dimension')
            axes[0, 0].set_title('Validation loss by architecture')

            # Dropout vs latent batch norm
            axes[0, 1].scatter(df['dropout'], df['latent_batch_norm'].astype(int), c=df['val_loss'], cmap='viridis', s=100, alpha=0.7)
            axes[0, 1].set_xlabel('Dropout')
            axes[0, 1].set_ylabel('Latent batch norm (0/1)')
            axes[0, 1].set_title('Validation loss vs regularisation')

            # Batch size vs init scale
            axes[0, 2].scatter(df['batch_size'], df['init_scale'], c=df['val_loss'], cmap='viridis', s=100, alpha=0.7)
            axes[0, 2].set_xlabel('Batch size')
            axes[0, 2].set_ylabel('Init scale')
            axes[0, 2].set_title('Training hyperparameters')

            # Parameters vs train/val gap
            axes[1, 0].scatter(df['n_params'], df['train_val_gap'], c=df['val_loss'], cmap='viridis', s=100, alpha=0.7)
            axes[1, 0].set_xlabel('Number of parameters')
            axes[1, 0].set_ylabel('Train-val gap')
            axes[1, 0].set_title('Overfitting diagnostic')

            # Histogram of validation loss
            axes[1, 1].hist(df['val_loss'], bins=15, color='skyblue', edgecolor='black')
            axes[1, 1].axvline(df['val_loss'].min(), color='red', linestyle='--', label=f"Best: {df['val_loss'].min():.4e}")
            axes[1, 1].set_xlabel('Validation loss')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Validation loss distribution')
            axes[1, 1].legend()

            # Top configurations
            top_n = df.sort_values('val_loss').head(10)
            axes[1, 2].barh(range(len(top_n)), top_n['val_loss'], color='lightcoral')
            axes[1, 2].set_yticks(range(len(top_n)))
            axes[1, 2].set_yticklabels([
                f"layers={row.layers}, hidden={row.hidden}, bn={row.latent_batch_norm}" for row in top_n.itertuples()
            ])
            axes[1, 2].invert_yaxis()
            axes[1, 2].set_xlabel('Validation loss')
            axes[1, 2].set_title('Top 10 configurations')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, 'sweep_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Plots saved to {self.output_dir}/sweep_analysis.png")
    
    def _print_summary(self):
        """Print summary of best configurations."""
        if not self.results:
            print("No results to summarize")
            return

        sorted_results = sorted(self.results, key=lambda x: x['best_val_loss'])

        if self.model_type == 'mrbno_lift_v2':
            model_title = 'MRBNO LIFT V2'
        elif self.model_type == 'mrbno_lift_v3':
            model_title = 'MRBNO LIFT V3'
        else:
            model_title = 'MRBNO SIMPLE'
        print(f"\n{'='*80}")
        print(f"{model_title} HYPERPARAMETER SWEEP SUMMARY")
        print(f"{'='*80}")

        print("\nTop 10 configurations by validation loss:")
        print(f"{'Rank':<4} {'Val Loss':<12} {'Train Loss':<12} {'Gap':<12} {'Params':<10} {'Key Params'}")
        print("-" * 80)

        for i, result in enumerate(sorted_results[:10]):
            config = result['config']
            if self.model_type == 'mrbno_lift_v2':
                key_params = (f"layers={config['layers']}, width={config['width']}, "
                              f"dropout={config['dropout']}, batch={config['batch_size']}, "
                              f"init_scale={config.get('init_scale', 0.1)}")
            elif self.model_type == 'mrbno_lift_v3':
                key_params = (f"layers={config['layers']}, width={config['width']}, "
                              f"dropout={config['dropout']}, bn={config.get('use_batch_norm', False)}, "
                              f"coords={config.get('use_coordinates', False)}, "
                              f"init_scale={config.get('init_scale', 0.1)}")
            else:
                key_params = (f"layers={config['layers']}, hidden={config['hidden']}, "
                              f"dropout={config['dropout']}, bn={config['latent_batch_norm']}, "
                              f"batch={config['batch_size']}, init_scale={config['init_scale']}")
            print(f"{i+1:<4} {result['best_val_loss']:<12.4e} {result['best_train_loss']:<12.4e} "
                  f"{result['train_val_gap']:<12.4e} {result['n_params']:<10,} {key_params}")

        print(f"\n{'='*80}")
        print("PATTERN ANALYSIS")
        print(f"{'='*80}")

        best_configs = sorted_results[:20]
        layers = sorted(set(cfg['config']['layers'] for cfg in best_configs))
        dropout = sorted(set(cfg['config']['dropout'] for cfg in best_configs))

        print(f"Best layer counts (top 20): {layers}")
        print(f"Best dropout values (top 20): {dropout}")
        
        if self.model_type in ['mrbno_lift_v2', 'mrbno_lift_v3']:
            width = sorted(set(cfg['config']['width'] for cfg in best_configs))
            print(f"Best width values (top 20): {width}")
            if self.model_type == 'mrbno_lift_v3':
                bn = sorted(set(cfg['config'].get('use_batch_norm', False) for cfg in best_configs))
                coords = sorted(set(cfg['config'].get('use_coordinates', False) for cfg in best_configs))
                print(f"Batch norm usage (top 20): {bn}")
                print(f"Coordinates usage (top 20): {coords}")
        else:
            hidden = sorted(set(cfg['config']['hidden'] for cfg in best_configs))
            bn = sorted(set(cfg['config']['latent_batch_norm'] for cfg in best_configs))
            print(f"Best hidden sizes (top 20): {hidden}")
            print(f"Latent batch norm usage (top 20): {bn}")


def main():
    parser = argparse.ArgumentParser(description='MRBNO Hyperparameter Sweep')
    parser.add_argument('--dataset', default='small',
                       choices=['small', 'big'],
                       help='Dataset choice: small (data folder) or big ($SCRATCH)')
    parser.add_argument('--data_path', default=None,
                       help='Path to dataset (overrides --dataset if provided)')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory for results (default: $SCRATCH/logs)')
    parser.add_argument('--max_configs', type=int, default=200,
                       help='Maximum configurations to test')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Epochs per configuration (default 300 to match cosine_lr_high strategy)')
    parser.add_argument('--epochs_per_config', type=int, dest='epochs',
                       help='Alias for --epochs (for backwards compatibility)')
    parser.add_argument('--search_strategy', default='bayesian',
                       choices=['random', 'grid', 'bayesian'],
                       help='Search strategy: random, grid, or bayesian')
    parser.add_argument('--device', default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--model_type', default='mrbno',
                       choices=['mrbno', 'mrbno_lift_v2', 'mrbno_lift_v3'],
                       help='Model type')
    parser.add_argument('--auto_k', action='store_true', default=True,
                       help='Automatically select k_in and k_out using k_from_cutoff (default: True)')
    parser.add_argument('--no_auto_k', dest='auto_k', action='store_false',
                       help='Use fixed k_in and k_out values')
    parser.add_argument('--k_in', type=int, default=None,
                       help='Fixed number of PCA components for inputs (only used if --no_auto_k)')
    parser.add_argument('--k_out', type=int, default=None,
                       help='Fixed number of PCA components for outputs (only used if --no_auto_k)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (None = random)')
    
    args = parser.parse_args()
    
    # Determine data path
    if args.data_path is None:
        if args.dataset == 'small':
            data_path = 'data/nonlinear_diffusion_dataset.npz'
        else:  # big
            scratch_dir = os.environ.get('SCRATCH', '/pscratch/sd/d/dvod')
            data_path = os.path.join(scratch_dir, 'nonlinear_diffusion_dataset_big.npz')
    else:
        data_path = args.data_path
    
    # Create and run sweep
    sweep = HyperparameterSweep(
        data_path=data_path,
        output_dir=args.output_dir,
        device=args.device,
        model_type=args.model_type,
        seed=args.seed,
        k_in=args.k_in,
        k_out=args.k_out,
        auto_k=args.auto_k,
    )
    
    sweep.run_sweep(
        max_configs=args.max_configs,
        epochs_per_config=args.epochs,
        search_strategy=args.search_strategy
    )


if __name__ == '__main__':
    main()
