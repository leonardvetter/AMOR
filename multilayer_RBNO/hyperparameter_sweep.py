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
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Add models directory to path
sys.path.append('models')
from MRBNO_model_multichannel_v2 import MultiChannelMRBNOv2
from MRBNO_model_multichannel_v2_norm import MultiChannelMRBNOv2Norm
from MRBNO_hybrid_learnable import HybridLearnableMRBNO
from MRBNO_progressive_learnable import ProgressiveLearnableMRBNO
from MRBNO_adaptive_pca import AdaptivePCAMRBNO
from MRBNO_multiscale import MultiScaleMRBNO
from MRBNO_progressive import ProgressiveMRBNO
from MRBNO_pca_attention import PCAAttentionMRBNO
from MRBNO_pca_attention_v2 import PCAAttentionMRBNOv2
from MRBNO_learnable_v2 import LearnableMultiChannelMRBNOv2
from MRBNO_simple import MRBNOSimple, PCAReducerMultiChannel

class HyperparameterSweep:
    def __init__(self, data_path: str = '/pscratch/sd/d/dvod/nonlinear_diffusion_dataset_big.npz', 
                 output_dir: str = 'sweep_results', device: str = 'auto', 
                 model_type: str = 'mrbno', seed: int = 42):
        """
        Initialize hyperparameter sweep.
        
        Args:
            data_path: Path to the nonlinear diffusion dataset
            output_dir: Directory to save sweep results
            device: Device to use ('auto', 'cuda', 'cpu')
            model_type: Model type ('mrbno', 'mrbno_v2', 'mrbno_v2_norm', 'multiscale', 'progressive', 'pca_attention', 'pca_attention_v2', 'hybrid_learnable', 'progressive_learnable', 'adaptive_pca', 'learnable')
            seed: Random seed for reproducibility (None = random)
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.device = self._setup_device(device)
        self.model_type = model_type
        self.seed = seed
        
        # Set random seed if provided
        if seed is not None:
            self._set_seed(seed)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and prepare data
        self._load_data()
        
        # Define hyperparameter search space
        self._define_search_space()
        
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
        """Load and preprocess the dataset."""
        print(f"Loading data from {self.data_path}...")
        
        # Load dataset
        data = np.load(self.data_path)
        f_all = data['f']  # (1000, 100, 100)
        m_all = data['m']  # (1000, 100, 100)
        u_all = data['u']  # (1000, 100, 100)
        
        # Stack f and m as 2-channel input
        X = np.stack([f_all, m_all], axis=1)  # (1000, 2, 100, 100)
        Y = u_all  # (1000, 100, 100)
        
        # Train/val split FIRST (before normalization to avoid data leakage)
        train_frac = 0.7
        n_train = int(train_frac * len(X))
        
        X_train_raw, Y_train_raw = X[:n_train], Y[:n_train]
        X_val_raw, Y_val_raw = X[n_train:], Y[n_train:]
        
        # Normalize using TRAIN statistics only (fixes data leakage)
        x_mean = X_train_raw.mean()
        x_std = X_train_raw.std()
        y_mean = Y_train_raw.mean()
        y_std = Y_train_raw.std()
        
        # Apply normalization to both train and val
        self.X_train = (X_train_raw - x_mean) / (x_std + 1e-8)
        self.Y_train = (Y_train_raw - y_mean) / (y_std + 1e-8)
        self.X_val = (X_val_raw - x_mean) / (x_std + 1e-8)
        self.Y_val = (Y_val_raw - y_mean) / (y_std + 1e-8)
        
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
        
        print(f"Data loaded: {len(self.X_train)} train, {len(self.X_val)} val samples")
        print(f"Using device: {self.device}")
    
    def _define_search_space(self):
        """Define the hyperparameter search space."""
        # Base search space for all models
        base_search_space = {
            # Architecture
            'layers': [1, 2, 3],
            'hidden': [256, 512, 768],
            'depth': [0, 1, 2],
            
            # Regularization
            'dropout': [0.0, 0.1, 0.2],
            'block0_dropout': [0.0, 0.1, 0.2],
            'weight_decay': [1e-5, 1e-4, 1e-3],
            
            # Training
            'lr': [1e-3, 3e-3, 5e-3],
            'batch_size': [4],
            'warmup_epochs': [10],  # Always use warmup (5-15 epochs)
            'max_grad_norm': [1.0, 2.0, 5.0, 10.0],  # Gradient clipping threshold
            
            # Boundary weighting
            'use_boundary_weighting': [False],
            'bc_weight': [1.0],
            'boundary_threshold': [0.1],
            
            # Skip connections
            'use_skip_connections': [True],
        }
        
        # Model-specific search spaces
        if self.model_type == 'mrbno':
            self.search_space = {
                'layers': [1, 2, 3],
                'hidden': [256, 512, 768],
                'dropout': [0.0, 0.1, 0.2],
                'latent_batch_norm': [False, True],
                'weight_decay': [1e-5, 1e-4, 1e-3],
                'lr': [1e-3, 3e-3, 5e-3],
                'batch_size': [4],
                'warmup_epochs': [10],
                'max_grad_norm': [1.0, 2.0, 5.0, 10.0],
                'use_boundary_weighting': [False],
                'bc_weight': [1.0],
                'boundary_threshold': [0.1],
                'use_skip_connections': [True],
                'k_in': [40, 60, 80, 100, 120, 150, 180, 200],
                'k_out': [40, 50, 60, 75, 90, 100, 125, 150, 200],
            }
        elif self.model_type in ['mrbno_v2', 'mrbno_v2_norm', 'multiscale', 'progressive', 'pca_attention', 'pca_attention_v2', 'hybrid_learnable', 'progressive_learnable', 'adaptive_pca', 'learnable']:
            self.search_space = base_search_space.copy()
            if self.model_type in ['mrbno_v2', 'mrbno_v2_norm', 'pca_attention', 'learnable']:
                self.search_space.update({
                    'k_in': [40, 60, 80, 100, 120, 150, 180, 200],
                    'k_out': [40, 50, 60, 75, 90, 100, 125, 150, 200],
                })
            if self.model_type == 'mrbno_v2':
                self.search_space.update({'activation': ['gelu']})
            elif self.model_type == 'mrbno_v2_norm':
                self.search_space.update({'activation': ['gelu'], 'norm_type': ['batch']})
            elif self.model_type == 'multiscale':
                self.search_space.update({
                    'k_in_coarse': [20, 40, 60],
                    'k_out_coarse': [20, 40, 60],
                    'k_in_fine': [80, 120, 160],
                    'k_out_fine': [80, 120, 160],
                })
            elif self.model_type == 'progressive':
                self.search_space.update({
                    'k_in_start': [40, 60],
                    'k_out_start': [40, 60],
                    'k_in_mid': [80, 100],
                    'k_out_mid': [80, 100],
                    'k_in_end': [120, 150],
                    'k_out_end': [120, 150],
                })
            elif self.model_type == 'hybrid_learnable':
                self.search_space.update({'k_in': [80, 100, 120], 'k_out': [80, 100, 120], 'refinement_layers': [1, 2, 3]})
            elif self.model_type == 'progressive_learnable':
                self.search_space.update({'k_in': [80, 100, 120], 'k_out': [80, 100, 120]})
            elif self.model_type == 'adaptive_pca':
                self.search_space.update({'max_k_in': [80, 120, 160], 'max_k_out': [80, 120, 160]})
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Calculate total combinations
        total_combinations = 1
        for values in self.search_space.values():
            total_combinations *= len(values)
        
        print(f"Search space defined for {self.model_type}: {total_combinations:,} total combinations")
        print("Key hyperparameters:")
        for key, values in self.search_space.items():
            print(f"  {key}: {values}")
    
    def _create_pca_reducer(self, config: Dict[str, Any]) -> PCAReducerMultiChannel:
        """Create PCA reducer with given configuration."""
        return PCAReducerMultiChannel(
            bc_weight=config['bc_weight'],
            boundary_threshold=config['boundary_threshold'],
            use_boundary_weighting=config['use_boundary_weighting']
        ).fit(self.X_train, self.Y_train)
    
    def _create_model(self, config: Dict[str, Any], rb: PCAReducerMultiChannel):
        """Create MRBNO model with given configuration."""
        if self.model_type == 'mrbno':
            return MRBNOSimple(
                rb=rb,
                k_in=config['k_in'],
                k_out=config['k_out'],
                layers=config['layers'],
                hidden=config['hidden'],
                dropout=config['dropout'],
                use_skip_connections=config['use_skip_connections'],
                latent_batch_norm=config.get('latent_batch_norm', False)
            )
        elif self.model_type == 'mrbno_v2':
            return MultiChannelMRBNOv2(
                rb=rb,
                k_in=config['k_in'],
                k_out=config['k_out'],
                layers=config['layers'],
                hidden=config['hidden'],
                depth=config['depth'],
                dropout=config['dropout'],
                block0_dropout=config['block0_dropout'],
                use_skip_connections=config['use_skip_connections'],
                activation=config.get('activation', 'gelu')
            )
        elif self.model_type == 'mrbno_v2_norm':
            return MultiChannelMRBNOv2Norm(
                rb=rb,
                k_in=config['k_in'],
                k_out=config['k_out'],
                layers=config['layers'],
                hidden=config['hidden'],
                depth=config['depth'],
                dropout=config['dropout'],
                block0_dropout=config['block0_dropout'],
                use_skip_connections=config['use_skip_connections'],
                activation=config.get('activation', 'gelu'),
                use_norm=config.get('use_norm', True),
                norm_type=config.get('norm_type', 'batch')
            )
        elif self.model_type == 'multiscale':
            return MultiScaleMRBNO(
                rb=rb,
                k_in_coarse=config['k_in_coarse'],
                k_out_coarse=config['k_out_coarse'],
                k_in_fine=config['k_in_fine'],
                k_out_fine=config['k_out_fine'],
                layers=config['layers'],
                hidden=config['hidden'],
                depth=config['depth'],
                dropout=config['dropout'],
                block0_dropout=config['block0_dropout'],
                use_skip_connections=config['use_skip_connections']
            )
        elif self.model_type == 'progressive':
            return ProgressiveMRBNO(
                rb=rb,
                k_in_start=config['k_in_start'],
                k_out_start=config['k_out_start'],
                k_in_mid=config['k_in_mid'],
                k_out_mid=config['k_out_mid'],
                k_in_end=config['k_in_end'],
                k_out_end=config['k_out_end'],
                layers=config['layers'],
                hidden=config['hidden'],
                depth=config['depth'],
                dropout=config['dropout'],
                block0_dropout=config['block0_dropout'],
                use_skip_connections=config['use_skip_connections']
            )
        elif self.model_type == 'pca_attention':
            return PCAAttentionMRBNO(
                rb=rb,
                k_in=config['k_in'],
                k_out=config['k_out'],
                layers=config['layers'],
                hidden=config['hidden'],
                depth=config['depth'],
                dropout=config['dropout'],
                block0_dropout=config['block0_dropout'],
                use_skip_connections=config['use_skip_connections']
            )
        elif self.model_type == 'pca_attention_v2':
            return PCAAttentionMRBNOv2(
                rb=rb,
                k_in=config['k_in'],
                k_out=config['k_out'],
                layers=config['layers'],
                hidden=config['hidden'],
                depth=config['depth'],
                dropout=config['dropout'],
                block0_dropout=config['block0_dropout'],
                use_skip_connections=config['use_skip_connections'],
                attention_heads=config.get('attention_heads', 4)
            )
        elif self.model_type == 'hybrid_learnable':
            return HybridLearnableMRBNO(
                rb=rb,
                k_in=config['k_in'],
                k_out=config['k_out'],
                layers=config['layers'],
                hidden=config['hidden'],
                depth=config['depth'],
                dropout=config['dropout'],
                block0_dropout=config['block0_dropout'],
                use_skip_connections=config['use_skip_connections'],
                refinement_layers=config['refinement_layers']
            )
        elif self.model_type == 'progressive_learnable':
            return ProgressiveLearnableMRBNO(
                rb=rb,
                k_in=config['k_in'],
                k_out=config['k_out'],
                layers=config['layers'],
                hidden=config['hidden'],
                depth=config['depth'],
                dropout=config['dropout'],
                block0_dropout=config['block0_dropout'],
                use_skip_connections=config['use_skip_connections']
            )
        elif self.model_type == 'adaptive_pca':
            return AdaptivePCAMRBNO(
                rb=rb,
                max_k_in=config['max_k_in'],
                max_k_out=config['max_k_out'],
                layers=config['layers'],
                hidden=config['hidden'],
                depth=config['depth'],
                dropout=config['dropout'],
                block0_dropout=config['block0_dropout'],
                use_skip_connections=config['use_skip_connections']
            )
        elif self.model_type == 'learnable':
            return LearnableMultiChannelMRBNOv2(
                rb_shape=rb.shape_,
                n_channels=rb.n_channels,
                k_in=config['k_in'],
                k_out=config['k_out'],
                layers=config['layers'],
                hidden=config['hidden'],
                depth=config['depth'],
                dropout=config['dropout'],
                block0_dropout=config['block0_dropout'],
                use_skip_connections=config['use_skip_connections']
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _train_model(self, model, config: Dict[str, Any], epochs: int = 100) -> Tuple[float, float, float]:
        """
        Train a single model configuration.
        
        Returns:
            (best_train_loss, best_val_loss, final_val_loss)
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
        
        # Always use AdamW optimizer (better weight decay handling and more robust)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Always use warmup scheduler for better convergence and reduced initialization sensitivity
        warmup_epochs = config.get('warmup_epochs', 10)
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        
        # Gradient clipping threshold from config (always use, but with different values)
        max_grad_norm = config.get('max_grad_norm', 5.0)
        
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
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            scheduler.step()
        
        return best_train_loss, best_val_loss, val_loss
    
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
                # Create PCA reducer
                rb = self._create_pca_reducer(config)
                
                # Create model
                model = self._create_model(config, rb)
                
                # Count parameters
                n_params = sum(p.numel() for p in model.parameters())
                
                # Train model
                train_loss, val_loss, final_val_loss = self._train_model(model, config, epochs_per_config)
                
                # Store results
                result = {
                    'config': config,
                    'n_params': n_params,
                    'best_train_loss': train_loss,
                    'best_val_loss': val_loss,
                    'final_val_loss': final_val_loss,
                    'train_val_gap': val_loss - train_loss,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.results.append(result)
                
                print(f"Results: train={train_loss:.4e}, val={val_loss:.4e}, params={n_params:,}")
                
                # Save intermediate results
                if (i + 1) % save_frequency == 0:
                    self._save_results(f"intermediate_results_{i+1}.json")
                
                # Clean up model and reducer to free GPU memory
                del model, rb
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
                    # Create PCA reducer
                    rb = self._create_pca_reducer(config)
                    
                    # Create model
                    model = self._create_model(config, rb)
                    
                    # Count parameters
                    n_params = sum(p.numel() for p in model.parameters())
                    
                    # Train model
                    train_loss, val_loss, final_val_loss = self._train_model(model, config, epochs_per_config)
                    
                    # Store results
                    result = {
                        'config': config,
                        'n_params': n_params,
                        'best_train_loss': train_loss,
                        'best_val_loss': val_loss,
                        'final_val_loss': final_val_loss,
                        'train_val_gap': val_loss - train_loss,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.results.append(result)
                    
                    # Clean up model and reducer
                    del model, rb
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    print(f"Results: train={train_loss:.4e}, val={val_loss:.4e}, params={n_params:,}")
                    
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
        """Generate random configurations."""
        configs = []
        attempts = 0
        max_attempts = max_configs * 10
        
        while len(configs) < max_configs and attempts < max_attempts:
            config = {}
            for key, values in self.search_space.items():
                config[key] = np.random.choice(values)
            
            # Validate configuration for pca_attention_v2
            if self.model_type == 'pca_attention_v2':
                k_out = config['k_out']
                attention_heads = config['attention_heads']
                if k_out % attention_heads != 0:
                    attempts += 1
                    continue
            
            # Convert numpy types to regular Python types
            config = self._convert_numpy_types(config)
            configs.append(config)
            attempts += 1
        
        if len(configs) < max_configs:
            print(f"Warning: Only generated {len(configs)} valid configs out of {max_configs} requested")
        
        return configs
    
    def _generate_grid_configs(self, max_configs: int) -> List[Dict[str, Any]]:
        """Generate grid search configurations."""
        # Create all combinations
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        
        configs = []
        for combination in itertools.product(*values):
            config = dict(zip(keys, combination))
            # Convert numpy types to regular Python types
            config = self._convert_numpy_types(config)
            configs.append(config)
            
            if len(configs) >= max_configs:
                break
        
        return configs
    
    def _config_to_vector(self, config: Dict[str, Any]) -> np.ndarray:
        """Convert configuration to numerical vector for Bayesian optimization."""
        vector = []
        
        if self.model_type == 'mrbno':
            vector.append(config['layers'])
            vector.append(config['hidden'])
            vector.append(config['dropout'])
            vector.append(1 if config.get('latent_batch_norm', False) else 0)
        else:
            common_params = ['layers', 'hidden', 'depth', 'dropout', 'block0_dropout']
            for key in common_params:
                vector.append(config[key])
        
        vector.append(config.get('warmup_epochs', 10))
        vector.append(config.get('max_grad_norm', 5.0))
        
        if self.model_type in ['mrbno', 'pca_attention', 'learnable']:
            vector.extend([config['k_in'], config['k_out']])
        elif self.model_type == 'mrbno_v2':
            vector.extend([config['k_in'], config['k_out']])
            activation_map = {'gelu': 0, 'lrelu': 1}
            vector.append(activation_map.get(config.get('activation', 'gelu'), 0))
        elif self.model_type == 'mrbno_v2_norm':
            vector.extend([config['k_in'], config['k_out']])
            activation_map = {'gelu': 0, 'lrelu': 1}
            norm_type_map = {'batch': 0, 'layer': 1}
            vector.append(activation_map.get(config.get('activation', 'gelu'), 0))
            vector.append(norm_type_map.get(config.get('norm_type', 'batch'), 0))
        elif self.model_type == 'pca_attention_v2':
            vector.extend([config['k_in'], config['k_out'], config.get('attention_heads', 4)])
        elif self.model_type == 'multiscale':
            vector.extend([config['k_in_coarse'], config['k_out_coarse'], 
                         config['k_in_fine'], config['k_out_fine']])
        elif self.model_type == 'progressive':
            vector.extend([config['k_in_start'], config['k_out_start'],
                         config['k_in_mid'], config['k_out_mid'],
                         config['k_in_end'], config['k_out_end']])
        elif self.model_type == 'hybrid_learnable':
            vector.extend([config['k_in'], config['k_out'], config.get('refinement_layers', 2)])
        elif self.model_type == 'progressive_learnable':
            vector.extend([config['k_in'], config['k_out']])
        elif self.model_type == 'adaptive_pca':
            vector.extend([config['max_k_in'], config['max_k_out']])
        
        return np.array(vector)
    
    def _vector_to_config(self, vector: np.ndarray) -> Dict[str, Any]:
        """Convert numerical vector back to configuration."""
        config = {}
        offset = 0
        
        if self.model_type == 'mrbno':
            layers_values = self.search_space['layers']
            hidden_values = self.search_space['hidden']
            dropout_values = self.search_space['dropout']
            bn_values = self.search_space['latent_batch_norm']

            idx_layers = np.argmin(np.abs(np.array(layers_values) - vector[offset])); config['layers'] = layers_values[idx_layers]; offset += 1
            idx_hidden = np.argmin(np.abs(np.array(hidden_values) - vector[offset])); config['hidden'] = hidden_values[idx_hidden]; offset += 1
            idx_dropout = np.argmin(np.abs(np.array(dropout_values) - vector[offset])); config['dropout'] = dropout_values[idx_dropout]; offset += 1
            bn_choice = bn_values[int(round(vector[offset])) % len(bn_values)]; config['latent_batch_norm'] = bn_choice; offset += 1
        else:
            common_params = ['layers', 'hidden', 'depth', 'dropout', 'block0_dropout']
            for key in common_params:
                values = self.search_space[key]
                idx = np.argmin(np.abs(np.array(values) - vector[offset]))
                config[key] = values[idx]
                offset += 1
        
        warmup_values = self.search_space['warmup_epochs']
        grad_norm_values = self.search_space['max_grad_norm']
        idx_warmup = np.argmin(np.abs(np.array(warmup_values) - vector[offset])); config['warmup_epochs'] = warmup_values[idx_warmup]; offset += 1
        idx_grad = np.argmin(np.abs(np.array(grad_norm_values) - vector[offset])); config['max_grad_norm'] = grad_norm_values[idx_grad]; offset += 1
        
        if self.model_type in ['mrbno', 'pca_attention', 'learnable']:
            k_params = ['k_in', 'k_out']
            for i, key in enumerate(k_params):
                values = self.search_space[key]
                idx = np.argmin(np.abs(np.array(values) - vector[offset + i]))
                config[key] = values[idx]
            offset += len(k_params)
        elif self.model_type == 'mrbno_v2':
            k_params = ['k_in', 'k_out']
            for i, key in enumerate(k_params):
                values = self.search_space[key]
                idx = np.argmin(np.abs(np.array(values) - vector[offset + i]))
                config[key] = values[idx]
            offset += len(k_params)
            activation_map = {0: 'gelu', 1: 'lrelu'}
            config['activation'] = activation_map.get(int(round(vector[offset])), 'gelu'); offset += 1
        elif self.model_type == 'mrbno_v2_norm':
            k_params = ['k_in', 'k_out']
            for i, key in enumerate(k_params):
                values = self.search_space[key]
                idx = np.argmin(np.abs(np.array(values) - vector[offset + i]))
                config[key] = values[idx]
            offset += len(k_params)
            activation_map = {0: 'gelu', 1: 'lrelu'}
            norm_type_map = {0: 'batch', 1: 'layer'}
            config['activation'] = activation_map.get(int(round(vector[offset])), 'gelu'); offset += 1
            config['norm_type'] = norm_type_map.get(int(round(vector[offset])), 'batch'); offset += 1
        elif self.model_type == 'pca_attention_v2':
            k_params = ['k_in', 'k_out', 'attention_heads']
            for i, key in enumerate(k_params):
                values = self.search_space[key]
                idx = np.argmin(np.abs(np.array(values) - vector[offset + i]))
                config[key] = values[idx]
            offset += len(k_params)
        elif self.model_type == 'multiscale':
            k_params = ['k_in_coarse', 'k_out_coarse', 'k_in_fine', 'k_out_fine']
            for i, key in enumerate(k_params):
                values = self.search_space[key]
                idx = np.argmin(np.abs(np.array(values) - vector[offset + i]))
                config[key] = values[idx]
            offset += len(k_params)
        elif self.model_type == 'progressive':
            k_params = ['k_in_start', 'k_out_start', 'k_in_mid', 'k_out_mid', 'k_in_end', 'k_out_end']
            for i, key in enumerate(k_params):
                values = self.search_space[key]
                idx = np.argmin(np.abs(np.array(values) - vector[offset + i]))
                config[key] = values[idx]
            offset += len(k_params)
        elif self.model_type == 'hybrid_learnable':
            k_params = ['k_in', 'k_out', 'refinement_layers']
            for i, key in enumerate(k_params):
                values = self.search_space[key]
                idx = np.argmin(np.abs(np.array(values) - vector[offset + i]))
                config[key] = values[idx]
            offset += len(k_params)
        elif self.model_type == 'progressive_learnable':
            k_params = ['k_in', 'k_out']
            for i, key in enumerate(k_params):
                values = self.search_space[key]
                idx = np.argmin(np.abs(np.array(values) - vector[offset + i]))
                config[key] = values[idx]
            offset += len(k_params)
        elif self.model_type == 'adaptive_pca':
            k_params = ['max_k_in', 'max_k_out']
            for i, key in enumerate(k_params):
                values = self.search_space[key]
                idx = np.argmin(np.abs(np.array(values) - vector[offset + i]))
                config[key] = values[idx]
            offset += len(k_params)
        
        config.setdefault('weight_decay', self.search_space.get('weight_decay', [1e-4])[0])
        config.setdefault('lr', self.search_space.get('lr', [3e-3])[0])
        config.setdefault('batch_size', self.search_space.get('batch_size', [16])[0])
        config.setdefault('use_boundary_weighting', self.search_space.get('use_boundary_weighting', [False])[0])
        config.setdefault('bc_weight', self.search_space.get('bc_weight', [1.0])[0])
        config.setdefault('boundary_threshold', self.search_space.get('boundary_threshold', [0.1])[0])
        config.setdefault('use_skip_connections', self.search_space.get('use_skip_connections', [True])[0])
        
        config = self._convert_numpy_types(config)
        return config
    
    def _acquisition_function(self, x: np.ndarray, gp: GaussianProcessRegressor, 
                            best_val_loss: float, exploration_weight: float = 0.1) -> float:
        """Expected Improvement acquisition function."""
        mean, std = gp.predict(x.reshape(1, -1), return_std=True)
        improvement = best_val_loss - mean[0]
        return improvement + exploration_weight * std[0]
    
    def _bayesian_optimization_step(self, n_candidates: int = 5) -> List[Dict[str, Any]]:
        """Generate next configurations using Bayesian optimization."""
        if len(self.results) < 5:  # Need at least 5 points for GP
            return self._generate_random_configs(n_candidates)
        
        # Prepare data for GP
        X = []
        y = []
        for result in self.results:
            config_vector = self._config_to_vector(result['config'])
            X.append(config_vector)
            y.append(result['best_val_loss'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Fit Gaussian Process
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
        gp.fit(X, y)
        
        # Find best configuration so far
        best_val_loss = min(y)
        
        # Generate candidates using acquisition function
        candidates = []
        for _ in range(n_candidates * 10):  # Generate more candidates than needed
            config_vector = []
            
            if self.model_type == 'mrbno':
                config_vector.append(np.random.choice(self.search_space['layers']))
                config_vector.append(np.random.choice(self.search_space['hidden']))
                config_vector.append(np.random.choice(self.search_space['dropout']))
                bn_choice = np.random.choice(self.search_space['latent_batch_norm'])
                config_vector.append(1 if bn_choice else 0)
            else:
                common_params = ['layers', 'hidden', 'depth', 'dropout', 'block0_dropout']
                for key in common_params:
                    values = self.search_space[key]
                    config_vector.append(np.random.choice(values))
            
            config_vector.append(np.random.choice(self.search_space['warmup_epochs']))
            config_vector.append(np.random.choice(self.search_space['max_grad_norm']))
            
            if self.model_type in ['mrbno', 'pca_attention', 'learnable']:
                k_params = ['k_in', 'k_out']
                for key in k_params:
                    values = self.search_space[key]
                    config_vector.append(np.random.choice(values))
            elif self.model_type == 'mrbno_v2':
                k_params = ['k_in', 'k_out']
                for key in k_params:
                    values = self.search_space[key]
                    config_vector.append(np.random.choice(values))
                # Add activation parameter (categorical -> numerical)
                activation_values = self.search_space['activation']
                activation_map = {'gelu': 0, 'lrelu': 1}
                activation_choice = np.random.choice(activation_values)
                config_vector.append(activation_map[activation_choice])
            elif self.model_type == 'mrbno_v2_norm':
                k_params = ['k_in', 'k_out']
                for key in k_params:
                    values = self.search_space[key]
                    config_vector.append(np.random.choice(values))
                # Add activation and norm_type parameters (categorical -> numerical)
                activation_values = self.search_space['activation']
                norm_type_values = self.search_space['norm_type']
                activation_map = {'gelu': 0, 'lrelu': 1}
                norm_type_map = {'batch': 0, 'layer': 1}
                activation_choice = np.random.choice(activation_values)
                norm_type_choice = np.random.choice(norm_type_values)
                config_vector.append(activation_map[activation_choice])
                config_vector.append(norm_type_map[norm_type_choice])
            elif self.model_type == 'pca_attention_v2':
                k_params = ['k_in', 'k_out', 'attention_heads']
                for key in k_params:
                    values = self.search_space[key]
                    config_vector.append(np.random.choice(values))
            elif self.model_type == 'multiscale':
                k_params = ['k_in_coarse', 'k_out_coarse', 'k_in_fine', 'k_out_fine']
                for key in k_params:
                    values = self.search_space[key]
                    config_vector.append(np.random.choice(values))
            elif self.model_type == 'progressive':
                k_params = ['k_in_start', 'k_out_start', 'k_in_mid', 'k_out_mid', 'k_in_end', 'k_out_end']
                for key in k_params:
                    values = self.search_space[key]
                    config_vector.append(np.random.choice(values))
            elif self.model_type == 'hybrid_learnable':
                k_params = ['k_in', 'k_out', 'refinement_layers']
                for key in k_params:
                    values = self.search_space[key]
                    config_vector.append(np.random.choice(values))
            elif self.model_type == 'progressive_learnable':
                k_params = ['k_in', 'k_out']
                for key in k_params:
                    values = self.search_space[key]
                    config_vector.append(np.random.choice(values))
            elif self.model_type == 'adaptive_pca':
                k_params = ['max_k_in', 'max_k_out']
                for key in k_params:
                    values = self.search_space[key]
                    config_vector.append(np.random.choice(values))
            
            config_vector = np.array(config_vector)
            
            # Calculate acquisition value
            acq_value = self._acquisition_function(config_vector, gp, best_val_loss)
            candidates.append((acq_value, config_vector))
        
        # Select top candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected_configs = []
        for _, config_vector in candidates[:n_candidates]:
            config = self._vector_to_config(config_vector)
            # Ensure types are converted (should already be done in _vector_to_config)
            config = self._convert_numpy_types(config)
            
            # Validate and adjust k_out for pca_attention_v2
            if self.model_type == 'pca_attention_v2':
                k_out = config['k_out']
                attention_heads = config['attention_heads']
                if k_out % attention_heads != 0:
                    # Adjust k_out to nearest valid value
                    k_out_adjusted = ((k_out // attention_heads) + 1) * attention_heads
                    config['k_out'] = k_out_adjusted
                    # Ensure it's within search space bounds
                    max_k_out = max(self.search_space['k_out'])
                    if k_out_adjusted > max_k_out:
                        config['k_out'] = max_k_out
        
            selected_configs.append(config)
        
        return selected_configs
    
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
        """Create comprehensive plots of sweep results."""
        if not self.results:
            print("No results to plot")
            return
        
        print("Creating plots...")
        
        # Convert to DataFrame for easier plotting
        df_data = []
        for result in self.results:
            row = result['config'].copy()
            row.update({
                'val_loss': result['best_val_loss'],
                'train_loss': result['best_train_loss'],
                'train_val_gap': result['train_val_gap'],
                'n_params': result['n_params']
            })
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.model_type.upper()} Hyperparameter Sweep Results', fontsize=16)
        
        # 1. Validation loss vs PCA ranks (model-specific)
        ax1 = axes[0, 0]
        if self.model_type in ['mrbno', 'mrbno_v2', 'mrbno_v2_norm', 'pca_attention', 'learnable']:
            scatter = ax1.scatter(df['k_in'], df['k_out'], c=df['val_loss'], 
                                 cmap='viridis', s=100, alpha=0.7)
            ax1.set_xlabel('k_in (Input PCA Components)')
            ax1.set_ylabel('k_out (Output PCA Components)')
            ax1.set_title('Validation Loss vs PCA Ranks')
        elif self.model_type == 'multiscale':
            scatter = ax1.scatter(df['k_in_coarse'], df['k_out_coarse'], c=df['val_loss'], 
                                 cmap='viridis', s=100, alpha=0.7)
            ax1.set_xlabel('k_in_coarse (Coarse Input PCA)')
            ax1.set_ylabel('k_out_coarse (Coarse Output PCA)')
            ax1.set_title('Validation Loss vs Coarse PCA Ranks')
        elif self.model_type == 'progressive':
            scatter = ax1.scatter(df['k_in_start'], df['k_out_start'], c=df['val_loss'], 
                                 cmap='viridis', s=100, alpha=0.7)
            ax1.set_xlabel('k_in_start (Start Input PCA)')
            ax1.set_ylabel('k_out_start (Start Output PCA)')
            ax1.set_title('Validation Loss vs Start PCA Ranks')
        elif self.model_type in ['hybrid_learnable', 'progressive_learnable']:
            scatter = ax1.scatter(df['k_in'], df['k_out'], c=df['val_loss'], 
                                 cmap='viridis', s=100, alpha=0.7)
            ax1.set_xlabel('k_in (Input PCA Components)')
            ax1.set_ylabel('k_out (Output PCA Components)')
            ax1.set_title('Validation Loss vs PCA Ranks')
        elif self.model_type == 'adaptive_pca':
            scatter = ax1.scatter(df['max_k_in'], df['max_k_out'], c=df['val_loss'], 
                                 cmap='viridis', s=100, alpha=0.7)
            ax1.set_xlabel('max_k_in (Max Input PCA Components)')
            ax1.set_ylabel('max_k_out (Max Output PCA Components)')
            ax1.set_title('Validation Loss vs Max PCA Ranks')
        else:
            # Default case for any other model types
            scatter = ax1.scatter(df['k_in'], df['k_out'], c=df['val_loss'], 
                                 cmap='viridis', s=100, alpha=0.7)
            ax1.set_xlabel('k_in (Input PCA Components)')
            ax1.set_ylabel('k_out (Output PCA Components)')
            ax1.set_title('Validation Loss vs PCA Ranks')
        plt.colorbar(scatter, ax=ax1, label='Validation Loss')
        
        # 2. Validation loss vs Architecture
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(df['layers'], df['hidden'], c=df['val_loss'], 
                              cmap='viridis', s=100, alpha=0.7)
        ax2.set_xlabel('Number of Layers')
        ax2.set_ylabel('Hidden Dimension')
        ax2.set_title('Validation Loss vs Architecture')
        plt.colorbar(scatter2, ax=ax2, label='Validation Loss')
        
        # 3. Validation loss vs Regularization
        ax3 = axes[0, 2]
        if 'block0_dropout' in df.columns:
            reg_y = df['block0_dropout']
            ax3.set_ylabel('Block0 Dropout')
        else:
            reg_y = df['latent_batch_norm'].astype(int)
            ax3.set_ylabel('Latent BatchNorm (0/1)')
        scatter3 = ax3.scatter(df['dropout'], reg_y, c=df['val_loss'],
                              cmap='viridis', s=100, alpha=0.7)
        ax3.set_xlabel('Dropout')
        ax3.set_title('Validation Loss vs Regularization')
        plt.colorbar(scatter3, ax=ax3, label='Validation Loss')
        
        # 4. Train-Val Gap vs Parameters
        ax4 = axes[1, 0]
        scatter4 = ax4.scatter(df['n_params'], df['train_val_gap'], c=df['val_loss'], 
                              cmap='viridis', s=100, alpha=0.7)
        ax4.set_xlabel('Number of Parameters')
        ax4.set_ylabel('Train-Val Gap')
        ax4.set_title('Overfitting vs Model Size')
        plt.colorbar(scatter4, ax=ax4, label='Validation Loss')
        
        # 5. Validation loss distribution
        ax5 = axes[1, 1]
        ax5.hist(df['val_loss'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.set_xlabel('Validation Loss')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Validation Loss Distribution')
        ax5.axvline(df['val_loss'].min(), color='red', linestyle='--', label=f'Best: {df["val_loss"].min():.4e}')
        ax5.legend()
        
        # 6. Top 10 configurations
        ax6 = axes[1, 2]
        top_10 = df.head(10)
        bars = ax6.barh(range(len(top_10)), top_10['val_loss'], color='lightcoral')
        ax6.set_yticks(range(len(top_10)))
        
        # Create labels based on model type
        if self.model_type in ['mrbno', 'mrbno_v2', 'mrbno_v2_norm', 'pca_attention', 'learnable']:
            labels = [f"k_in={row['k_in']}, k_out={row['k_out']}" for _, row in top_10.iterrows()]
        elif self.model_type == 'multiscale':
            labels = [f"k_coarse={row['k_in_coarse']}, k_fine={row['k_in_fine']}" for _, row in top_10.iterrows()]
        elif self.model_type == 'progressive':
            labels = [f"k_start={row['k_in_start']}, k_end={row['k_out_end']}" for _, row in top_10.iterrows()]
        elif self.model_type in ['hybrid_learnable', 'progressive_learnable']:
            labels = [f"k_in={row['k_in']}, k_out={row['k_out']}" for _, row in top_10.iterrows()]
        elif self.model_type == 'adaptive_pca':
            labels = [f"max_k_in={row['max_k_in']}, max_k_out={row['max_k_out']}" for _, row in top_10.iterrows()]
        else:
            # Default case
            labels = [f"k_in={row['k_in']}, k_out={row['k_out']}" for _, row in top_10.iterrows()]
        
        ax6.set_yticklabels(labels)
        ax6.set_xlabel('Validation Loss')
        ax6.set_title('Top 10 Configurations')
        ax6.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sweep_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plots saved to {self.output_dir}/")
    
    def _print_summary(self):
        """Print summary of best configurations."""
        if not self.results:
            print("No results to summarize")
            return
        
        # Sort by validation loss
        sorted_results = sorted(self.results, key=lambda x: x['best_val_loss'])
        
        print(f"\n{'='*80}")
        print(f"{self.model_type.upper()} HYPERPARAMETER SWEEP SUMMARY")
        print(f"{'='*80}")
        
        print(f"\nTop 10 configurations by validation loss:")
        print(f"{'Rank':<4} {'Val Loss':<12} {'Train Loss':<12} {'Gap':<12} {'Params':<10} {'Key Params'}")
        print("-" * 80)
        
        for i, result in enumerate(sorted_results[:10]):
            config = result['config']
            if self.model_type == 'mrbno':
                key_params = (f"k_in={config['k_in']}, k_out={config['k_out']}, "
                              f"layers={config['layers']}, hidden={config['hidden']}, "
                              f"bn={config.get('latent_batch_norm', False)}")
            elif self.model_type in ['mrbno_v2', 'mrbno_v2_norm', 'pca_attention', 'learnable']:
                key_params = f"k_in={config['k_in']}, k_out={config['k_out']}, layers={config['layers']}, hidden={config['hidden']}"
            elif self.model_type == 'multiscale':
                key_params = f"k_coarse={config['k_in_coarse']}, k_fine={config['k_in_fine']}, layers={config['layers']}, hidden={config['hidden']}"
            elif self.model_type == 'progressive':
                key_params = f"k_start={config['k_in_start']}, k_end={config['k_out_end']}, layers={config['layers']}, hidden={config['hidden']}"
            elif self.model_type in ['hybrid_learnable', 'progressive_learnable']:
                key_params = f"k_in={config['k_in']}, k_out={config['k_out']}, layers={config['layers']}, hidden={config['hidden']}"
            elif self.model_type == 'adaptive_pca':
                key_params = f"max_k_in={config['max_k_in']}, max_k_out={config['max_k_out']}, layers={config['layers']}, hidden={config['hidden']}"
            else:
                # Default case
                key_params = f"k_in={config['k_in']}, k_out={config['k_out']}, layers={config['layers']}, hidden={config['hidden']}"
            print(f"{i+1:<4} {result['best_val_loss']:<12.4e} {result['best_train_loss']:<12.4e} "
                  f"{result['train_val_gap']:<12.4e} {result['n_params']:<10,} {key_params}")
        
        # Analyze patterns
        print(f"\n{'='*80}")
        print("PATTERN ANALYSIS")
        print(f"{'='*80}")
        
        # Best configurations
        best_configs = sorted_results[:20]
        
        # Model-specific analysis
        if self.model_type in ['mrbno', 'mrbno_v2', 'mrbno_v2_norm', 'pca_attention', 'learnable']:
            k_in_values = [r['config']['k_in'] for r in best_configs]
            k_out_values = [r['config']['k_out'] for r in best_configs]
            print(f"\nBest k_in values (top 20): {sorted(set(k_in_values))}")
            print(f"Best k_out values (top 20): {sorted(set(k_out_values))}")
        elif self.model_type == 'multiscale':
            k_coarse_in = [r['config']['k_in_coarse'] for r in best_configs]
            k_coarse_out = [r['config']['k_out_coarse'] for r in best_configs]
            k_fine_in = [r['config']['k_in_fine'] for r in best_configs]
            k_fine_out = [r['config']['k_out_fine'] for r in best_configs]
            print(f"\nBest k_in_coarse values (top 20): {sorted(set(k_coarse_in))}")
            print(f"Best k_out_coarse values (top 20): {sorted(set(k_coarse_out))}")
            print(f"Best k_in_fine values (top 20): {sorted(set(k_fine_in))}")
            print(f"Best k_out_fine values (top 20): {sorted(set(k_fine_out))}")
        elif self.model_type == 'progressive':
            k_start_in = [r['config']['k_in_start'] for r in best_configs]
            k_start_out = [r['config']['k_out_start'] for r in best_configs]
            k_mid_in = [r['config']['k_in_mid'] for r in best_configs]
            k_mid_out = [r['config']['k_out_mid'] for r in best_configs]
            k_end_in = [r['config']['k_in_end'] for r in best_configs]
            k_end_out = [r['config']['k_out_end'] for r in best_configs]
            print(f"\nBest k_in_start values (top 20): {sorted(set(k_start_in))}")
            print(f"Best k_out_start values (top 20): {sorted(set(k_start_out))}")
            print(f"Best k_in_mid values (top 20): {sorted(set(k_mid_in))}")
            print(f"Best k_out_mid values (top 20): {sorted(set(k_mid_out))}")
            print(f"Best k_in_end values (top 20): {sorted(set(k_end_in))}")
            print(f"Best k_out_end values (top 20): {sorted(set(k_end_out))}")
        
        # Best architecture
        layer_values = [r['config']['layers'] for r in best_configs]
        hidden_values = [r['config']['hidden'] for r in best_configs]
        
        print(f"Best layer counts (top 20): {sorted(set(layer_values))}")
        print(f"Best hidden sizes (top 20): {sorted(set(hidden_values))}")
        
        # Best regularization
        dropout_values = [r['config']['dropout'] for r in best_configs]
        print(f"Best dropout values (top 20): {sorted(set(dropout_values))}")


def main():
    parser = argparse.ArgumentParser(description='MRBNO Hyperparameter Sweep')
    parser.add_argument('--data_path', default='data/nonlinear_diffusion_dataset.npz',
                       help='Path to dataset')
    parser.add_argument('--output_dir', default='sweep_results',
                       help='Output directory for results')
    parser.add_argument('--max_configs', type=int, default=200,
                       help='Maximum configurations to test')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Epochs per configuration')
    parser.add_argument('--search_strategy', default='bayesian',
                       choices=['random', 'grid', 'bayesian'],
                       help='Search strategy: random, grid, or bayesian')
    parser.add_argument('--device', default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--model_type', default='mrbno',
                       choices=['mrbno', 'mrbno_v2', 'mrbno_v2_norm', 'multiscale', 'progressive', 'pca_attention', 'pca_attention_v2', 'hybrid_learnable', 'progressive_learnable', 'adaptive_pca', 'learnable'],
                       help='Model type: mrbno, mrbno_v2, mrbno_v2_norm, multiscale, progressive, pca_attention, pca_attention_v2, hybrid_learnable, progressive_learnable, adaptive_pca, or learnable')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (None = random)')
    
    args = parser.parse_args()
    
    # Create and run sweep
    sweep = HyperparameterSweep(
        data_path=args.data_path,
        output_dir=args.output_dir,
        device=args.device,
        model_type=args.model_type,
        seed=args.seed
    )
    
    sweep.run_sweep(
        max_configs=args.max_configs,
        epochs_per_config=args.epochs,
        search_strategy=args.search_strategy
    )


if __name__ == '__main__':
    main()
