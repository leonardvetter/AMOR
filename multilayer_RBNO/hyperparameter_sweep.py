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
from MRBNO_model_multichannel import PCAReducerMultiChannel, MultiLayerReducedBasisResNetMultiChannel

class HyperparameterSweep:
    def __init__(self, data_path: str = 'data/nonlinear_diffusion_dataset.npz', 
                 output_dir: str = 'sweep_results', device: str = 'auto'):
        """
        Initialize hyperparameter sweep.
        
        Args:
            data_path: Path to the nonlinear diffusion dataset
            output_dir: Directory to save sweep results
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.device = self._setup_device(device)
        
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
        
        # Normalize
        x_mean = X.mean()
        x_std = X.std()
        y_mean = Y.mean()
        y_std = Y.std()
        
        Xn = (X - x_mean) / (x_std + 1e-8)
        Yn = (Y - y_mean) / (y_std + 1e-8)
        
        # Train/val split
        train_frac = 0.7
        n_train = int(train_frac * len(Xn))
        
        self.X_train, self.Y_train = Xn[:n_train], Yn[:n_train]
        self.X_val, self.Y_val = Xn[n_train:], Yn[n_train:]
        
        # Convert to PyTorch tensors
        self.X_train_t = torch.from_numpy(self.X_train).float()
        self.Y_train_t = torch.from_numpy(self.Y_train).float().unsqueeze(1)
        self.X_val_t = torch.from_numpy(self.X_val).float()
        self.Y_val_t = torch.from_numpy(self.Y_val).float().unsqueeze(1)
        
        print(f"Data loaded: {len(self.X_train)} train, {len(self.X_val)} val samples")
        print(f"Using device: {self.device}")
    
    def _define_search_space(self):
        """Define the hyperparameter search space."""
        self.search_space = {
            # PCA ranks (most important) - expanded range up to 200
            'k_in': [20, 40, 60, 80, 100, 120, 150, 180, 200],
            'k_out': [10, 20, 30, 40, 50, 60, 75, 90, 100],
            
            # Architecture
            'layers': [2, 3, 4, 5],
            'hidden': [64, 128, 256, 512],
            'depth': [1, 2, 3, 4],
            
            # Regularization
            'dropout': [0.0, 0.1, 0.2, 0.3],
            'block0_dropout': [0.0, 0.1, 0.2],
            'weight_decay': [1e-5, 1e-4, 1e-3],
            
            # Training
            'lr': [1e-3, 3e-3, 5e-3],
            'batch_size': [16, 32],
            
            # Boundary weighting
            'use_boundary_weighting': [False, True],
            'bc_weight': [1.0, 1.5, 2.0],
            'boundary_threshold': [0.05, 0.1, 0.15],
            
            # Skip connections
            'use_skip_connections': [True, False],
        }
        
        # Calculate total combinations
        total_combinations = 1
        for values in self.search_space.values():
            total_combinations *= len(values)
        
        print(f"Search space defined: {total_combinations:,} total combinations")
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
    
    def _create_model(self, config: Dict[str, Any], rb: PCAReducerMultiChannel) -> MultiLayerReducedBasisResNetMultiChannel:
        """Create MRBNO model with given configuration."""
        return MultiLayerReducedBasisResNetMultiChannel(
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
    
    def _train_model(self, model: MultiLayerReducedBasisResNetMultiChannel, 
                    config: Dict[str, Any], epochs: int = 100) -> Tuple[float, float, float]:
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
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)
            
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
                
            except Exception as e:
                print(f"Error with config {config}: {e}")
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
                    
                    print(f"Results: train={train_loss:.4e}, val={val_loss:.4e}, params={n_params:,}")
                    
                    # Save intermediate results
                    if (config_idx) % save_frequency == 0:
                        self._save_results(f"intermediate_results_{config_idx}.json")
                    
                except Exception as e:
                    print(f"Error with config {config}: {e}")
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
        for _ in range(max_configs):
            config = {}
            for key, values in self.search_space.items():
                config[key] = np.random.choice(values)
            # Convert numpy types to regular Python types
            config = self._convert_numpy_types(config)
            configs.append(config)
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
        for key in ['k_in', 'k_out', 'layers', 'hidden', 'depth', 'dropout', 'block0_dropout']:
            vector.append(config[key])
        return np.array(vector)
    
    def _vector_to_config(self, vector: np.ndarray) -> Dict[str, Any]:
        """Convert numerical vector back to configuration."""
        keys = ['k_in', 'k_out', 'layers', 'hidden', 'depth', 'dropout', 'block0_dropout']
        config = {}
        for i, key in enumerate(keys):
            values = self.search_space[key]
            # Find closest value
            idx = np.argmin(np.abs(np.array(values) - vector[i]))
            config[key] = values[idx]
        
        # Add other parameters with default values
        config.update({
            'weight_decay': 1e-4,
            'lr': 3e-3,
            'batch_size': 16,
            'use_boundary_weighting': False,
            'bc_weight': 1.0,
            'boundary_threshold': 0.1,
            'use_skip_connections': True
        })
        
        # Convert numpy types to regular Python types
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
            # Random sampling in the search space
            config_vector = []
            for key in ['k_in', 'k_out', 'layers', 'hidden', 'depth', 'dropout', 'block0_dropout']:
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
        fig.suptitle('MRBNO Hyperparameter Sweep Results', fontsize=16)
        
        # 1. Validation loss vs PCA ranks
        ax1 = axes[0, 0]
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
        scatter3 = ax3.scatter(df['dropout'], df['block0_dropout'], c=df['val_loss'], 
                              cmap='viridis', s=100, alpha=0.7)
        ax3.set_xlabel('Dropout')
        ax3.set_ylabel('Block0 Dropout')
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
        ax6.set_yticklabels([f"k_in={row['k_in']}, k_out={row['k_out']}" for _, row in top_10.iterrows()])
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
        print("HYPERPARAMETER SWEEP SUMMARY")
        print(f"{'='*80}")
        
        print(f"\nTop 10 configurations by validation loss:")
        print(f"{'Rank':<4} {'Val Loss':<12} {'Train Loss':<12} {'Gap':<12} {'Params':<10} {'Key Params'}")
        print("-" * 80)
        
        for i, result in enumerate(sorted_results[:10]):
            config = result['config']
            key_params = f"k_in={config['k_in']}, k_out={config['k_out']}, layers={config['layers']}, hidden={config['hidden']}"
            print(f"{i+1:<4} {result['best_val_loss']:<12.4e} {result['best_train_loss']:<12.4e} "
                  f"{result['train_val_gap']:<12.4e} {result['n_params']:<10,} {key_params}")
        
        # Analyze patterns
        print(f"\n{'='*80}")
        print("PATTERN ANALYSIS")
        print(f"{'='*80}")
        
        # Best k_in, k_out
        best_configs = sorted_results[:20]
        k_in_values = [r['config']['k_in'] for r in best_configs]
        k_out_values = [r['config']['k_out'] for r in best_configs]
        
        print(f"\nBest k_in values (top 20): {sorted(set(k_in_values))}")
        print(f"Best k_out values (top 20): {sorted(set(k_out_values))}")
        
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
    
    args = parser.parse_args()
    
    # Create and run sweep
    sweep = HyperparameterSweep(
        data_path=args.data_path,
        output_dir=args.output_dir,
        device=args.device
    )
    
    sweep.run_sweep(
        max_configs=args.max_configs,
        epochs_per_config=args.epochs,
        search_strategy=args.search_strategy
    )


if __name__ == '__main__':
    main()
