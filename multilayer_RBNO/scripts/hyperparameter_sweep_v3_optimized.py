#!/usr/bin/env python3
"""
Hyperparameter sweep for Variant 3 Optimized model to reduce overfitting.
Tests different combinations of width, layers, k_in/k_out, and dropout.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import time
import json
from collections import defaultdict

# Add models directory to path
cwd = os.getcwd()
models_dir = os.path.join(cwd, 'multilayer_RBNO', 'models')
if os.path.exists(models_dir):
    sys.path.insert(0, models_dir)
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from MRBNO_lift_v3_optimized import ReducedBasisBlockMultiChannelV3_Variant3_Optimized1
from MRBNO_lift_v3_variants import MRBNOLiftV3_Variant
from MRBNO_simple import PCAReducerMultiChannel

# Import data loading functions
sys.path.insert(0, os.path.join(os.getcwd(), 'multilayer_RBNO'))
from test_all_models_comprehensive_5seeds import load_kolmogorov_data, prepare_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scratch_dir = '/mnt/c/Users/dvod/MRBNO/data'


def train_model(model, train_loader, val_loader, epochs=300, lr=2e-3, weight_decay=1e-4, 
                max_grad_norm=3.0, device='cpu', early_stopping_patience=50, use_rel_l1_loss=True):
    """Train model with relative L1 loss and early stopping."""
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    
    if use_rel_l1_loss:
        def rel_l1_loss(pred, target):
            pred_flat = pred.flatten(start_dim=1)
            target_flat = target.flatten(start_dim=1)
            l1_error = torch.abs(pred_flat - target_flat).sum(dim=1)
            l1_target = torch.abs(target_flat).sum(dim=1)
            return (l1_error / (l1_target + 1e-10)).mean()
        criterion = rel_l1_loss
    else:
        criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        scheduler.step()
        
        # Print every 10 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch < 5:
            status = "✓" if val_loss < best_val_loss else "✗"
            print(f"  [{epoch+1:3d}/{epochs}] {status} train={train_loss:.6f} ({train_loss*100:.4f}%) "
                  f"val={val_loss:.6f} ({val_loss*100:.4f}%), best={best_val_loss:.6f} @ epoch {best_epoch+1}, "
                  f"patience={patience_counter}/{early_stopping_patience}, ratio={train_loss/val_loss:.4f}")
        
        if patience_counter >= early_stopping_patience:
            print(f"  ⚠ Early stopping at epoch {epoch+1}")
            break
    
    total_epochs = len(train_losses)
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch + 1,
        'total_epochs': total_epochs,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'train_val_ratio': train_losses[-1] / val_losses[-1] if val_losses[-1] > 0 else float('inf')
    }


def compute_test_error(model, test_loader, device, use_rel_l1_loss=True):
    """Compute test error."""
    model.eval()
    if use_rel_l1_loss:
        def rel_l1_loss(pred, target):
            pred_flat = pred.flatten(start_dim=1)
            target_flat = target.flatten(start_dim=1)
            l1_error = torch.abs(pred_flat - target_flat).sum(dim=1)
            l1_target = torch.abs(target_flat).sum(dim=1)
            return (l1_error / (l1_target + 1e-10)).mean()
        criterion = rel_l1_loss
    else:
        criterion = nn.MSELoss()
    
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            test_loss += loss.item() * xb.size(0)
    test_loss /= len(test_loader.dataset)
    return test_loss


def test_config(width, layers, k_modes, dropout, seed=42, num_samples=5000):
    """Test a specific hyperparameter configuration."""
    print(f"\n{'='*80}")
    print(f"Testing: width={width}, layers={layers}, k_modes={k_modes}, dropout={dropout} (seed={seed})")
    print(f"{'='*80}")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Load data
    X, Y = load_kolmogorov_data(scratch_dir, reynolds=40, num_samples=num_samples)
    
    # Prepare data
    (X_train_t, Y_train_t, X_val_t, Y_val_t, X_test_t, Y_test_t), norm_stats = prepare_data(X, Y)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t), batch_size=batch_size, shuffle=False)
    
    # Create PCA reducer
    rb = PCAReducerMultiChannel(
        bc_weight=1.0, 
        boundary_threshold=0.1, 
        use_boundary_weighting=False,
    ).fit(X_train_t.numpy(), Y_train_t.numpy())
    
    epsilon = 1e-5
    k_in_max, k_out_max = rb.k_from_cutoff(j_in=1-1*epsilon, j_out=1-epsilon)
    
    # Use specified k_modes, but cap at available modes
    k_in = min(k_modes, k_in_max)
    k_out = min(k_modes, k_out_max)
    
    print(f"  Using k_in={k_in}, k_out={k_out} (requested {k_modes}, available: {k_in_max}, {k_out_max})")
    
    # Create model
    model = MRBNOLiftV3_Variant(
        ReducedBasisBlockMultiChannelV3_Variant3_Optimized1,
        width, rb, k_in, k_out, layers, dropout, init_scale=0.1,
        use_skip_connections=True, use_batch_norm=False, use_coordinates=False
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")
    
    # Train
    train_results = train_model(
        model, train_loader, val_loader, epochs=300, lr=2e-3, 
        weight_decay=1e-4, max_grad_norm=3.0, device=device,
        early_stopping_patience=50, use_rel_l1_loss=True
    )
    
    # Test
    test_loss = compute_test_error(model, test_loader, device, use_rel_l1_loss=True)
    
    results = {
        'width': width,
        'layers': layers,
        'k_modes': k_modes,
        'k_in': k_in,
        'k_out': k_out,
        'dropout': dropout,
        'seed': seed,
        'param_count': param_count,
        'best_val_loss': train_results['best_val_loss'],
        'best_epoch': train_results['best_epoch'],
        'total_epochs': train_results['total_epochs'],
        'final_train_loss': train_results['final_train_loss'],
        'final_val_loss': train_results['final_val_loss'],
        'test_loss': test_loss,
        'train_val_ratio': train_results['train_val_ratio']
    }
    
    print(f"  Results: Best Val={train_results['best_val_loss']*100:.2f}%, "
          f"Test={test_loss*100:.2f}%, Train/Val ratio={train_results['train_val_ratio']:.4f}")
    
    return results


def main():
    print("="*80)
    print("HYPERPARAMETER SWEEP - Variant 3 Optimized")
    print("Testing: width, layers, k_modes, dropout")
    print("="*80)
    
    # Hyperparameter grid
    widths = [20, 24, 28]
    layers_list = [3, 4]
    k_modes_list = [20, 30, 40, 50]  # Same for k_in and k_out
    dropouts = [0.0, 0.1, 0.2, 0.3]
    
    # Start with smaller config as requested
    configs = []
    
    # Start with: layers=3, width=20, then expand
    for layers in layers_list:
        for width in widths:
            for k_modes in k_modes_list:
                for dropout in dropouts:
                    # Prioritize: start with layers=3, width=20
                    if layers == 3 and width == 20:
                        priority = 1
                    elif layers == 3:
                        priority = 2
                    elif width == 20:
                        priority = 3
                    else:
                        priority = 4
                    configs.append((priority, layers, width, k_modes, dropout))
    
    # Sort by priority
    configs.sort(key=lambda x: x[0])
    configs = [(l, w, k, d) for _, l, w, k, d in configs]
    
    print(f"\nTotal configurations to test: {len(configs)}")
    print("Starting with: layers=3, width=20, then expanding...\n")
    
    all_results = []
    seed = 42
    
    for i, (layers, width, k_modes, dropout) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing: layers={layers}, width={width}, k_modes={k_modes}, dropout={dropout}")
        try:
            results = test_config(width, layers, k_modes, dropout, seed=seed, num_samples=5000)
            all_results.append(results)
            
            # Save intermediate results
            with open('hyperparameter_sweep_v3_optimized.json', 'w') as f:
                json.dump(all_results, f, indent=2)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - Best Configurations")
    print("="*80)
    
    # Sort by best validation loss
    all_results_sorted = sorted(all_results, key=lambda x: x['best_val_loss'])
    
    print(f"\n{'Rank':<6} {'Width':<8} {'Layers':<8} {'K_modes':<10} {'Dropout':<10} {'Params':<12} "
          f"{'Best Val':<12} {'Test':<12} {'Train/Val':<12} {'Best Epoch':<12}")
    print("-" * 120)
    
    for i, r in enumerate(all_results_sorted[:20]):  # Top 20
        print(f"{i+1:<6} {r['width']:<8} {r['layers']:<8} {r['k_modes']:<10} {r['dropout']:<10.1f} "
              f"{r['param_count']:<12,} {r['best_val_loss']*100:<12.2f}% {r['test_loss']*100:<12.2f}% "
              f"{r['train_val_ratio']:<12.4f} {r['best_epoch']:<12}")
    
    # Save final results
    with open('hyperparameter_sweep_v3_optimized.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to hyperparameter_sweep_v3_optimized.json")
    print(f"✓ Tested {len(all_results)} configurations")
    
    # Find best configuration
    if all_results:
        best = min(all_results, key=lambda x: x['best_val_loss'])
        print(f"\n{'='*80}")
        print("BEST CONFIGURATION:")
        print(f"{'='*80}")
        print(f"  Width: {best['width']}")
        print(f"  Layers: {best['layers']}")
        print(f"  K_modes: {best['k_modes']}")
        print(f"  Dropout: {best['dropout']}")
        print(f"  Parameters: {best['param_count']:,}")
        print(f"  Best Val Loss: {best['best_val_loss']*100:.2f}%")
        print(f"  Test Loss: {best['test_loss']*100:.2f}%")
        print(f"  Train/Val Ratio: {best['train_val_ratio']:.4f}")
        print(f"  Best Epoch: {best['best_epoch']}")


if __name__ == "__main__":
    main()

