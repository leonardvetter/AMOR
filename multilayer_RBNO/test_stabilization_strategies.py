#!/usr/bin/env python3
"""
Adapted learning-rate strategy sweep from main branch to evaluate both MRBNO
Simple v1 and v2 across all baseline schedulers. For every strategy we train:
  • MRBNO Simple v1 with latent batch norm disabled
  • (commented) MRBNO Simple v1 with latent batch norm enabled
  • MRBNO Simple v2 with latent batch norm disabled
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import time
import random
import json

import sys
sys.path.append('models')
from MRBNO_simple import MRBNOSimple, PCAReducerMultiChannel
from MRBNO_simple_v2 import MRBNOSimpleV2

DATA_PATH = Path('data/nonlinear_diffusion_dataset.npz')
OUTPUT_DIR = Path('stabilization_results')
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_FRAC = 0.7
NUM_SAMPLES = 1000
BATCH_SIZE = 32
EPOCHS = 300
NUM_RUNS = 10
SEEDS = list(range(42, 42 + NUM_RUNS))

BASE_LR = 1e-3
BASE_WEIGHT_DECAY = 1e-4

MRBNO_CONFIG = {
    'layers': 7,
    'hidden': 1024,
    'use_skip_connections': True,
    'dropout': 0.2,
    'init_scale': 0.1,
}

MODEL_VARIANTS = [
    ('mrbno_simple_v1__bn_false', MRBNOSimple, False),
    # ('mrbno_simple_v1__bn_true', MRBNOSimple, True),  # Deferred run
    ('mrbno_simple_v2__bn_false', MRBNOSimpleV2, False),
]

LR_STRATEGIES = {
    'plateau_default': {
        'description': 'ReduceLROnPlateau (factor=0.5, patience=20)',
        'scheduler': 'plateau',
        'scheduler_config': {'factor': 0.5, 'patience': 20, 'threshold': 1e-4},
        'lr': BASE_LR,
        'weight_decay': BASE_WEIGHT_DECAY,
    },
#    'plateau_fast': {
#        'description': 'ReduceLROnPlateau (factor=0.5, patience=10)',
#        'scheduler': 'plateau',
#        'scheduler_config': {'factor': 0.5, 'patience': 10, 'threshold': 5e-4},
#        'lr': BASE_LR,
#        'weight_decay': BASE_WEIGHT_DECAY,
#    },
    'plateau_aggressive': {
        'description': 'ReduceLROnPlateau (factor=0.3, patience=15)',
        'scheduler': 'plateau',
        'scheduler_config': {'factor': 0.3, 'patience': 15, 'threshold': 1e-5},
        'lr': BASE_LR,
        'weight_decay': BASE_WEIGHT_DECAY,
    },
    'cosine_base': {
        'description': 'CosineAnnealingLR baseline (T_max=epochs, eta_min=0)',
        'scheduler': 'cosine',
        'scheduler_config': {'eta_min': 0.0},
        'lr': BASE_LR,
        'weight_decay': BASE_WEIGHT_DECAY,
    },
    'cosine_short_cycle': {
        'description': 'CosineAnnealingLR with T_max=150',
        'scheduler': 'cosine',
        'scheduler_config': {'T_max': 150, 'eta_min': 0.0},
        'lr': BASE_LR,
        'weight_decay': BASE_WEIGHT_DECAY,
    },
    'cosine_low_min': {
        'description': 'CosineAnnealingLR with eta_min=1e-5',
        'scheduler': 'cosine',
        'scheduler_config': {'eta_min': 1e-5},
        'lr': BASE_LR,
        'weight_decay': BASE_WEIGHT_DECAY,
    },
#    'cosine_lr_low': {
#        'description': 'CosineAnnealingLR with lower LR (5e-4)',
#        'scheduler': 'cosine',
#        'scheduler_config': {'eta_min': 0.0},
#        'lr': 5e-4,
#        'weight_decay': BASE_WEIGHT_DECAY,
#    },
    'cosine_lr_high': {
        'description': 'CosineAnnealingLR with higher LR (2e-3)',
        'scheduler': 'cosine',
        'scheduler_config': {'eta_min': 0.0},
        'lr': 2e-3,
        'weight_decay': BASE_WEIGHT_DECAY,
    },
    'cosine_restart': {
        'description': 'CosineAnnealingWarmRestarts (T_0=60)',
        'scheduler': 'cosine_restart',
        'scheduler_config': {'T_0': 60, 'T_mult': 1, 'eta_min': 1e-5},
        'lr': BASE_LR,
        'weight_decay': BASE_WEIGHT_DECAY,
    },
    'warmup_cosine_warm5': {
        'description': 'Warmup 5 epochs then cosine',
        'scheduler': 'warmup_cosine',
        'scheduler_config': {'warmup_epochs': 5, 'min_lr_scale': 0.1},
        'lr': BASE_LR,
        'weight_decay': BASE_WEIGHT_DECAY,
    },
#    'warmup_cosine_warm20': {
#        'description': 'Warmup 20 epochs then cosine',
#        'scheduler': 'warmup_cosine',
#        'scheduler_config': {'warmup_epochs': 20, 'min_lr_scale': 0.1},
#        'lr': BASE_LR,
#        'weight_decay': BASE_WEIGHT_DECAY,
#    },
    'step_decay': {
        'description': 'StepLR step=75 gamma=0.5',
        'scheduler': 'step',
        'scheduler_config': {'step_size': 75, 'gamma': 0.5},
        'lr': BASE_LR,
        'weight_decay': BASE_WEIGHT_DECAY,
    },
#    'step_decay_fast': {
#        'description': 'StepLR step=50 gamma=0.5',
#        'scheduler': 'step',
#        'scheduler_config': {'step_size': 50, 'gamma': 0.5},
#        'lr': BASE_LR,
#        'weight_decay': BASE_WEIGHT_DECAY,
#    },
    'multistep_decay': {
        'description': 'MultiStepLR milestones=[75,150,225]',
        'scheduler': 'multistep',
        'scheduler_config': {'milestones': [75, 150, 225], 'gamma': 0.5},
        'lr': BASE_LR,
        'weight_decay': BASE_WEIGHT_DECAY,
    },
    'linear_decay': {
        'description': 'Linear LR decay to zero',
        'scheduler': 'linear_decay',
        'lr': BASE_LR,
        'weight_decay': BASE_WEIGHT_DECAY,
    },
    'constant_lr': {
        'description': 'Constant learning rate (no scheduler)',
        'scheduler': 'none',
        'lr': BASE_LR,
        'weight_decay': BASE_WEIGHT_DECAY,
    },
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def remove_latent_activations(model):
    for module in model.modules():
        if hasattr(module, 'latent_net') and isinstance(module.latent_net, nn.Sequential):
            module.latent_net = nn.Sequential(
                *[layer for layer in module.latent_net if not isinstance(layer, nn.GELU)]
            )


def train_one_model(model, train_loader, val_loader, *, epochs, lr, weight_decay,
                    scheduler_type='cosine', scheduler_kwargs=None, device='cpu'):
    scheduler_kwargs = scheduler_kwargs or {}

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_kwargs.get('T_max', epochs), eta_min=scheduler_kwargs.get('eta_min', 0.0))
    elif scheduler_type == 'cosine_restart':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_kwargs.get('T_0', max(1, epochs // 6)),
            T_mult=scheduler_kwargs.get('T_mult', 1),
            eta_min=scheduler_kwargs.get('eta_min', 0.0)
        )
    elif scheduler_type == 'warmup_cosine':
        warmup_epochs = scheduler_kwargs.get('warmup_epochs', 10)
        min_lr_scale = scheduler_kwargs.get('min_lr_scale', 0.0)

        def lr_lambda(epoch):
            if warmup_epochs <= 0:
                progress = epoch / max(1, epochs)
                return min_lr_scale + (1 - min_lr_scale) * 0.5 * (1 + np.cos(np.pi * progress))
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            progress = np.clip(progress, 0.0, 1.0)
            return min_lr_scale + (1 - min_lr_scale) * 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_kwargs.get('factor', 0.5),
            patience=scheduler_kwargs.get('patience', 20),
            threshold=scheduler_kwargs.get('threshold', 1e-4)
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_kwargs.get('step_size', max(1, epochs // 3)),
            gamma=scheduler_kwargs.get('gamma', 0.5)
        )
    elif scheduler_type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_kwargs.get('milestones', [epochs // 2, int(epochs * 0.75)]),
            gamma=scheduler_kwargs.get('gamma', 0.5)
        )
    elif scheduler_type == 'linear_decay':
        def lr_lambda(epoch):
            return max(0.0, 1 - epoch / max(1, epochs))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    train_losses = []
    val_losses = []
    best_val = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

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
        best_val = min(best_val, val_loss)

        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if (epoch + 1) % 50 == 0 or epoch == 1:
            print(f"    Epoch {epoch+1:03d}/{epochs} train={train_loss:.4e} val={val_loss:.4e} best={best_val:.4e}")

    return train_losses, val_losses, best_val


def main():
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data = np.load(DATA_PATH)
    f_all = data['f']
    m_all = data['m']
    u_all = data['u']

    X = np.stack([f_all, m_all], axis=1)
    Y = u_all

    n_train = int(TRAIN_FRAC * len(X))
    X_train_full, Y_train_full = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]

    X_train = X_train_full[:NUM_SAMPLES]
    Y_train = Y_train_full[:NUM_SAMPLES]

    print(f"Total samples: {len(X)}")
    print(f"Training subset used: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    x_mean, x_std = X_train.mean(), X_train.std()
    y_mean, y_std = Y_train.mean(), Y_train.std()

    X_train = (X_train - x_mean) / (x_std + 1e-8)
    Y_train = (Y_train - y_mean) / (y_std + 1e-8)
    X_val = (X_val - x_mean) / (x_std + 1e-8)
    Y_val = (Y_val - y_mean) / (y_std + 1e-8)

    X_train_t = torch.from_numpy(X_train).float()
    Y_train_t = torch.from_numpy(Y_train).float().unsqueeze(1)
    X_val_t = torch.from_numpy(X_val).float()
    Y_val_t = torch.from_numpy(Y_val).float().unsqueeze(1)

    train_dataset = TensorDataset(X_train_t, Y_train_t)
    val_dataset = TensorDataset(X_val_t, Y_val_t)

    print("\n" + "=" * 70)
    print("COMPUTING PCA")
    print("=" * 70)
    rb = PCAReducerMultiChannel(bc_weight=2.0, boundary_threshold=0.05,
                                use_boundary_weighting=False, whiten=True)
    rb.fit(X_train, Y_train)

    k_in, k_out = 122, 61
    print(f"Using k_in={k_in}, k_out={k_out}")

    results = {}

    for strategy_name, cfg in LR_STRATEGIES.items():
        print("\n" + "=" * 70)
        print(f"STRATEGY: {strategy_name}")
        print(cfg['description'])
        print("=" * 70)

        strategy_results = {}

        for model_label, model_cls, use_bn in MODEL_VARIANTS:
            print(f"\n--- MODEL: {model_label} ---")

            model_runs = []

            for run_idx, seed in enumerate(SEEDS, 1):
                print(f"  Run {run_idx}/{NUM_RUNS} (seed={seed})")
                set_seed(seed)

                model = model_cls(
                    rb=rb,
                    k_in=k_in,
                    k_out=k_out,
                    latent_batch_norm=use_bn,
                    **MRBNO_CONFIG,
                )

                remove_latent_activations(model)

                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

                start_time = time.time()
                train_losses, val_losses, best_val = train_one_model(
                    model,
                    train_loader,
                    val_loader,
                    epochs=EPOCHS,
                    lr=cfg.get('lr', BASE_LR),
                    weight_decay=cfg.get('weight_decay', BASE_WEIGHT_DECAY),
                    scheduler_type=cfg.get('scheduler', 'cosine'),
                    scheduler_kwargs=cfg.get('scheduler_config'),
                    device=device,
                )
                elapsed = time.time() - start_time

                print(f"    Best val loss: {best_val:.4e} (time: {elapsed:.1f}s)")

                model_runs.append({
                    'seed': seed,
                    'best_val_loss': best_val,
                    'final_val_loss': val_losses[-1],
                    'final_train_loss': train_losses[-1],
                    'training_time': elapsed,
                })

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            best_vals = np.array([r['best_val_loss'] for r in model_runs])
            stats = {
                'mean': float(best_vals.mean()),
                'std': float(best_vals.std()),
                'min': float(best_vals.min()),
                'max': float(best_vals.max()),
                'cv': float(best_vals.std() / best_vals.mean()) if best_vals.mean() > 0 else float('nan'),
            }

            print("\n  Statistics across runs:")
            print(f"    Mean: {stats['mean']:.4e}")
            print(f"    Std:  {stats['std']:.4e}")
            print(f"    CV:   {stats['cv']:.4f}")
            print(f"    Min:  {stats['min']:.4e}")
            print(f"    Max:  {stats['max']:.4e}")

            strategy_results[model_label] = {
                'description': f"{cfg['description']} (batch_norm={use_bn})",
                'stats': stats,
                'runs': model_runs,
            }

        results[strategy_name] = strategy_results

    output_path = OUTPUT_DIR / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to", output_path)


if __name__ == '__main__':
    main()
