"""
Collect MRBNO test metrics and print comparison table vs Geo-FNO paper models.

Uses same input-output mapping and splits as Geo-FNO (see geofno_loader and
Geo-FNO repo: elasticity ntrain=1000/ntest=200, airfoil ntrain=1000/ntest=200, etc.).
Paper reports relative L2 error (Eq. 2). Run after training all datasets.

Usage:
  python compare_with_geofno_paper.py [--device cpu]
"""

import os
import sys
import argparse
import math

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
sys.path.insert(1, os.path.join(_script_dir, '..', 'models'))

import torch
from torch.utils.data import DataLoader
from geofno_loader import prepare_geofno_data
from losses import LpLoss
from MRBNO import MRBNO
from PCAReducer import PCAReducerMultiChannel

DATA_BASE_DIR = '/mnt/c/Users/dvod/MRBNO/data'
CHECKPOINT_DIR = os.path.join(DATA_BASE_DIR, 'checkpoints', 'geofno')

DATASET_ORDER = ['elasticity', 'airfoil', 'pipe', 'plasticity', 'car-cfd', 'channel-shocks']


def compute_test_rel_metrics(model, test_loader, device):
    """Return dict with rel_l1, rel_l2 (ratios), and rel_l2_pct (rel L2 in percent).
    Averages over test samples (same as paper Eq. 2). LpLoss with size_average=False returns sum over batch."""
    model.eval()
    rel_l1_crit = LpLoss(d=2, p=1, size_average=False, reduction=True)
    rel_l1_crit.forward = lambda x, y: rel_l1_crit.rel_l1(x, y)
    rel_l2_crit = LpLoss(d=2, p=2, size_average=False, reduction=True)
    rel_l2_crit.forward = lambda x, y: rel_l2_crit.rel(x, y)
    rel_l1_sum, rel_l2_sum, n_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            batch_data, batch_target, _ = batch
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            out = model(batch_data)
            rel_l1_sum += rel_l1_crit(out, batch_target).item()
            rel_l2_sum += rel_l2_crit(out, batch_target).item()
            n_samples += batch_target.size(0)
    rel_l2 = rel_l2_sum / n_samples if n_samples else float('nan')
    return {
        'rel_l1': rel_l1_sum / n_samples if n_samples else float('nan'),
        'rel_l2': rel_l2,
        'rel_l2_pct': rel_l2 * 100.0 if not math.isnan(rel_l2) else float('nan'),
    }


def load_model_and_test(checkpoint_path, dataset_type, device):
    """Load checkpoint and return test rel_l1, rel_l2, and parameter count."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    datasets, _, _ = prepare_geofno_data(
        args['dataset_path'],
        dataset_type=dataset_type,
        normalize=False,
        use_geofno_split=True,
    )
    train_dataset = datasets['train']
    test_dataset = datasets['test']
    train_data_3d = train_dataset.data.numpy()
    train_targets_3d = train_dataset.targets.numpy()
    rb = PCAReducerMultiChannel(
        svd_solver="auto", whiten=False, random_state=42,
        bc_weight=1.0, boundary_threshold=0.1, use_boundary_weighting=False,
    ).fit(train_data_3d, train_targets_3d)
    k_in = checkpoint.get('k_in', args.get('n_modes', 144))
    k_out = checkpoint.get('k_out', args.get('n_modes', 144))
    model = MRBNO(
        width=args['width'],
        rb=rb,
        k_in=k_in,
        k_out=k_out,
        layers=args['n_layers'],
        dropout=0.0,
        init_scale=0.1,
        use_skip_connections=True,
        use_batch_norm=True,
        use_coordinates=args.get('use_coordinates', args.get('add_coords', False)),
        add_constant_mode=False,
        use_factorized_weights=False,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    n_params = sum(p.numel() for p in model.parameters())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    metrics = compute_test_rel_metrics(model, test_loader, device)
    metrics['n_params'] = n_params
    return metrics


def main():
    parser = argparse.ArgumentParser(description='MRBNO vs Geo-FNO paper comparison table')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR)
    parser.add_argument('--datasets', type=str, nargs='*', default=None,
                        help='Only these datasets (default: all)')
    args = parser.parse_args()
    device = torch.device(args.device)
    order = [n for n in DATASET_ORDER if args.datasets is None or n in args.datasets]

    results = {}
    for name in order:
        ckpt = os.path.join(args.checkpoint_dir, f'checkpoint_best_{name}.pt')
        if not os.path.isfile(ckpt):
            results[name] = None
            continue
        try:
            results[name] = load_model_and_test(ckpt, name, device)
        except Exception as e:
            print(f"Error loading {name}: {e}", file=sys.stderr)
            results[name] = None

    # Paper: Li et al. JMLR 24(388) 2023. Table 1/2 report relative L2 (ratio); we show percent.
    # Elasticity: from paper Table 1 (hyper-elastic). Other datasets: fill from paper Table 1/2 when available.
    PAPER_TABLE = {
        'elasticity': {'fno_rel_l2_pct': 4.95, 'unet_rel_l2_pct': 5.34, 'gno_rel_l2_pct': 12.7, 'deeponet_rel_l2_pct': 9.65, 'geofno_rel_l2_pct': 3.41, 'fno_params': '—', 'unet_params': '—', 'gno_params': '—', 'deeponet_params': '—', 'geofno_params': '—'},
        'airfoil': {'fno_rel_l2_pct': '—', 'unet_rel_l2_pct': '—', 'gno_rel_l2_pct': '—', 'deeponet_rel_l2_pct': '—', 'geofno_rel_l2_pct': '—', 'fno_params': '—', 'unet_params': '—', 'gno_params': '—', 'deeponet_params': '—', 'geofno_params': '—'},
        'pipe': {'fno_rel_l2_pct': '—', 'unet_rel_l2_pct': '—', 'gno_rel_l2_pct': '—', 'deeponet_rel_l2_pct': '—', 'geofno_rel_l2_pct': '—', 'fno_params': '—', 'unet_params': '—', 'gno_params': '—', 'deeponet_params': '—', 'geofno_params': '—'},
        'plasticity': {'fno_rel_l2_pct': '—', 'unet_rel_l2_pct': '—', 'gno_rel_l2_pct': '—', 'deeponet_rel_l2_pct': '—', 'geofno_rel_l2_pct': '—', 'fno_params': '—', 'unet_params': '—', 'gno_params': '—', 'deeponet_params': '—', 'geofno_params': '—'},
        'car-cfd': {'fno_rel_l2_pct': '—', 'unet_rel_l2_pct': '—', 'gno_rel_l2_pct': '—', 'deeponet_rel_l2_pct': '—', 'geofno_rel_l2_pct': '—', 'fno_params': '—', 'unet_params': '—', 'gno_params': '—', 'deeponet_params': '—', 'geofno_params': '—'},
        'channel-shocks': {'fno_rel_l2_pct': '—', 'unet_rel_l2_pct': '—', 'gno_rel_l2_pct': '—', 'deeponet_rel_l2_pct': '—', 'geofno_rel_l2_pct': '—', 'fno_params': '—', 'unet_params': '—', 'gno_params': '—', 'deeponet_params': '—', 'geofno_params': '—'},
    }
    print()
    print("=" * 120)
    print("FULL COMPARISON: Test relative L2 error (%) and parameters (all models from Geo-FNO paper + MRBNO)")
    print("All error values in PERCENT (100 * relative L2). Same mapping and splits as paper.")
    print("=" * 120)
    print(f"{'Dataset':<18} {'Model':<24} {'Test Rel L2 (%)':<18} {'Parameters':<14}")
    print("-" * 120)
    PAPER_MODELS = [
        ('FNO (interp, paper)', 'fno_rel_l2_pct', 'fno_params'),
        ('UNet (paper)', 'unet_rel_l2_pct', 'unet_params'),
        ('GNO (paper)', 'gno_rel_l2_pct', 'gno_params'),
        ('DeepONet (paper)', 'deeponet_rel_l2_pct', 'deeponet_params'),
        ('Geo-FNO (paper)', 'geofno_rel_l2_pct', 'geofno_params'),
    ]
    for name in order:
        p = PAPER_TABLE.get(name, {})
        mrbno = results.get(name)
        mrbno_pct = f"{mrbno['rel_l2_pct']:.2f}" if mrbno and not math.isnan(mrbno.get('rel_l2_pct', float('nan'))) else "—"
        mrbno_par = f"{mrbno['n_params']:,}" if mrbno and mrbno.get('n_params') else "—"
        for label, key_pct, key_par in PAPER_MODELS:
            val_pct = p.get(key_pct, '—')
            val_par = p.get(key_par, '—')
            print(f"{name:<18} {label:<24} {str(val_pct):<18} {str(val_par):<14}")
        print(f"{'':18} {'MRBNO (this work)':<24} {mrbno_pct:<18} {mrbno_par:<14}")
        print("-" * 120)
    print("=" * 120)
    print()
    print("Paper: Li et al. JMLR 24(388) 2023, Section 4, Table 1 and Table 2. Elasticity from Table 1 (hyper-elastic).")

    # Write table to file for COMPARISON_RESULTS.md
    out_path = os.path.join(_script_dir, 'comparison_results.txt')
    with open(out_path, 'w') as f:
        f.write("Test Rel L2 (%) and Parameters (re-run compare_with_geofno_paper.py to refresh)\n")
        f.write("All values in percent.\n\n")
        for name in order:
            p = PAPER_TABLE.get(name, {})
            mrbno = results.get(name)
            mrbno_pct = f"{mrbno['rel_l2_pct']:.2f}" if mrbno and not math.isnan(mrbno.get('rel_l2_pct', float('nan'))) else "—"
            mrbno_par = f"{mrbno['n_params']:,}" if mrbno and mrbno.get('n_params') else "—"
            for label, key_pct, key_par in PAPER_MODELS:
                val_pct = p.get(key_pct, '—')
                val_par = p.get(key_par, '—')
                f.write(f"{name}\t{label}\t{val_pct}\t{val_par}\n")
            f.write(f"{name}\tMRBNO (this work)\t{mrbno_pct}\t{mrbno_par}\n")
    print(f"Table written to {out_path}")

    # Flag datasets where MRBNO rel L2 >= 50% or significantly worse than best paper model
    HIGH_ERROR_PCT = 50.0
    print()
    print("Datasets with MRBNO test Rel L2 >= 50% or worth investigating:")
    for name in order:
        mrbno = results.get(name)
        if not mrbno or math.isnan(mrbno.get('rel_l2_pct', float('nan'))):
            continue
        pct = mrbno['rel_l2_pct']
        best_paper = None
        p = PAPER_TABLE.get(name, {})
        for key in ('geofno_rel_l2_pct', 'fno_rel_l2_pct', 'unet_rel_l2_pct', 'gno_rel_l2_pct', 'deeponet_rel_l2_pct'):
            v = p.get(key)
            if isinstance(v, (int, float)):
                best_paper = v if best_paper is None else min(best_paper, v)
        if pct >= HIGH_ERROR_PCT:
            print(f"  {name}: MRBNO {pct:.1f}% (>= {HIGH_ERROR_PCT}%)")
        elif best_paper is not None and pct >= 2.0 * best_paper:
            print(f"  {name}: MRBNO {pct:.1f}% vs best paper {best_paper:.1f}% (≥2× worse)")


if __name__ == "__main__":
    main()
