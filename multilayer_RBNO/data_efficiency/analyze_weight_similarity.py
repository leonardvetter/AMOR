#!/usr/bin/env python3
"""
Analyze weight similarity between baseline and two-stage training models.

Measures:
1. PCA space weights (block weights) similarity:
   - Stage 1 vs Baseline
   - Stage 2 (v1-v7) vs Baseline
2. Real space weights (lift, proj, global_skip) similarity:
   - Stage 2 (v1-v7) vs Baseline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import torch
import numpy as np
import argparse
from collections import OrderedDict

from MRBNO import MRBNO
from train_two_stage import MRBNOStage1, PCAReducerMultiChannel


def cosine_similarity(w1, w2):
    """Compute cosine similarity between two weight tensors."""
    w1_flat = w1.flatten()
    w2_flat = w2.flatten()
    dot_product = torch.dot(w1_flat, w2_flat)
    norm1 = torch.norm(w1_flat)
    norm2 = torch.norm(w2_flat)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return (dot_product / (norm1 * norm2)).item()


def l2_distance(w1, w2):
    """Compute L2 distance between two weight tensors."""
    w1_flat = w1.flatten()
    w2_flat = w2.flatten()
    return torch.norm(w1_flat - w2_flat).item()


def relative_l2_distance(w1, w2):
    """Compute relative L2 distance: ||w1 - w2||_2 / ||w1||_2"""
    w1_flat = w1.flatten()
    w2_flat = w2.flatten()
    diff = torch.norm(w1_flat - w2_flat)
    norm1 = torch.norm(w1_flat)
    if norm1 == 0:
        return float('inf')
    return (diff / norm1).item()


def extract_pca_weights(model, prefix=""):
    """Extract PCA space weights from model (block weights)."""
    weights = OrderedDict()
    
    if isinstance(model, MRBNOStage1):
        blocks = [('block0', model.block0), ('block1', model.block1)] + \
                 [(f'blocks.{i}', block) for i, block in enumerate(model.blocks)]
    elif isinstance(model, MRBNO):
        blocks = [('block0', model.block0), ('block1', model.block1)] + \
                 [(f'blocks.{i}', block) for i, block in enumerate(model.blocks)]
    else:
        return weights
    
    for name, block in blocks:
        full_name = f"{prefix}{name}" if prefix else name
        
        if hasattr(block, 'weight') and block.weight is not None:
            weights[f"{full_name}.weight"] = block.weight.data.clone()
        elif hasattr(block, 'width_weights') and hasattr(block, 'mode_weight'):
            weights[f"{full_name}.width_weights"] = block.width_weights.data.clone()
            weights[f"{full_name}.mode_weight"] = block.mode_weight.data.clone()
    
    return weights


def extract_real_space_weights(model, prefix=""):
    """Extract real space weights from model (lift, proj, global_skip)."""
    weights = OrderedDict()
    
    if isinstance(model, MRBNOStage1):
        if hasattr(model, 'lift') and model.lift is not None:
            weights[f"{prefix}lift.weight"] = model.lift.weight.data.clone()
        if hasattr(model, 'proj1') and model.proj1 is not None:
            weights[f"{prefix}proj1.weight"] = model.proj1.weight.data.clone()
        if hasattr(model, 'proj2') and model.proj2 is not None:
            weights[f"{prefix}proj2.weight"] = model.proj2.weight.data.clone()
    elif isinstance(model, MRBNO):
        if hasattr(model, 'lift') and model.lift is not None:
            weights[f"{prefix}lift.weight"] = model.lift.weight.data.clone()
        if hasattr(model, 'proj1') and model.proj1 is not None:
            weights[f"{prefix}proj1.weight"] = model.proj1.weight.data.clone()
        if hasattr(model, 'proj2') and model.proj2 is not None:
            weights[f"{prefix}proj2.weight"] = model.proj2.weight.data.clone()
        if hasattr(model, 'global_skip_proj') and model.global_skip_proj is not None:
            weights[f"{prefix}global_skip_proj.weight"] = model.global_skip_proj.weight.data.clone()
        
        blocks = [('block0', model.block0), ('block1', model.block1)] + \
                 [(f'blocks.{i}', block) for i, block in enumerate(model.blocks)]
        for name, block in blocks:
            full_name = f"{prefix}{name}" if prefix else name
            if hasattr(block, 'skip_conv') and block.skip_conv is not None:
                weights[f"{full_name}.skip_conv.weight"] = block.skip_conv.weight.data.clone()
    
    return weights


def compare_weights(weights1, weights2, name1="Model 1", name2="Model 2"):
    """Compare two sets of weights and return metrics."""
    results = {
        'cosine_similarity': {},
        'l2_distance': {},
        'relative_l2_distance': {},
        'common_keys': [],
        'missing_in_1': [],
        'missing_in_2': []
    }
    
    all_keys = set(weights1.keys()) | set(weights2.keys())
    common_keys = set(weights1.keys()) & set(weights2.keys())
    
    results['common_keys'] = sorted(common_keys)
    results['missing_in_1'] = sorted(set(weights2.keys()) - set(weights1.keys()))
    results['missing_in_2'] = sorted(set(weights1.keys()) - set(weights2.keys()))
    
    for key in common_keys:
        w1 = weights1[key]
        w2 = weights2[key]
        
        if w1.shape != w2.shape:
            print(f"WARNING: Shape mismatch for {key}: {w1.shape} vs {w2.shape}")
            continue
        
        results['cosine_similarity'][key] = cosine_similarity(w1, w2)
        results['l2_distance'][key] = l2_distance(w1, w2)
        results['relative_l2_distance'][key] = relative_l2_distance(w1, w2)
    
    return results


def print_comparison(results, name1="Model 1", name2="Model 2"):
    """Print comparison results in a readable format."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {name1} vs {name2}")
    print(f"{'='*80}")
    
    if not results['common_keys']:
        print("No common weights found!")
        return
    
    print(f"\nCommon weights: {len(results['common_keys'])}")
    if results['missing_in_1']:
        print(f"Missing in {name1}: {results['missing_in_1']}")
    if results['missing_in_2']:
        print(f"Missing in {name2}: {results['missing_in_2']}")
    
    print(f"\n{'Weight':<40} {'Cosine Sim':<15} {'Rel L2 Dist':<15} {'L2 Dist':<15}")
    print("-" * 80)
    
    for key in results['common_keys']:
        cos_sim = results['cosine_similarity'].get(key, float('nan'))
        rel_l2 = results['relative_l2_distance'].get(key, float('nan'))
        l2_dist = results['l2_distance'].get(key, float('nan'))
        print(f"{key:<40} {cos_sim:>14.6f} {rel_l2:>14.6f} {l2_dist:>14.6f}")
    
    avg_cos_sim = np.mean([v for v in results['cosine_similarity'].values()])
    avg_rel_l2 = np.mean([v for v in results['relative_l2_distance'].values()])
    avg_l2 = np.mean([v for v in results['l2_distance'].values()])
    
    print("-" * 80)
    print(f"{'AVERAGE':<40} {avg_cos_sim:>14.6f} {avg_rel_l2:>14.6f} {avg_l2:>14.6f}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze weight similarity')
    parser.add_argument('--baseline_checkpoint', type=str, required=True,
                       help='Path to baseline model checkpoint')
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                       help='Path to Stage 1 checkpoint')
    parser.add_argument('--stage2_checkpoints', type=str, nargs='+', default=[],
                       help='Paths to Stage 2 checkpoints (v1-v7). If not provided, will look for stage2_v*_best.pt in checkpoint directory.')
    parser.add_argument('--stage2_names', type=str, nargs='+', default=[],
                       help='Names for Stage 2 versions (e.g., v1, v2, v3)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Directory containing checkpoints (default: $SCRATCH/data_efficiency/checkpoints)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    if not args.stage2_checkpoints:
        if args.checkpoint_dir is None:
            checkpoint_dir = '/mnt/c/Users/dvod/MRBNO/data/data_efficiency/checkpoints'
        else:
            checkpoint_dir = args.checkpoint_dir
        import glob
        stage2_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'stage2_v*_best.pt')))
        if stage2_checkpoints:
            args.stage2_checkpoints = stage2_checkpoints
            if not args.stage2_names:
                args.stage2_names = [f"v{i+1}" for i in range(len(stage2_checkpoints))]
            print(f"Found {len(stage2_checkpoints)} Stage 2 checkpoints: {[os.path.basename(c) for c in stage2_checkpoints]}")
        else:
            print("WARNING: No Stage 2 checkpoints found. Please provide --stage2_checkpoints.")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("WEIGHT SIMILARITY ANALYSIS")
    print("="*80)
    
    print("\nLoading Stage 1 model (to get shared PCA basis)...")
    stage1_ckpt = torch.load(args.stage1_checkpoint, map_location='cpu', weights_only=False)
    stage1_rb_data = stage1_ckpt.get('rb_data', stage1_ckpt.get('rb'))
    if isinstance(stage1_rb_data, dict):
        stage1_rb = PCAReducerMultiChannel()
        stage1_rb.pca_in = stage1_rb_data['pca_in']
        stage1_rb.pca_out = stage1_rb_data['pca_out']
        stage1_rb.n_channels = stage1_rb_data['n_channels']
        stage1_rb.out_channels = stage1_rb_data['out_channels']
        stage1_rb.shape_ = stage1_rb_data['shape_']
    else:
        stage1_rb = stage1_rb_data
    
    print("Loading baseline model...")
    baseline_ckpt = torch.load(args.baseline_checkpoint, map_location='cpu', weights_only=False)
    baseline_rb_data = baseline_ckpt.get('rb_data', baseline_ckpt.get('rb'))
    if baseline_rb_data is not None:
        if isinstance(baseline_rb_data, dict):
            baseline_rb = PCAReducerMultiChannel()
            baseline_rb.pca_in = baseline_rb_data['pca_in']
            baseline_rb.pca_out = baseline_rb_data['pca_out']
            baseline_rb.n_channels = baseline_rb_data['n_channels']
            baseline_rb.out_channels = baseline_rb_data['out_channels']
            baseline_rb.shape_ = baseline_rb_data['shape_']
        else:
            baseline_rb = baseline_rb_data
    else:
        print("WARNING: Baseline checkpoint doesn't contain rb. Using Stage 1's rb (should be identical).")
        baseline_rb = stage1_rb
    baseline_model = MRBNO(
        width=baseline_ckpt.get('width', 24) if isinstance(baseline_ckpt, dict) else 24,
        rb=baseline_rb,
        k_in=baseline_ckpt.get('k_in', 144) if isinstance(baseline_ckpt, dict) else 144,
        k_out=baseline_ckpt.get('k_out', 144) if isinstance(baseline_ckpt, dict) else 144,
        layers=baseline_ckpt.get('layers', 4) if isinstance(baseline_ckpt, dict) else 4,
    )
    if isinstance(baseline_ckpt, dict) and 'model_state_dict' in baseline_ckpt:
        baseline_model.load_state_dict(baseline_ckpt['model_state_dict'], strict=False)
    else:
        baseline_model.load_state_dict(baseline_ckpt, strict=False)
    baseline_model.eval()
    
    stage1_model = MRBNOStage1(
        width=stage1_ckpt.get('width', 24),
        rb=stage1_rb,
        k_in=stage1_ckpt.get('k_in', 144),
        k_out=stage1_ckpt.get('k_out', 144),
        layers=stage1_ckpt.get('layers', 4),
    )
    stage1_model.load_state_dict(stage1_ckpt['model_state_dict'])
    stage1_model.eval()
    
    print("\n" + "="*80)
    print("1. PCA SPACE WEIGHTS: Stage 1 vs Baseline")
    print("="*80)
    baseline_pca = extract_pca_weights(baseline_model)
    stage1_pca = extract_pca_weights(stage1_model)
    stage1_vs_baseline_pca = compare_weights(stage1_pca, baseline_pca, "Stage 1", "Baseline")
    print_comparison(stage1_vs_baseline_pca, "Stage 1", "Baseline")
    
    print("\n" + "="*80)
    print("2. PCA SPACE WEIGHTS: Stage 2 vs Baseline")
    print("="*80)
    
    stage2_names = args.stage2_names if args.stage2_names else \
                   [f"v{i+1}" for i in range(len(args.stage2_checkpoints))]
    
    for ckpt_path, name in zip(args.stage2_checkpoints, stage2_names):
        print(f"\nLoading Stage 2 {name}...")
        stage2_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(stage2_ckpt, dict) and 'rb_data' in stage2_ckpt:
            stage2_rb_data = stage2_ckpt.get('rb_data', stage2_ckpt.get('rb'))
            if isinstance(stage2_rb_data, dict):
                stage2_rb = PCAReducerMultiChannel()
                stage2_rb.pca_in = stage2_rb_data['pca_in']
                stage2_rb.pca_out = stage2_rb_data['pca_out']
                stage2_rb.n_channels = stage2_rb_data['n_channels']
                stage2_rb.out_channels = stage2_rb_data['out_channels']
                stage2_rb.shape_ = stage2_rb_data['shape_']
            else:
                stage2_rb = stage2_rb_data
        else:
            stage2_rb = stage1_rb
        stage2_model = MRBNO(
            width=stage1_ckpt.get('width', 24),
            rb=stage2_rb,
            k_in=stage1_ckpt.get('k_in', 144),
            k_out=stage1_ckpt.get('k_out', 144),
            layers=stage1_ckpt.get('layers', 4),
        )
        if isinstance(stage2_ckpt, dict) and 'model_state_dict' in stage2_ckpt:
            stage2_model.load_state_dict(stage2_ckpt['model_state_dict'], strict=False)
        else:
            stage2_model.load_state_dict(stage2_ckpt, strict=False)
        stage2_model.eval()
        
        stage2_pca = extract_pca_weights(stage2_model)
        stage2_vs_baseline_pca = compare_weights(stage2_pca, baseline_pca, f"Stage 2 {name}", "Baseline")
        print_comparison(stage2_vs_baseline_pca, f"Stage 2 {name}", "Baseline")
    
    print("\n" + "="*80)
    print("3. REAL SPACE WEIGHTS: Stage 2 vs Baseline")
    print("="*80)
    
    baseline_real = extract_real_space_weights(baseline_model)
    
    for ckpt_path, name in zip(args.stage2_checkpoints, stage2_names):
        print(f"\nLoading Stage 2 {name}...")
        stage2_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(stage2_ckpt, dict) and 'rb_data' in stage2_ckpt:
            stage2_rb_data = stage2_ckpt.get('rb_data', stage2_ckpt.get('rb'))
            if isinstance(stage2_rb_data, dict):
                stage2_rb = PCAReducerMultiChannel()
                stage2_rb.pca_in = stage2_rb_data['pca_in']
                stage2_rb.pca_out = stage2_rb_data['pca_out']
                stage2_rb.n_channels = stage2_rb_data['n_channels']
                stage2_rb.out_channels = stage2_rb_data['out_channels']
                stage2_rb.shape_ = stage2_rb_data['shape_']
            else:
                stage2_rb = stage2_rb_data
        else:
            stage2_rb = stage1_rb
        stage2_model = MRBNO(
            width=stage2_ckpt.get('width', 24),
            rb=stage2_rb,
            k_in=stage2_ckpt.get('k_in', 144),
            k_out=stage2_ckpt.get('k_out', 144),
            layers=stage2_ckpt.get('layers', 4),
        )
        if isinstance(stage2_ckpt, dict) and 'model_state_dict' in stage2_ckpt:
            stage2_model.load_state_dict(stage2_ckpt['model_state_dict'], strict=False)
        else:
            stage2_model.load_state_dict(stage2_ckpt, strict=False)
        stage2_model.eval()
        
        stage2_real = extract_real_space_weights(stage2_model)
        stage2_vs_baseline_real = compare_weights(stage2_real, baseline_real, f"Stage 2 {name}", "Baseline")
        print_comparison(stage2_vs_baseline_real, f"Stage 2 {name}", "Baseline")
        
        print("\n" + "-"*80)
        print("REAL SPACE WEIGHTS BREAKDOWN BY LAYER TYPE")
        print("-"*80)
        
        lift_proj_keys = [k for k in stage2_vs_baseline_real['common_keys'] if 'lift' in k or 'proj' in k]
        skip_conv_keys = [k for k in stage2_vs_baseline_real['common_keys'] if 'skip_conv' in k]
        global_skip_keys = [k for k in stage2_vs_baseline_real['common_keys'] if 'global_skip' in k]
        
        def print_layer_group(keys, group_name):
            if not keys:
                return
            print(f"\n{group_name}:")
            cos_sims = [stage2_vs_baseline_real['cosine_similarity'][k] for k in keys]
            rel_l2s = [stage2_vs_baseline_real['relative_l2_distance'][k] for k in keys]
            print(f"  Cosine Similarity: {np.mean(cos_sims):.6f} (min: {np.min(cos_sims):.6f}, max: {np.max(cos_sims):.6f})")
            print(f"  Relative L2 Distance: {np.mean(rel_l2s):.6f} (min: {np.min(rel_l2s):.6f}, max: {np.max(rel_l2s):.6f})")
            for k in keys:
                print(f"    {k:<50} cos_sim={stage2_vs_baseline_real['cosine_similarity'][k]:>8.6f} rel_l2={stage2_vs_baseline_real['relative_l2_distance'][k]:>8.6f}")
        
        print_layer_group(lift_proj_keys, "LIFT/PROJ LAYERS")
        print_layer_group(skip_conv_keys, "BLOCK SKIP CONV LAYERS")
        print_layer_group(global_skip_keys, "GLOBAL SKIP LAYERS")
        
        del stage2_model
        import gc
        gc.collect()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
