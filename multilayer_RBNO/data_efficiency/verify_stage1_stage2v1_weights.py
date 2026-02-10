#!/usr/bin/env python3
"""
Verify that Stage 1 and Stage 2 v1 have identical PCA space weights (since v1 freezes blocks).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import torch
import argparse
from analyze_weight_similarity import extract_pca_weights, compare_weights, print_comparison
from train_two_stage import MRBNOStage1, PCAReducerMultiChannel
from MRBNO import MRBNO


def main():
    parser = argparse.ArgumentParser(description='Verify Stage 1 and Stage 2 v1 have identical PCA weights')
    parser.add_argument('--stage1_checkpoint', type=str, required=True)
    parser.add_argument('--stage2_v1_checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    print("="*80)
    print("VERIFYING STAGE 1 vs STAGE 2 V1 PCA SPACE WEIGHTS")
    print("="*80)
    print("(Stage 2 v1 should have identical PCA weights since blocks are frozen)")
    print()
    
    print("Loading Stage 1 model...")
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
    stage1_model = MRBNOStage1(
        width=stage1_ckpt.get('width', 24),
        rb=stage1_rb,
        k_in=stage1_ckpt.get('k_in', 144),
        k_out=stage1_ckpt.get('k_out', 144),
        layers=stage1_ckpt.get('layers', 4),
    )
    stage1_model.load_state_dict(stage1_ckpt['model_state_dict'])
    stage1_model.eval()
    
    print("Loading Stage 2 v1 model...")
    stage2_ckpt = torch.load(args.stage2_v1_checkpoint, map_location='cpu', weights_only=False)
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
    
    stage1_pca = extract_pca_weights(stage1_model)
    stage2_pca = extract_pca_weights(stage2_model)
    
    comparison = compare_weights(stage1_pca, stage2_pca, "Stage 1", "Stage 2 v1")
    print_comparison(comparison, "Stage 1", "Stage 2 v1")
    
    print("\n" + "="*80)
    print("VERIFICATION RESULT")
    print("="*80)
    
    all_identical = True
    for key in comparison['common_keys']:
        l2_dist = comparison['l2_distance'][key]
        cos_sim = comparison['cosine_similarity'][key]
        if l2_dist > 1e-6:
            print(f"WARNING: {key} is NOT identical (L2 dist = {l2_dist:.2e})")
            all_identical = False
    
    if all_identical:
        print("✓ All PCA space weights are IDENTICAL (as expected for frozen blocks)")
    else:
        print("✗ Some PCA space weights differ (this should not happen if blocks are frozen!)")
    
    print()


if __name__ == '__main__':
    main()
