#!/usr/bin/env python3
"""
Plot and analyze ablation study results.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def extract_test_metrics(log_file):
    """Extract test metrics from log file."""
    if not os.path.exists(log_file):
        return None
    
    rel_l1 = None
    rel_l2 = None
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'Test relative L1 error:' in line:
                match = re.search(r'Test relative L1 error:\s+([\d.e+-]+)', line)
                if match:
                    rel_l1 = float(match.group(1))
            if 'Test relative L2 error:' in line:
                match = re.search(r'Test relative L2 error:\s+([\d.e+-]+)', line)
                if match:
                    rel_l2 = float(match.group(1))
    
    return {'rel_l1': rel_l1, 'rel_l2': rel_l2} if (rel_l1 is not None and rel_l2 is not None) else None


def plot_pca_epsilon_ablation(log_dir, output_dir):
    """Plot PCA epsilon ablation study results."""
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    rel_l1_values = []
    rel_l2_values = []
    
    for eps in epsilons:
        eps_name = f"{eps:.0e}".replace('-0', '-')
        log_file = os.path.join(log_dir, f'stage2_v1_eps{eps_name}.log')
        metrics = extract_test_metrics(log_file)
        if metrics:
            rel_l1_values.append(metrics['rel_l1'])
            rel_l2_values.append(metrics['rel_l2'])
        else:
            rel_l1_values.append(np.nan)
            rel_l2_values.append(np.nan)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.semilogx(epsilons, rel_l1_values, 'o-', linewidth=2, markersize=8, label='Rel L1')
    ax1.set_xlabel('PCA Epsilon', fontsize=12)
    ax1.set_ylabel('Test Relative L1 Error', fontsize=12)
    ax1.set_title('PCA Epsilon Ablation: Relative L1 Error', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.semilogx(epsilons, rel_l2_values, 'o-', linewidth=2, markersize=8, label='Rel L2', color='orange')
    ax2.set_xlabel('PCA Epsilon', fontsize=12)
    ax2.set_ylabel('Test Relative L2 Error', fontsize=12)
    ax2.set_title('PCA Epsilon Ablation: Relative L2 Error', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'ablation_pca_epsilon.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()
    
    return {'epsilons': epsilons, 'rel_l1': rel_l1_values, 'rel_l2': rel_l2_values}


def plot_data_split_ablation(log_dir, output_dir):
    """Plot data split ablation study results."""
    splits = [
        (0.5, '50_50'),
        (0.7, '70_30'),
        (0.9, '90_10'),
        (0.99, '99_1')
    ]
    
    stage1_ratios = []
    rel_l1_values = []
    rel_l2_values = []
    
    for stage1_ratio, split_name in splits:
        log_file = os.path.join(log_dir, f'stage2_v1_{split_name}.log')
        metrics = extract_test_metrics(log_file)
        if metrics:
            stage1_ratios.append(stage1_ratio)
            rel_l1_values.append(metrics['rel_l1'])
            rel_l2_values.append(metrics['rel_l2'])
        else:
            stage1_ratios.append(stage1_ratio)
            rel_l1_values.append(np.nan)
            rel_l2_values.append(np.nan)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(stage1_ratios, rel_l1_values, 'o-', linewidth=2, markersize=8, label='Rel L1')
    ax1.set_xlabel('Stage 1 Data Ratio', fontsize=12)
    ax1.set_ylabel('Test Relative L1 Error', fontsize=12)
    ax1.set_title('Data Split Ablation: Relative L1 Error', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(stage1_ratios, rel_l2_values, 'o-', linewidth=2, markersize=8, label='Rel L2', color='orange')
    ax2.set_xlabel('Stage 1 Data Ratio', fontsize=12)
    ax2.set_ylabel('Test Relative L2 Error', fontsize=12)
    ax2.set_title('Data Split Ablation: Relative L2 Error', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'ablation_data_split.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()
    
    return {'stage1_ratios': stage1_ratios, 'rel_l1': rel_l1_values, 'rel_l2': rel_l2_values}


def main():
    parser = argparse.ArgumentParser(description='Plot ablation study results')
    parser.add_argument('--scratch_dir', type=str, default=None,
                       help='Scratch directory (default: $SCRATCH/data_efficiency)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: scratch_dir/plots)')
    args = parser.parse_args()
    
    if args.scratch_dir is None:
        scratch_dir = '/users/PAS3244/dvoderhol/data/data_efficiency'
    else:
        scratch_dir = args.scratch_dir
    
    if args.output_dir is None:
        output_dir = os.path.join(scratch_dir, 'plots')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("ABLATION STUDY RESULTS PLOTTING")
    print("="*70)
    
    pca_epsilon_log_dir = os.path.join(scratch_dir, 'logs', 'ablation_pca_epsilon')
    data_split_log_dir = os.path.join(scratch_dir, 'logs', 'ablation_data_split')
    
    print("\n1. PCA Epsilon Ablation Study")
    print("-" * 70)
    if os.path.exists(pca_epsilon_log_dir):
        pca_results = plot_pca_epsilon_ablation(pca_epsilon_log_dir, output_dir)
        print(f"Epsilons: {pca_results['epsilons']}")
        print(f"Rel L1: {pca_results['rel_l1']}")
        print(f"Rel L2: {pca_results['rel_l2']}")
    else:
        print(f"Log directory not found: {pca_epsilon_log_dir}")
    
    print("\n2. Data Split Ablation Study")
    print("-" * 70)
    if os.path.exists(data_split_log_dir):
        split_results = plot_data_split_ablation(data_split_log_dir, output_dir)
        print(f"Stage 1 Ratios: {split_results['stage1_ratios']}")
        print(f"Rel L1: {split_results['rel_l1']}")
        print(f"Rel L2: {split_results['rel_l2']}")
    else:
        print(f"Log directory not found: {data_split_log_dir}")
    
    print("\n" + "="*70)
    print("PLOTTING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
