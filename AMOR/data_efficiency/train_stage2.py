#!/usr/bin/env python3
"""
Stage 2 training: Fine-tune complete AMOR.py model on 10% of train/val data.
Three versions:
  - Version 1: Only new weights trained (frozen blocks)
  - Version 2: All weights trained, higher LR for new weights
  - Version 3: All weights trained equally
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AMOR.paths import expand  # noqa: E402

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from train_two_stage import (
    load_nonlinear_dataset, PCAReducerMultiChannel, AMORStage1,
    train_stage2, train_stage2_progressive_unfreeze, train_stage2_lr_warmup,
    train_stage2_knowledge_distillation, initialize_lift_proj_from_pca,
    train_stage2_lift_proj_high_lr, train_stage2_proj_high_lr,
    train_stage2_lift_proj_first, train_stage2_phased_lift_proj,
    evaluate_model
)


def main():
    parser = argparse.ArgumentParser(description='Stage 2 training')
    parser.add_argument('--data_path', type=str, default=expand('${AMOR_SCRATCH_DATA}/nonlinear_fvar_5_flen_0.2_mvar_1_mlen_0.3_big.npz'))  # rename path on a different cluster
    parser.add_argument('--stage1_checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lr_blocks_factor', type=float, default=0.1,
                       help='LR multiplier for blocks (Version 2). Use 1.0 for Version 3.')
    parser.add_argument('--version', type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], default=1,
                       help='1=frozen blocks, 8=lift/proj high LR, 11=proj only high LR, ...')
    parser.add_argument('--stage1_ratio', type=float, default=0.9,
                       help='Fraction of train/val data used for Stage 1 (must match Stage 1 training, default: 0.9)')
    parser.add_argument('--lift_proj_lr_factor', type=float, default=5.0,
                       help='LR multiplier for lift/proj in version 8 (default: 5.0)')
    parser.add_argument('--proj_lr_factor', type=float, default=5.0,
                       help='LR multiplier for proj only in version 11 (default: 5.0)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help=expand('Directory for Stage 2 checkpoints (default: ${AMOR_DATA_ROOT}/data_efficiency/checkpoints)'))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--train_loss',
        type=str,
        choices=('l1', 'mse'),
        default='l1',
        help="Optimization objective for Stage 2 (default: l1). Best checkpoint always uses val_rel_l1.",
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help="Skip training; load stage2_{version}_best.pt from --checkpoint_dir and print train/val/test "
        "relative L1/L2 via evaluate_model (versions 1–3 only).",
    )
    args = parser.parse_args()
    
    if args.checkpoint_dir is not None:
        checkpoint_dir = os.path.expandvars(args.checkpoint_dir)
    else:
        checkpoint_dir = expand('${AMOR_DATA_ROOT}/data_efficiency/checkpoints')
    log_dir = expand('${AMOR_DATA_ROOT}/data_efficiency/logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    s1_path = os.path.abspath(os.path.expandvars(args.stage1_checkpoint))
    if not os.path.isfile(s1_path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {s1_path}")
    if os.path.getsize(s1_path) < 512:
        raise RuntimeError(
            f"Stage 1 checkpoint appears corrupt or empty ({os.path.getsize(s1_path)} bytes): {s1_path}"
        )
    try:
        checkpoint = torch.load(s1_path, map_location=device, weights_only=False)
    except (OSError, EOFError, RuntimeError) as e:
        raise RuntimeError(
            f"Failed to load Stage 1 checkpoint (file may be truncated): {s1_path}\n  {e}"
        ) from e
    
    (X_train, Y_train, X_val, Y_val, X_test, Y_test), norm_stats = load_nonlinear_dataset(args.data_path)
    
    print("\n" + "="*70)
    print(f"STAGE 2 DATA SPLIT: {(1-args.stage1_ratio)*100:.0f}% of train/val for Stage 2")
    print("="*70)
    
    n_train_val = len(X_train) + len(X_val)
    n_stage1 = int(args.stage1_ratio * n_train_val)
    n_stage2 = n_train_val - n_stage1
    n_stage2_train = int(0.7 * n_stage2)
    
    X_stage2_all = np.concatenate([X_train, X_val], axis=0)[n_stage1:]
    Y_stage2_all = np.concatenate([Y_train, Y_val], axis=0)[n_stage1:]
    
    X_stage2_train = X_stage2_all[:n_stage2_train]
    Y_stage2_train = Y_stage2_all[:n_stage2_train]
    X_stage2_val = X_stage2_all[n_stage2_train:]
    Y_stage2_val = Y_stage2_all[n_stage2_train:]
    
    print(f"Stage 2 training samples: {len(X_stage2_train)}")
    print(f"Stage 2 validation samples: {len(X_stage2_val)}")
    
    rb_data = checkpoint['rb_data']
    rb = PCAReducerMultiChannel()
    rb.pca_in = rb_data['pca_in']
    rb.pca_out = rb_data['pca_out']
    rb.n_channels = rb_data['n_channels']
    rb.out_channels = rb_data['out_channels']
    rb.shape_ = rb_data['shape_']
    k_in = checkpoint['k_in']
    k_out = checkpoint['k_out']
    width = checkpoint['width']
    layers = checkpoint['layers']
    
    print("\n" + "="*70)
    print(f"STAGE 2 VERSION {args.version}: Fine-tuning complete AMOR")
    print("="*70)
    
    from AMOR.models.amor import AMOR
    
    stage1_model = AMORStage1(
        width=width, rb=rb, k_in=k_in, k_out=k_out, layers=layers,
        dropout=0.0, init_scale=0.1, use_skip_connections=True,
        use_batch_norm=False, add_constant_mode=False, use_factorized_weights=False
    )
    stage1_model.load_state_dict(checkpoint['model_state_dict'])
    
    freeze_blocks = (args.version in [1, 8, 9, 10, 11])
    lr_blocks_factor = args.lr_blocks_factor if args.version == 2 else 1.0
    
    stage2_base = AMOR(
        width=width, rb=rb, k_in=k_in, k_out=k_out, layers=layers,
        dropout=0.0, init_scale=0.1, use_skip_connections=True,
        use_batch_norm=False, use_coordinates=False, add_constant_mode=False,
        use_factorized_weights=False
    )
    
    stage1_state = stage1_model.state_dict()
    stage2_state = stage2_base.state_dict()
    
    copied_weights = []
    for name, param in stage1_state.items():
        if name in stage2_state:
            if name.startswith('block') or name.startswith('gate'):
                stage2_state[name] = param.clone()
                copied_weights.append(name)
    
    stage2_base.load_state_dict(stage2_state, strict=False)
    print(f"Copied {len(copied_weights)} PCA space weights from Stage 1 to Stage 2")
    
    if args.version not in (1, 2, 3) and args.train_loss != 'mse':
        raise SystemExit("--train_loss l1 is wired for Stage 2 versions 1, 2, and 3 only.")

    if args.version == 7:
        initialize_lift_proj_from_pca(stage2_base, rb, k_in, k_out, width, device)
        print("Version 7: Better initialization - initialized lift/proj from PCA")
    
    if freeze_blocks:
        frozen_params = []
        for param in stage2_base.block0.parameters():
            param.requires_grad = False
            frozen_params.append(param)
        for param in stage2_base.block1.parameters():
            param.requires_grad = False
            frozen_params.append(param)
        for block in stage2_base.blocks:
            for param in block.parameters():
                param.requires_grad = False
                frozen_params.append(param)
        for param in stage2_base.gate0.parameters():
            param.requires_grad = False
            frozen_params.append(param)
        for param in stage2_base.gate1.parameters():
            param.requires_grad = False
            frozen_params.append(param)
        for gate in stage2_base.gates:
            for param in gate.parameters():
                param.requires_grad = False
                frozen_params.append(param)
        print(f"Frozen {len(frozen_params)} block/gate parameters (requires_grad=False)")
        
        trainable_params = [p for p in stage2_base.parameters() if p.requires_grad]
        print(f"Trainable parameters: {len(trainable_params)}")
        for name, param in stage2_base.named_parameters():
            if param.requires_grad:
                print(f"  - {name}")
    
    model = stage2_base
    
    version_name = f"v{args.version}"
    if args.version == 1:
        print("Version 1: Only new weights trained (blocks frozen)")
    elif args.version == 2:
        print(f"Version 2: All weights trained, blocks LR = {args.lr * lr_blocks_factor:.2e} (factor {lr_blocks_factor})")
    elif args.version == 3:
        print("Version 3: All weights trained equally")
    elif args.version == 4:
        print("Version 4: Progressive unfreezing (blocks unfrozen at 25% of training)")
    elif args.version == 5:
        print("Version 5: LR warmup for new layers")
    elif args.version == 6:
        print("Version 6: Knowledge distillation from Stage 1")
    elif args.version == 7:
        print("Version 7: Better initialization (lift/proj initialized from PCA)")
    elif args.version == 8:
        print(f"Version 8: Frozen blocks, higher LR for lift/proj ({args.lift_proj_lr_factor}x)")
    elif args.version == 9:
        print("Version 9: Frozen blocks, first train lift/proj/global_skip, then add skip_conv")
    elif args.version == 10:
        print("Version 10: Frozen blocks, phased: proj -> lift -> skip_conv -> global_skip")
    elif args.version == 11:
        print(f"Version 11: Frozen blocks, higher LR for proj only ({args.proj_lr_factor}x)")
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_stage2_train).float(),
                     torch.from_numpy(Y_stage2_train).float()),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_stage2_val).float(),
                     torch.from_numpy(Y_stage2_val).float()),
        batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float(),
                     torch.from_numpy(Y_test).float()),
        batch_size=args.batch_size, shuffle=False
    )
    
    if args.eval_only:
        if args.version not in (1, 2, 3):
            raise SystemExit("--eval_only is implemented for Stage 2 versions 1, 2, and 3 only.")
        best_path = os.path.join(checkpoint_dir, f'stage2_{version_name}_best.pt')
        if not os.path.isfile(best_path):
            raise FileNotFoundError(f"--eval_only: expected checkpoint {best_path}")
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
        print("\n" + "=" * 70)
        print(f"EVAL_ONLY: loaded {best_path}")
        print(
            "Metrics: relative L1/L2 after undoing Y z-score "
            "(see train_two_stage.relative_l1_loss / relative_l2_error)."
        )
        print("=" * 70)
        _ym, _ys = float(norm_stats[2]), float(norm_stats[3])
        train_rel_l1, train_rel_l2 = evaluate_model(
            model, train_loader, device, is_stage1=False, y_mean=_ym, y_std=_ys
        )
        val_rel_l1, val_rel_l2 = evaluate_model(
            model, val_loader, device, is_stage1=False, y_mean=_ym, y_std=_ys
        )
        test_rel_l1, test_rel_l2 = evaluate_model(
            model, test_loader, device, is_stage1=False, y_mean=_ym, y_std=_ys
        )
        print(f"Train relative L1: {train_rel_l1:.4e}  Train relative L2: {train_rel_l2:.4e}")
        print(f"Val relative L1:   {val_rel_l1:.4e}  Val relative L2:   {val_rel_l2:.4e}")
        print(f"Test relative L1:  {test_rel_l1:.4e}  Test relative L2:  {test_rel_l2:.4e}")
        return
    
    if args.version == 4:
        train_stage2_progressive_unfreeze(model, train_loader, val_loader, args.epochs, args.lr,
                                          device, checkpoint_dir, version_name, start_epoch=0)
    elif args.version == 8:
        train_stage2_lift_proj_high_lr(model, train_loader, val_loader, args.epochs, args.lr,
                                       device, checkpoint_dir, version_name, start_epoch=0,
                                       lift_proj_lr_factor=args.lift_proj_lr_factor)
    elif args.version == 11:
        train_stage2_proj_high_lr(model, train_loader, val_loader, args.epochs, args.lr,
                                  device, checkpoint_dir, version_name, start_epoch=0,
                                  proj_lr_factor=args.proj_lr_factor)
    elif args.version == 9:
        train_stage2_lift_proj_first(model, train_loader, val_loader, args.epochs, args.lr,
                                     device, checkpoint_dir, version_name, start_epoch=0)
    elif args.version == 10:
        train_stage2_phased_lift_proj(model, train_loader, val_loader, args.epochs, args.lr,
                                      device, checkpoint_dir, version_name, start_epoch=0)
    elif args.version == 5:
        train_stage2_lr_warmup(model, train_loader, val_loader, args.epochs, args.lr,
                              device, checkpoint_dir, version_name, start_epoch=0)
    elif args.version == 6:
        train_stage2_knowledge_distillation(model, stage1_model, train_loader, val_loader, args.epochs, args.lr,
                                           device, checkpoint_dir, version_name, rb, k_in, k_out, start_epoch=0)
    else:
        train_stage2(
            model,
            train_loader,
            val_loader,
            args.epochs,
            args.lr,
            lr_blocks_factor,
            device,
            checkpoint_dir,
            version_name,
            start_epoch=0,
            train_loss=args.train_loss,
        )
    
    best_path = os.path.join(checkpoint_dir, f'stage2_{version_name}_best.pt')
    if os.path.isfile(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
    
    print("\n" + "="*70)
    print("EVALUATING BEST CHECKPOINT (train / val / test)")
    print("="*70)
    print(
        "Relative L1/L2 after undoing Y z-score (same formulas as train_baseline.py). "
        "Test set is the held-out split from load_nonlinear_dataset; "
        "Stage 2 train/val are only the post-stage1 slice."
    )
    _ym, _ys = float(norm_stats[2]), float(norm_stats[3])
    train_rel_l1, train_rel_l2 = evaluate_model(
        model, train_loader, device, is_stage1=False, y_mean=_ym, y_std=_ys
    )
    val_rel_l1, val_rel_l2 = evaluate_model(
        model, val_loader, device, is_stage1=False, y_mean=_ym, y_std=_ys
    )
    test_rel_l1, test_rel_l2 = evaluate_model(
        model, test_loader, device, is_stage1=False, y_mean=_ym, y_std=_ys
    )
    print(f"Train relative L1: {train_rel_l1:.4e}  Train relative L2: {train_rel_l2:.4e}")
    print(f"Val relative L1:   {val_rel_l1:.4e}  Val relative L2:   {val_rel_l2:.4e}")
    print(f"Test relative L1:  {test_rel_l1:.4e}  Test relative L2:  {test_rel_l2:.4e}")
    
    if freeze_blocks:
        print("\n" + "="*70)
        print(f"VERIFYING FROZEN BLOCKS: Stage 1 vs Stage 2 {version_name} PCA Weights")
        print("="*70)
        # Inline check (no external analysis helper).
        stage1_sd = stage1_model.state_dict()
        stage2_sd = model.state_dict()
        all_identical = True
        max_diff = 0.0
        for key, w1 in stage1_sd.items():
            if "pca" not in key.lower() and "basis" not in key.lower():
                continue
            if key not in stage2_sd:
                continue
            w2 = stage2_sd[key]
            if not torch.is_floating_point(w1):
                continue
            diff = torch.norm(w1.detach().cpu().float() - w2.detach().cpu().float()).item()
            max_diff = max(max_diff, diff)
            if diff > 1e-6:
                print(f"WARNING: {key} differs by {diff:.2e}")
                all_identical = False

        if all_identical:
            print(f"All PCA/basis weights IDENTICAL (max diff: {max_diff:.2e})")
        else:
            print(f"Some PCA/basis weights differ (max diff: {max_diff:.2e})")


if __name__ == '__main__':
    main()
