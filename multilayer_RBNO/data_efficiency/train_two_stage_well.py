#!/usr/bin/env python3
"""
Two-stage training on The Well datasets with streaming from HuggingFace.

Stage 1: Train in PCA space with loss on PCA coefficients.
Stage 2: Fine-tune on real-space data with frozen blocks.

Uses FNO benchmark hyperparameters from The Well benchmarks.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import argparse
import time
from sklearn.decomposition import IncrementalPCA

try:
    from the_well.data import WellDataset
except ImportError:
    print("Error: the_well package not found. Install with: pip install the_well")
    sys.exit(1)

from mrbno_two_stage import MRBNOStage1, MRBNOStage2


class PCAReducerMultiChannel:
    """Fit PCA bases for input/output tensors using incremental PCA for streaming."""
    
    def __init__(self, svd_solver="auto", whiten=False, random_state=None):
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.random_state = random_state
        self.pca_in = None
        self.pca_out = None
        self.n_channels = None
        self.out_channels = None
        self.shape_ = None
        
    def fit_incremental(self, dataset, n_samples_for_pca=1000):
        """
        Fit PCA using a sample of data from streaming dataset.
        """
        print(f"Fitting PCA on {n_samples_for_pca} samples...")
        
        sample_data = []
        sample_solutions = []
        
        for i, batch in enumerate(dataset):
            if i >= n_samples_for_pca:
                break
            sample_data.append(batch['input'].numpy())
            sample_solutions.append(batch['output'].numpy())
        
        X_sample = np.array(sample_data)
        Y_sample = np.array(sample_solutions)
        
        if X_sample.ndim == 3:
            X_sample = X_sample[:, np.newaxis, :, :]
        if Y_sample.ndim == 3:
            Y_sample = Y_sample[:, np.newaxis, :, :]
        
        num_samples, c_in, n, m = X_sample.shape
        self.n_channels = c_in
        self.shape_ = (n, m)
        
        _, c_out, n_out, m_out = Y_sample.shape
        if (n_out, m_out) != (n, m):
            raise ValueError("Input and output spatial dimensions must match")
        self.out_channels = c_out
        
        x_in = X_sample.reshape(num_samples, -1).astype(np.float32, copy=False)
        y_out = Y_sample.reshape(num_samples, -1).astype(np.float32, copy=False)
        
        from sklearn.decomposition import PCA
        self.pca_in = PCA(n_components=None, svd_solver=self.svd_solver,
                          whiten=self.whiten, random_state=self.random_state).fit(x_in)
        self.pca_out = PCA(n_components=None, svd_solver=self.svd_solver,
                           whiten=self.whiten, random_state=self.random_state).fit(y_out)
        
        return self
    
    @staticmethod
    def _k_from_cutoff(evr, cutoff):
        return int(np.searchsorted(np.cumsum(evr), cutoff) + 1)
    
    def k_from_cutoff(self, j_in=0.99, j_out=0.99):
        if self.pca_in is None or self.pca_out is None:
            raise ValueError("PCAReducer must be fitted before requesting cutoffs")
        k_in = self._k_from_cutoff(self.pca_in.explained_variance_ratio_, j_in)
        k_out = self._k_from_cutoff(self.pca_out.explained_variance_ratio_, j_out)
        return k_in, k_out
    
    @property
    def input_components(self):
        if self.pca_in is None:
            raise ValueError("PCAReducer has not been fitted")
        return self.pca_in.components_
    
    @property
    def output_components(self):
        if self.pca_out is None:
            raise ValueError("PCAReducer has not been fitted")
        return self.pca_out.components_
    
    @property
    def input_mean(self):
        return self.pca_in.mean_
    
    @property
    def output_mean(self):
        return self.pca_out.mean_


class WellDatasetWrapper(Dataset):
    """Wrapper for WellDataset to extract input-output pairs."""
    
    def __init__(self, well_dataset):
        self.well_dataset = well_dataset
        self.data = []
        
        print("Loading data from WellDataset...")
        for i, batch in enumerate(well_dataset):
            self.data.append({
                'input': batch['input'],
                'output': batch['output']
            })
            if (i + 1) % 1000 == 0:
                print(f"  Loaded {i + 1} samples...")
        
        print(f"Total samples loaded: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['input'], self.data[idx]['output']


def compute_pca_coefficients_batch(X, Y, rb, n_modes_in, n_modes_out):
    """Convert spatial data to PCA coefficients."""
    X_flat = X.reshape(len(X), -1)
    Y_flat = Y.reshape(len(Y), -1)
    
    X_centered = X_flat - rb.input_mean
    Y_centered = Y_flat - rb.output_mean
    
    X_pca = X_centered @ rb.input_components[:n_modes_in].T
    Y_pca = Y_centered @ rb.output_components[:n_modes_out].T
    
    X_pca = X_pca.reshape(len(X_pca), 1, n_modes_in)
    Y_pca = Y_pca.reshape(len(Y_pca), 1, n_modes_out)
    
    return X_pca, Y_pca


def train_stage1(model, train_loader, val_loader, epochs, lr, device):
    """Train Stage 1 model in PCA space."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
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
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Stage 1 Epoch {epoch+1:03d}/{epochs}] train={train_loss:.4e} val={val_loss:.4e} lr={scheduler.get_last_lr()[0]:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch+1})")
    
    return best_val_loss


def train_stage2(model, train_loader, val_loader, epochs, lr, device):
    """Train Stage 2 model in real space with frozen blocks."""
    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=3.0)
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
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Stage 2 Epoch {epoch+1:03d}/{epochs}] train={train_loss:.4e} val={val_loss:.4e} lr={scheduler.get_last_lr()[0]:.2e} (best: {best_val_loss:.4e} @ epoch {best_epoch+1})")
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Two-stage training on The Well datasets')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of The Well dataset (e.g., active_matter, MHD_256)')
    parser.add_argument('--split', type=str, default='train', help='Dataset split (train/val/test)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--stage1_epochs', type=int, default=300, help='Stage 1 epochs')
    parser.add_argument('--stage2_epochs', type=int, default=100, help='Stage 2 epochs')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--width', type=int, default=32, help='Model width')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory (default: $SCRATCH/checkpoints)')
    parser.add_argument('--n_pca_samples', type=int, default=1000, help='Number of samples for PCA fitting')
    args = parser.parse_args()
    
    # Set checkpoint directory to SCRATCH if not specified
    if args.checkpoint_dir is None:
        scratch_dir = os.environ.get('SCRATCH', '/tmp')
        args.checkpoint_dir = os.path.join(scratch_dir, 'checkpoints')
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset_name}, Split: {args.split}")
    
    print("\n" + "="*70)
    print("LOADING DATASET FROM HUGGINGFACE")
    print("="*70)
    
    try:
        well_dataset = WellDataset(
            well_base_path="hf://datasets/polymathic-ai/",
            well_dataset_name=args.dataset_name,
            well_split_name=args.split
        )
        print(f"Dataset loaded successfully from HuggingFace")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative path format...")
        well_dataset = WellDataset(
            well_base_path="hf://datasets/polymathic-ai/the_well",
            well_dataset_name=args.dataset_name,
            well_split_name=args.split
        )
    
    wrapped_dataset = WellDatasetWrapper(well_dataset)
    
    first_input, first_output = wrapped_dataset[0]
    if first_input.ndim == 2:
        first_input = first_input.unsqueeze(0)
    if first_output.ndim == 2:
        first_output = first_output.unsqueeze(0)
    
    H, W = first_input.shape[-2], first_input.shape[-1]
    in_channels = first_input.shape[0] if first_input.ndim == 3 else first_input.shape[1]
    out_channels = first_output.shape[0] if first_output.ndim == 3 else first_output.shape[1]
    
    print(f"\nDataset shape: input={first_input.shape}, output={first_output.shape}")
    print(f"Spatial dimensions: H={H}, W={W}")
    print(f"Channels: in={in_channels}, out={out_channels}")
    
    print("\n" + "="*70)
    print("COMPUTING PCA")
    print("="*70)
    
    sample_dataset = WellDataset(
        well_base_path="hf://datasets/polymathic-ai/",
        well_dataset_name=args.dataset_name,
        well_split_name=args.split
    )
    
    rb = PCAReducerMultiChannel().fit_incremental(sample_dataset, n_samples_for_pca=args.n_pca_samples)
    k_in, k_out = rb.k_from_cutoff(j_in=0.99, j_out=0.99)
    print(f"PCA modes for 99% variance: k_in={k_in}, k_out={k_out}")
    
    print("\n" + "="*70)
    print("PREPARING DATA")
    print("="*70)
    
    all_inputs = []
    all_outputs = []
    for i in range(len(wrapped_dataset)):
        inp, out = wrapped_dataset[i]
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        if out.ndim == 2:
            out = out.unsqueeze(0)
        all_inputs.append(inp.numpy())
        all_outputs.append(out.numpy())
    
    X = np.array(all_inputs)
    Y = np.array(all_outputs)
    
    if X.ndim == 3:
        X = X[:, np.newaxis, :, :]
    if Y.ndim == 3:
        Y = Y[:, np.newaxis, :, :]
    
    train_frac = 0.8
    n_train = int(train_frac * len(X))
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]
    
    print(f"Train/Val split: {len(X_train)} / {len(X_val)}")
    
    x_mean = X_train.mean()
    x_std = X_train.std()
    y_mean = Y_train.mean()
    y_std = Y_train.std()
    
    X_train = (X_train - x_mean) / (x_std + 1e-8)
    Y_train = (Y_train - y_mean) / (y_std + 1e-8)
    X_val = (X_val - x_mean) / (x_std + 1e-8)
    Y_val = (Y_val - y_mean) / (y_std + 1e-8)
    
    print("\n" + "="*70)
    print("STAGE 1: Training in PCA space")
    print("="*70)
    
    X_train_pca, Y_train_pca = compute_pca_coefficients_batch(X_train, Y_train, rb, k_in, k_out)
    X_val_pca, Y_val_pca = compute_pca_coefficients_batch(X_val, Y_val, rb, k_in, k_out)
    
    train_loader_pca = DataLoader(
        TensorDataset(torch.from_numpy(X_train_pca).float(), torch.from_numpy(Y_train_pca).float()),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader_pca = DataLoader(
        TensorDataset(torch.from_numpy(X_val_pca).float(), torch.from_numpy(Y_val_pca).float()),
        batch_size=args.batch_size, shuffle=False
    )
    
    stage1_model = MRBNOStage1(
        in_channels=in_channels, out_channels=out_channels, width=args.width,
        n_modes_in=k_in, n_modes_out=k_out, n_layers=args.n_layers, H=H, W=W,
        norm="group", dropout_p=0.0, residual=True,
        pca_components_in=rb.input_components, pca_mean_in=rb.input_mean,
        pca_components_out=rb.output_components, pca_mean_out=rb.output_mean
    )
    
    train_stage1(stage1_model, train_loader_pca, val_loader_pca, args.stage1_epochs, args.lr, device)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    stage1_path = os.path.join(args.checkpoint_dir, f'stage1_{args.dataset_name}.pt')
    torch.save(stage1_model.state_dict(), stage1_path)
    print(f"Stage 1 checkpoint saved to {stage1_path}")
    
    print("\n" + "="*70)
    print("STAGE 2: Fine-tuning in real space")
    print("="*70)
    
    stage2_model = MRBNOStage2(stage1_model, in_channels=in_channels, out_channels=out_channels, width=args.width, add_coords=False)
    
    n_stage2_samples = min(1000, len(X_train))
    X_train_stage2 = X_train[:n_stage2_samples]
    Y_train_stage2 = Y_train[:n_stage2_samples]
    X_val_stage2 = X_val[:min(200, len(X_val))]
    Y_val_stage2 = Y_val[:min(200, len(X_val))]
    
    train_loader_stage2 = DataLoader(
        TensorDataset(torch.from_numpy(X_train_stage2).float(), torch.from_numpy(Y_train_stage2).float()),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader_stage2 = DataLoader(
        TensorDataset(torch.from_numpy(X_val_stage2).float(), torch.from_numpy(Y_val_stage2).float()),
        batch_size=args.batch_size, shuffle=False
    )
    
    train_stage2(stage2_model, train_loader_stage2, val_loader_stage2, args.stage2_epochs, args.lr, device)
    
    stage2_path = os.path.join(args.checkpoint_dir, f'stage2_{args.dataset_name}.pt')
    torch.save(stage2_model.state_dict(), stage2_path)
    print(f"Stage 2 checkpoint saved to {stage2_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
