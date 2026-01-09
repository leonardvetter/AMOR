"""
Loss functions for training neural operators on irregular grids.

LpLoss adapted from Geo-FNO for irregular meshes.
"""

import torch
import torch.nn as nn
import numpy as np


class LpLoss(nn.Module):
    """
    Lp loss function adapted for irregular grids.
    
    For irregular grids, we approximate the mesh spacing h as 1/sqrt(N)
    where N is the number of nodes. This is similar to Geo-FNO's approach
    but adapted for point clouds.
    
    Args:
        d: Spatial dimension (default: 2 for 2D problems)
        p: Lp norm order (default: 2 for L2 loss)
        size_average: If True, return mean loss; if False, return sum
        reduction: If True, apply reduction; if False, return per-sample losses
    """
    def __init__(self, d=2, p=2, size_average=False, reduction=True):
        super(LpLoss, self).__init__()
        
        assert d > 0 and p > 0
        
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
    
    def abs(self, x, y):
        """
        Absolute Lp loss with mesh spacing normalization.
        
        Args:
            x: Predicted tensor (batch, ..., num_nodes)
            y: Target tensor (batch, ..., num_nodes)
            
        Returns:
            Loss value (scalar if reduction=True, else per-sample losses)
        """
        num_examples = x.size(0)
        num_nodes = x.size(-1)
        
        # Approximate mesh spacing for irregular grids
        # h = 1/sqrt(N) is a reasonable approximation
        h = 1.0 / np.sqrt(num_nodes)
        
        # Flatten all dimensions except batch
        x_flat = x.reshape(num_examples, -1)
        y_flat = y.reshape(num_examples, -1)
        
        # Compute Lp norm with mesh spacing normalization
        all_norms = (h ** (self.d / self.p)) * torch.norm(x_flat - y_flat, self.p, dim=1)
        
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        
        return all_norms
    
    def rel(self, x, y):
        """
        Relative Lp loss.
        
        Args:
            x: Predicted tensor (batch, ..., num_nodes)
            y: Target tensor (batch, ..., num_nodes)
            
        Returns:
            Relative loss value
        """
        num_examples = x.size(0)
        
        # Flatten all dimensions except batch
        x_flat = x.reshape(num_examples, -1)
        y_flat = y.reshape(num_examples, -1)
        
        diff_norms = torch.norm(x_flat - y_flat, self.p, dim=1)
        y_norms = torch.norm(y_flat, self.p, dim=1)
        
        # Avoid division by zero
        y_norms = torch.clamp(y_norms, min=1e-8)
        
        rel_loss = diff_norms / y_norms
        
        if self.reduction:
            if self.size_average:
                return torch.mean(rel_loss)
            else:
                return torch.sum(rel_loss)
        
        return rel_loss
    
    def forward(self, x, y):
        """
        Forward pass - uses relative loss by default.
        
        Args:
            x: Predicted tensor
            y: Target tensor
            
        Returns:
            Loss value
        """
        return self.rel(x, y)
    
    def rel_l1(self, x, y):
        """
        Relative L1 loss (L1 norm of difference divided by L1 norm of target).
        
        Args:
            x: Predicted tensor (batch, ..., num_nodes)
            y: Target tensor (batch, ..., num_nodes)
            
        Returns:
            Relative L1 loss value
        """
        num_examples = x.size(0)
        
        x_flat = x.reshape(num_examples, -1)
        y_flat = y.reshape(num_examples, -1)
        
        diff_l1 = torch.norm(x_flat - y_flat, p=1, dim=1)
        y_l1 = torch.norm(y_flat, p=1, dim=1)
        
        y_l1 = torch.clamp(y_l1, min=1e-8)
        
        rel_loss = diff_l1 / y_l1
        
        if self.reduction:
            if self.size_average:
                return torch.mean(rel_loss)
            else:
                return torch.sum(rel_loss)
        
        return rel_loss

