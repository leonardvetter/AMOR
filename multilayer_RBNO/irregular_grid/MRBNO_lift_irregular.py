"""
MRBNO Lift Irregular: Architecture for arbitrary grids/geometries using PCA basis.

Uses PCA encoding/decoding on point clouds. Assumes constant mesh (same node positions
across all samples) but arbitrary node positions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PCASpectralConv1d(nn.Module):
    """
    PCA-based spectral convolution for point clouds.
    
    Instead of FFT → multiply in frequency domain → IFFT,
    this does: encode to PCA → multiply in PCA space → decode.
    
    The PCA basis is computed once over the whole dataset and shared across all blocks.
    Supports material-adaptive PCA basis transformation.
    """
    def __init__(self, in_channels, out_channels, n_modes: int, pca_components, pca_mean, 
                 material_adaptive=False, material_prop_dim=None, width=None):
        """
        Args:
            in_channels: Input channel dimension
            out_channels: Output channel dimension
            n_modes: Number of PCA modes to keep
            pca_components: PCA components matrix (n_modes, N) - numpy array
            pca_mean: PCA mean vector (N,) - numpy array
            material_adaptive: If True, use material-adaptive PCA basis
            material_prop_dim: Material property dimension (required if material_adaptive=True)
            width: Model width (required if material_adaptive=True)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        max_available_modes = pca_components.shape[0]
        self.n_modes = min(n_modes, max_available_modes)
        
        if n_modes > max_available_modes:
            print(f"Warning: Requested {n_modes} modes but only {max_available_modes} available. Using {self.n_modes}.")
        
        self.register_buffer('pca_components', torch.from_numpy(pca_components[:self.n_modes, :].astype(np.float32)))
        self.register_buffer('pca_mean', torch.from_numpy(pca_mean.astype(np.float32)))
        
        self.material_adaptive = material_adaptive
        if material_adaptive:
            if material_prop_dim is None or width is None:
                raise ValueError("material_prop_dim and width required for material_adaptive=True")
            self.material_pca_transform = nn.Sequential(
                nn.Linear(material_prop_dim, width),
                nn.GELU(),
                nn.Linear(width, self.n_modes * self.n_modes)
            )
        else:
            self.material_pca_transform = None
        
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, self.n_modes) * scale
        )
    
    def forward(self, x, material_props=None):
        """
        x: (B, C_in, N) - point cloud data
        material_props: (B, material_prop_dim) - optional material properties
        returns: (B, C_out, N)
        """
        B, C, N = x.shape
        
        if self.material_adaptive and material_props is not None:
            T_flat = self.material_pca_transform(material_props)  # (B, n_modes * n_modes)
            T = T_flat.reshape(B, self.n_modes, self.n_modes)  # (B, n_modes, n_modes)
            pca_components_adapted = torch.einsum('bmn,mk->bnk', T, self.pca_components)  # (B, n_modes, N)
        else:
            pca_components_adapted = self.pca_components.unsqueeze(0)  # (1, n_modes, N)
        
        x_centered = x - self.pca_mean.unsqueeze(0).unsqueeze(0)  # (B, C_in, N)
        x_pca = torch.einsum('bcn,bmn->bcm', x_centered, pca_components_adapted)  # (B, C_in, n_modes)
        
        out_pca = torch.einsum('bcm,com->bom', x_pca, self.weight)  # (B, C_out, n_modes)
        
        out_flat = torch.einsum('bom,bmn->bon', out_pca, pca_components_adapted)  # (B, C_out, N)
        out_flat = out_flat + self.pca_mean.unsqueeze(0).unsqueeze(0)
        
        return out_flat


class MRBNOBlock1d(nn.Module):
    """
    MRBNO block for point clouds.
    
    Uses PCASpectralConv1d instead of SpectralConv2d. Skip/residual structure is the same.
    """
    def __init__(self, width, n_modes, pca_components, pca_mean, norm="group", dropout_p=0.0, residual=True,
                 material_adaptive=False, material_prop_dim=None, width_for_pca=None):
        super().__init__()
        self.spectral = PCASpectralConv1d(width, width, n_modes, pca_components, pca_mean,
                                         material_adaptive=material_adaptive,
                                         material_prop_dim=material_prop_dim,
                                         width=width_for_pca)
        self.w = nn.Conv1d(width, width, kernel_size=1)
        self.act = nn.GELU()
        self.residual = residual

        if norm == "group":
            num_groups = min(8, width)
            for g in range(min(8, width), 0, -1):
                if width % g == 0:
                    num_groups = g
                    break
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=width)
        elif norm == "instance":
            self.norm = nn.InstanceNorm1d(width, affine=True)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(width)
        else:
            self.norm = nn.Identity()

        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x, material_props=None):
        """
        x: (B, width, N)
        material_props: (B, material_prop_dim) - optional material properties
        returns: (B, width, N)
        """
        y = self.spectral(x, material_props=material_props) + self.w(x)
        y = self.norm(y)
        y = self.act(y)              
        y = self.dropout(y)
        return x + y if self.residual else y


class MRBNOLiftIrregular(nn.Module):
    """
    MRBNO Lift Irregular: Architecture using PCA basis for arbitrary grids.
    
    Structure is similar to FNO2d/MRBNOLiftSimple, adapted for point clouds.
    Assumes constant mesh (same node positions for all samples).
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        width: Hidden dimension width
        n_modes: Number of PCA modes to keep
        n_layers: Number of MRBNO blocks (default: 5 to match Geo-FNO)
        add_coords: Whether to add coordinate channels
        use_grid_skip: Whether to use grid-based skip connections (like Geo-FNO b layers)
        use_input_skip: Whether to use direct skip connection from input to output
        norm: Normalization type ("group", "instance", "batch", or None)
        dropout_p: Dropout probability
        residual: Whether to use residual connections
        pca_components: PCA components matrix (n_components, N) - numpy array
        pca_mean: PCA mean vector (N,) - numpy array
        mesh_coords: Mesh coordinates (N, 2) or (N, 3) - numpy array
    """
    def __init__(self, in_channels=1, out_channels=1, width=32, n_modes=256, n_layers=5,
                 add_coords=False, use_grid_skip=False, use_input_skip=False,
                 norm="group", dropout_p=0.0, residual=True,
                 pca_components=None, pca_mean=None, mesh_coords=None,
                 material_prop_dim=None, num_nodes=None,
                 use_multiplicative_conditioning=False, material_adaptive_pca=False):
        super().__init__()
        
        if pca_components is None or pca_mean is None:
            raise ValueError("pca_components and pca_mean must be provided")
        
        if add_coords and mesh_coords is None:
            raise ValueError("mesh_coords must be provided when add_coords=True")
        
        if use_grid_skip and mesh_coords is None:
            raise ValueError("mesh_coords must be provided when use_grid_skip=True")
        
        self.width = width
        self.add_coords = add_coords
        self.use_grid_skip = use_grid_skip
        self.use_input_skip = use_input_skip
        self.use_multiplicative_conditioning = use_multiplicative_conditioning
        self.material_adaptive_pca = material_adaptive_pca
        
        if mesh_coords is not None:
            mesh_coords = torch.from_numpy(mesh_coords.astype(np.float32))
            if mesh_coords.ndim == 2:
                coord_dim = mesh_coords.shape[1]
            else:
                coord_dim = 1
                mesh_coords = mesh_coords.unsqueeze(1)
            
            if use_grid_skip:
                mesh_coords_min = mesh_coords.min(dim=0, keepdim=True)[0]
                mesh_coords_max = mesh_coords.max(dim=0, keepdim=True)[0]
                mesh_coords_range = mesh_coords_max - mesh_coords_min
                mesh_coords_range = torch.clamp(mesh_coords_range, min=1e-8)
                mesh_coords_normalized = (mesh_coords - mesh_coords_min) / mesh_coords_range
            else:
                mesh_coords_normalized = mesh_coords
            
            self.register_buffer('mesh_coords', mesh_coords_normalized)  # (N, coord_dim), normalized to [0,1] if use_grid_skip
            self.coord_dim = coord_dim
            if num_nodes is None:
                num_nodes = mesh_coords.shape[0]
        else:
            self.mesh_coords = None
            self.coord_dim = 0
            if num_nodes is None:
                raise ValueError("num_nodes must be provided when mesh_coords is None")
        
        self.num_nodes = num_nodes
        self.material_prop_dim = material_prop_dim
        
        if material_prop_dim is not None and material_prop_dim > 0:
            self.use_material_conditioning = True
            encoding_dim = width // 4
            self.material_prop_dim = material_prop_dim
            self.encoding_dim = encoding_dim
            self.material_fc_code = nn.Linear(material_prop_dim, width)
            encoded_dim = material_prop_dim * 2 * encoding_dim
            self.material_fc_encoding = nn.Sequential(
                nn.Linear(encoded_dim, encoding_dim * 4),
                nn.GELU(),
                nn.Linear(encoding_dim * 4, width)
            )
            B_freqs = np.pi * np.power(2.0, np.arange(0, encoding_dim, dtype=np.float32))
            self.register_buffer('material_B', torch.from_numpy(B_freqs).reshape(1, 1, -1))
            self.material_lift_bias = nn.Parameter(torch.zeros(width, 1))
            self.material_lift_scale = nn.Parameter(torch.ones(width, 1))
            material_channels = 0
        else:
            self.use_material_conditioning = False
            self.material_fc_code = None
            self.material_fc_encoding = None
            self.material_B = None
            self.material_lift_bias = None
            self.material_lift_scale = None
            material_channels = 0

        lift_in = in_channels + (self.coord_dim if add_coords else 0) + material_channels
        self.fc0 = nn.Conv1d(lift_in, width, kernel_size=1)
        
        self.blocks = nn.ModuleList([
            MRBNOBlock1d(width, n_modes, pca_components, pca_mean, 
                        norm=norm, dropout_p=dropout_p, residual=residual,
                        material_adaptive=material_adaptive_pca,
                        material_prop_dim=material_prop_dim if material_adaptive_pca else None,
                        width_for_pca=width if material_adaptive_pca else None)
            for _ in range(n_layers)
        ])
        
        if use_grid_skip:
            self.b_layers = nn.ModuleList([
                nn.Conv1d(self.coord_dim, width, kernel_size=1)
                for _ in range(n_layers + 1)
            ])
        else:
            self.b_layers = None
        
        self.fc1 = nn.Conv1d(width, width, kernel_size=1)
        self.fc2 = nn.Conv1d(width, out_channels, kernel_size=1)
        
        if use_input_skip:
            self.input_skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.input_skip = None
        
        self.act = nn.GELU()

    def _concat_coords(self, x):
        """
        Coordinate concatenation using actual mesh coordinates.
        x: (B, C, N)
        returns: (B, C+coord_dim, N)
        """
        B, _, N = x.shape
        coords = self.mesh_coords.T.unsqueeze(0).expand(B, -1, -1)  # (B, coord_dim, N)
        return torch.cat([x, coords], dim=1)
    
    def _get_grid_features(self, batch_size, device):
        """
        Get grid-based features from mesh coordinates (similar to Geo-FNO b layers).
        Returns: (B, coord_dim, N) mesh coordinates as features
        """
        if self.mesh_coords is None:
            return None
        coords = self.mesh_coords.T.unsqueeze(0).expand(batch_size, -1, -1)  # (B, coord_dim, N)
        return coords.to(device)

    def _compute_material_conditioning(self, material_props):
        """
        Compute material property conditioning features.
        Uses multi-resolution positional encoding similar to IPHI.
        
        Args:
            material_props: (B, material_prop_dim) - material properties
            
        Returns:
            material_combined: (B, width, 1) - conditioning features
        """
        material_props = material_props.float()
        material_code = self.material_fc_code(material_props)  # (B, width)
        
        material_sin = torch.sin(self.material_B * material_props.unsqueeze(-1))  # (B, material_prop_dim, encoding_dim)
        material_cos = torch.cos(self.material_B * material_props.unsqueeze(-1))  # (B, material_prop_dim, encoding_dim)
        material_encoded = torch.cat([material_sin, material_cos], dim=-1)  # (B, material_prop_dim, 2*encoding_dim)
        material_encoded = material_encoded.flatten(1)  # (B, material_prop_dim * 2 * encoding_dim)
        material_features = self.material_fc_encoding(material_encoded)  # (B, width)
        
        material_combined = (material_code + material_features).unsqueeze(-1)  # (B, width, 1)
        return material_combined

    def forward(self, x, material_props=None):
        """
        Forward pass.
        x: (B, C_in, N) - point cloud data
        material_props: (B, material_prop_dim) - optional material properties
        returns: (B, C_out, N)
        """
        x_input = x
        
        if self.add_coords:
            x = self._concat_coords(x)
        
        x = self.fc0(x)
        
        material_conditioning = None
        if material_props is not None and self.use_material_conditioning:
            material_conditioning = self._compute_material_conditioning(material_props)  # (B, width, 1)
            material_bias = self.material_lift_bias.unsqueeze(0)  # (1, width, 1)
            material_scale = self.material_lift_scale.unsqueeze(0)  # (1, width, 1)
            x = x * (1.0 + material_scale * material_conditioning) + material_bias * material_conditioning
        
        if self.use_grid_skip:
            grid_feat = self._get_grid_features(x.shape[0], x.device)
            x = x + self.b_layers[0](grid_feat)
            x = self.act(x)
        
        for i, blk in enumerate(self.blocks):
            if material_conditioning is not None:
                if self.use_multiplicative_conditioning:
                    x = x * (1.0 + 0.5 * material_conditioning)
                else:
                    x = x + material_conditioning * 0.5
            x = blk(x, material_props=material_props if self.material_adaptive_pca else None)
            if self.use_grid_skip:
                grid_feat = self._get_grid_features(x.shape[0], x.device)
                x = x + self.b_layers[i + 1](grid_feat)
        
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        
        if self.use_input_skip:
            x_skip = self.input_skip(x_input)
            x = x + x_skip
        
        return x
