"""

Installation
------------
For optimal performance, install the entmax library:
    pip install entmax

If entmax is not available, a built-in sparsemax implementation will be used automatically.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple, Optional

try:
    from entmax import sparsemax as entmax_sparsemax
    SPARSEMAX_AVAILABLE = True
    print("[INFO] Using entmax library for sparsemax (optimized)")
except ImportError:
    SPARSEMAX_AVAILABLE = False
    print("[INFO] entmax library not found. Using built-in sparsemax implementation.")
    print("[INFO] For better performance, install entmax: pip install entmax")

#SPARSEMAX_AVAILABLE = True

# ══════════════════════════════════════════════════════════════════════════════
# Fallback Sparsemax Implementation (when entmax not available)
# ══════════════════════════════════════════════════════════════════════════════

def _sparsemax_forward_1d(z: torch.Tensor) -> torch.Tensor:
    """
    Sparsemax projection onto the probability simplex.
    Operates on a 1-D vector z of length K.
    
    Reference: Martins & Astudillo (2016) "From Softmax to Sparsemax"
    """
    K = z.shape[0]
    z_sorted, _ = torch.sort(z, descending=True)
    z_cumsum = torch.cumsum(z_sorted, dim=0)
    
    # Find threshold: largest k s.t. 1 + k*z_sorted[k] > cumsum[k]
    k_arange = torch.arange(1, K + 1, device=z.device, dtype=z.dtype)
    support = 1 + k_arange * z_sorted > z_cumsum
    
    rho = torch.nonzero(support, as_tuple=False)
    if rho.numel() == 0:
        k_z = K
    else:
        k_z = rho[-1].item() + 1
    
    # Compute threshold
    tau = (z_cumsum[k_z - 1] - 1.0) / k_z
    
    # Project onto simplex
    return torch.clamp(z - tau, min=0.0)


def _sparsemax_forward_batch(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Batched sparsemax for 2D inputs.
    
    Args:
        z: (B, K) or (K, B) input tensor
        dim: dimension to apply sparsemax along
    
    Returns:
        p: same shape as z, sparsemax probabilities
    """
    # Ensure dim is -1 or 1 for 2D
    if z.dim() != 2:
        raise ValueError(f"Expected 2D input, got shape {z.shape}")
    
    if dim == -1 or dim == 1:
        # Apply along last dimension: (B, K)
        return torch.stack([_sparsemax_forward_1d(z[i]) for i in range(z.shape[0])], dim=0)
    elif dim == 0:
        # Apply along first dimension: (K, B) -> transpose, apply, transpose back
        z_t = z.t()  # (B, K)
        p_t = torch.stack([_sparsemax_forward_1d(z_t[i]) for i in range(z_t.shape[0])], dim=0)
        return p_t.t()  # (K, B)
    else:
        raise ValueError(f"Invalid dim={dim} for 2D tensor")


def sparsemax_fallback(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax function with automatic shape handling.
    
    Works as a drop-in replacement for entmax.sparsemax().
    
    Args:
        z: Input tensor (1D or 2D)
        dim: Dimension to apply sparsemax along
    
    Returns:
        p: Sparsemax probabilities, same shape as input
    """
    if z.dim() == 1:
        return _sparsemax_forward_1d(z)
    elif z.dim() == 2:
        return _sparsemax_forward_batch(z, dim=dim)
    else:
        raise ValueError(f"Sparsemax only supports 1D or 2D inputs, got shape {z.shape}")


# Select which sparsemax implementation to use
if SPARSEMAX_AVAILABLE:
    sparsemax = entmax_sparsemax
else:
    sparsemax = sparsemax_fallback

