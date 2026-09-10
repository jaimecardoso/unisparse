"""
ORD-ACL and VS-SL Implementation
=================================
Based on: Yamasaki (2022) "Unimodal Likelihood Models for Ordinal Data", TMLR

Both methods enforce unimodality via:
1. ORD transformation: ensures logits are non-decreasing
2. Different link functions: ORD-ACL uses adjacent-category logits, VS-SL uses V-shaped softmax

Reference: https://openreview.net/forum?id=1l0sClLiPc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# ORD TRANSFORMATION (shared by both methods)
# ══════════════════════════════════════════════════════════════════════════════

def ord_transform(z: torch.Tensor, rho_fn: str = 'exp') -> torch.Tensor:
    """
    ORD transformation: converts K unconstrained logits into K non-decreasing values.
    
    Formula: z'_k = z_1 + Σ_{ℓ=2}^{k} ρ(z_ℓ)
    
    where ρ: ℝ → [0, ∞) ensures monotonicity.
    
    Args:
        z: (B, K) unconstrained logits
        rho_fn: 'exp' (ρ(u)=exp(u)) or 'square' (ρ(u)=u²)
    
    Returns:
        z': (B, K) non-decreasing logits (z'_1 ≤ z'_2 ≤ ... ≤ z'_K)
    """
    B, K = z.shape
    
    # Apply ρ transformation
    if rho_fn == 'exp':
        rho_z = torch.exp(z)
    elif rho_fn == 'square':
        rho_z = z ** 2
    else:
        raise ValueError(f"Unknown rho_fn: {rho_fn}")
    
    # z'_1 = z_1 (unchanged)
    # z'_k = z_1 + Σ_{ℓ=2}^{k} ρ(z_ℓ) for k ≥ 2
    z_prime = torch.zeros_like(z)
    z_prime[:, 0] = z[:, 0]
    
    for k in range(1, K):
        z_prime[:, k] = z_prime[:, k-1] + rho_z[:, k]
    
    return z_prime


# ══════════════════════════════════════════════════════════════════════════════
# ORD-ACL (Adjacent-Category Logit)
# ══════════════════════════════════════════════════════════════════════════════

class OrdACLModel(nn.Module):
    """
    ORD-ACL: Ordinal Adjacent-Category Logits
    
    Architecture:
        Backbone → K-1 logits → ORD transform → adjacent-category link → unimodal probabilities
    
    The adjacent-category link models:
        P(Y=k | Y∈{k,k+1}) = σ(z'_k)
    
    which implicitly defines the full probability distribution via recursive relations.
    
    This guarantees unimodality by construction.
    """
    
    def __init__(self, backbone: nn.Module, K: int, hidden: int, rho_fn: str = 'exp'):
        super().__init__()
        self.backbone = backbone
        self.K = K
        self.rho_fn = rho_fn
        # Output K-1 logits (adjacent ratios)
        self.head = nn.Linear(hidden, K - 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, d) input
        Returns:
            p: (B, K) unimodal probability distribution
        """
        h = self.backbone(x)           # (B, 128)
        z = self.head(h)               # (B, K-1) unconstrained
        z_ord = ord_transform(z, self.rho_fn)   # (B, K-1) non-decreasing
        
        # Adjacent-category probabilities: P(k | k or k+1) = sigmoid(z'_k)
        # Recursively compute full distribution
        p = self._acl_to_prob(z_ord)
        
        return p
    
    def _acl_to_prob(self, z_ord: torch.Tensor) -> torch.Tensor:
        """
        Convert adjacent-category logits to full probability distribution.
        
        Let a_k = P(Y=k | Y∈{k,k+1}) = σ(z'_k)
        Then: P(Y=k) = P(Y≥k) · a_k
              P(Y≥k+1) = P(Y≥k) · (1 - a_k)
        
        Starting from P(Y≥1) = 1, we recursively compute P(Y=k).
        """
        B, K_minus_1 = z_ord.shape
        K = K_minus_1 + 1
        
        a = torch.sigmoid(z_ord)   # (B, K-1) adjacent probs
        
        p = torch.zeros(B, K, device=z_ord.device)
        cumprob = torch.ones(B, device=z_ord.device)   # P(Y ≥ 1) = 1
        
        for k in range(K - 1):
            p[:, k] = cumprob * a[:, k]
            cumprob = cumprob * (1 - a[:, k])
        
        p[:, K-1] = cumprob   # remaining probability
        
        return p
    
    def predict(self, x):
        p =self(x) 
        return p, torch.argmax(p, dim = 1)        
        #return self(x).argmax(1)
    
    def loss(self, x, y):
        probs = self(x)
        return F.nll_loss(torch.log(probs + 1e-12), y)


# ══════════════════════════════════════════════════════════════════════════════
# VS-SL (V-Shaped Softmax Layer)
# ══════════════════════════════════════════════════════════════════════════════

class VSSLModel(nn.Module):
    """
    VS-SL: V-Shaped Softmax Layer
    
    Architecture:
        Backbone → K logits → ORD transform → V-shape transform → softmax → unimodal probs
    
    The V-shape transform τ: ℝ → ℝ₊ is symmetric, e.g.:
        τ(u) = |u| or τ(u) = u²
    
    Applied as: z''_k = -τ(z'_k) where z' is non-decreasing (from ORD).
    This creates a ∩-shaped curve that becomes unimodal after softmax.
    
    The negative inverted-V ensures the maximum logit corresponds to the peak.
    """
    
    def __init__(self, backbone: nn.Module, K: int, hidden: int, 
                 rho_fn: str = 'exp', tau_fn: str = 'abs'):
        super().__init__()
        self.backbone = backbone
        self.K = K
        self.rho_fn = rho_fn
        self.tau_fn = tau_fn
        # Output K logits
        self.head = nn.Linear(hidden, K)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, d) input
        Returns:
            p: (B, K) unimodal probability distribution
        """
        h = self.backbone(x)           # (B, 128)
        z = self.head(h)               # (B, K) unconstrained
        z_ord = ord_transform(z, self.rho_fn)   # (B, K) non-decreasing
        
        # V-shape transform: z''_k = -τ(z'_k)
        # This creates an inverted-V (∩-shape) because z' is increasing
        if self.tau_fn == 'abs':
            z_v = -torch.abs(z_ord)
        elif self.tau_fn == 'square':
            z_v = -(z_ord ** 2)
        else:
            raise ValueError(f"Unknown tau_fn: {self.tau_fn}")
        
        # Softmax produces unimodal distribution
        p = F.softmax(z_v, dim=-1)
        
        return p
    
    def predict(self, x):
        p =self(x) 
        return p, torch.argmax(p, dim = 1)         
        #return self(x).argmax(1)
    
    def loss(self, x, y):
        probs = self(x)
        return F.nll_loss(torch.log(probs + 1e-12), y)


# ══════════════════════════════════════════════════════════════════════════════
# VARIANTS: PO and HO
# ══════════════════════════════════════════════════════════════════════════════

class OrdACL_PO(nn.Module):
    """
    ORD-ACL-PO: Proportional Odds variant
    
    Uses Proportional Odds Model (POM) cumulative link instead of adjacent-category.
    Still applies ORD transformation to ensure monotonicity.
    """
    
    def __init__(self, backbone: nn.Module, K: int, rho_fn: str = 'exp'):
        super().__init__()
        self.backbone = backbone
        self.K = K
        self.rho_fn = rho_fn
        # Shared weight + K-1 thresholds (POM structure)
        self.shared_w = nn.Linear(128, 1, bias=False)
        self.thresholds = nn.Parameter(torch.randn(K - 1))
        nn.init.xavier_uniform_(self.shared_w.weight)
    
    def forward(self, x):
        h = self.backbone(x)           # (B, 128)
        eta = self.shared_w(h)         # (B, 1) shared predictor
        
        # Sort thresholds via ORD (ensures θ_1 ≤ θ_2 ≤ ... ≤ θ_{K-1})
        theta = ord_transform(self.thresholds.unsqueeze(0), self.rho_fn).squeeze(0)
        
        # Cumulative probabilities: P(Y ≤ k) = σ(θ_k - η)
        cum_probs = torch.sigmoid(theta - eta)  # (B, K-1)
        
        # Convert to class probabilities
        p = torch.zeros(x.shape[0], self.K, device=x.device)
        p[:, 0] = cum_probs[:, 0]
        for k in range(1, self.K - 1):
            p[:, k] = cum_probs[:, k] - cum_probs[:, k-1]
        p[:, self.K-1] = 1 - cum_probs[:, -1]
        
        return p
    
    def predict(self, x):
        return self(x).argmax(1)
    
    def loss(self, x, y):
        probs = self(x)
        return F.nll_loss(torch.log(probs + 1e-12), y)


class VSSL_HO(nn.Module):
    """
    VS-SL-HO: Heteroscedastic Odds variant
    
    Learns a sample-dependent scale parameter s(x) to control the spread
    of the unimodal distribution, allowing heteroscedasticity.
    """
    
    def __init__(self, backbone: nn.Module, K: int, 
                 rho_fn: str = 'exp', tau_fn: str = 'abs'):
        super().__init__()
        self.backbone = backbone
        self.K = K
        self.rho_fn = rho_fn
        self.tau_fn = tau_fn
        # Output K logits + 1 scale parameter
        self.head = nn.Linear(128, K)
        self.scale_head = nn.Linear(128, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)
    
    def forward(self, x):
        h = self.backbone(x)           # (B, 128)
        z = self.head(h)               # (B, K) unconstrained
        s = torch.exp(self.scale_head(h))  # (B, 1) positive scale
        
        z_ord = ord_transform(z, self.rho_fn)   # (B, K) non-decreasing
        
        # V-shape with scale
        if self.tau_fn == 'abs':
            z_v = -torch.abs(z_ord) / s
        elif self.tau_fn == 'square':
            z_v = -(z_ord ** 2) / s
        else:
            raise ValueError(f"Unknown tau_fn: {self.tau_fn}")
        
        p = F.softmax(z_v, dim=-1)
        return p
    
    def predict(self, x):
        return self(x).argmax(1)
    
    def loss(self, x, y):
        probs = self(x)
        return F.nll_loss(torch.log(probs + 1e-12), y)
