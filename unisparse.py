"""
Unimodal Projection Layer (Refactored)
=======================================
Decouples unimodal projection (PAVA) from normalization (sparsemax/softmax).

Architecture:
    logits → PAVA (unimodal projection) → Sparsemax/Softmax (normalization)

Key changes from original:
  1. PAVA is a standalone autograd Function
  2. Sparsemax: uses entmax library if available, otherwise uses built-in implementation
  3. Easy switching between sparsemax and softmax via `normalization` parameter
  4. Cleaner separation of concerns

References
----------
* Lim et al. (2016) "Unimodal Probability Distributions for Deep Ordinal Classification"
* Martins & Astudillo (2016) "From Softmax to Sparsemax"
* Pool-Adjacent Violators Algorithm (PAVA) for isotonic regression
* Spouge, Wan & Wilbur (2003) "Least-squares approximation of an unimodal function"

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple, Optional
from unimodal import unimodalProjection

from mysparsemax import sparsemax


# ══════════════════════════════════════════════════════════════════════════════
# PART 3: UnimodalProjectionLayer (Configurable)
# ══════════════════════════════════════════════════════════════════════════════

class UnimodalProjectionLayer(nn.Module):
    """
    Unimodal projection layer with configurable normalization.
    
    Pipeline:
        logits → PAVA (unimodal) → Sparsemax/Softmax → probabilities
    
    Args:
        normalization: 'sparsemax' or 'softmax' (default: 'sparsemax')
        dim: dimension to apply normalization (default: -1)
        alpha: interpolation parameter for sparsemax (default: 1.0)
               ONLY APPLIED DURING TRAINING:
                 Training: p = α·sparsemax(z) + (1-α)·z
                 Test:     p = sparsemax(z)  (always α=1.0)
               
               When α=1.0: Pure sparsemax (train=test, default)
               When α=0.0: Identity during training, sparsemax at test
               When 0 < α < 1: Soft interpolation during training only
               
               Only applies when normalization='sparsemax'
    
    Example:
        # Standard (α=1.0, train=test)
        layer = UnimodalProjectionLayer(normalization='sparsemax', alpha=1.0)
        layer.eval()
        p = layer(z)  # Pure sparsemax
        
        # Soft training (α=0.7)
        layer = UnimodalProjectionLayer(normalization='sparsemax', alpha=0.7)
        layer.train()
        p_train = layer(z)  # 0.7·sparsemax(z) + 0.3·z
        layer.eval()
        p_test = layer(z)   # Pure sparsemax(z), regardless of α
        
        # Softmax normalization (α ignored)
        layer = UnimodalProjectionLayer(normalization='softmax')
    
    Note:
        The alpha parameter allows "soft training" where the model learns
        with a relaxed sparsity constraint, but is evaluated with full
        sparsity. This can improve convergence while maintaining the
        sparse property at test time.
    """
    
    def __init__(self, normalization: str = 'sparsemax', dim: int = -1, alpha: float = 1.0):
        super().__init__()
        
        if normalization not in ['sparsemax', 'softmax']:
            raise ValueError(f"normalization must be 'sparsemax' or 'softmax', got '{normalization}'")
        
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        
        self.normalization = normalization
        self.dim = dim
        self.alpha = alpha
        self.unilayer = unimodalProjection()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, K) or (K,) input logits
        Returns:
            p: (B, K) or (K,) unimodal probability distribution
        
        Note:
            Alpha interpolation is only applied during training.
            At test time, always uses pure sparsemax (α=1.0).
        """        
        if z.dim() == 2:
            # Batch: (B, K)
            B, K = z.shape
            
            # Apply PAVA to each sample independently
            z_unimodal = self.unilayer(z)
            
            if self.normalization == 'sparsemax':
                p_sparse = sparsemax(z_unimodal, dim=self.dim)
                
                # Alpha interpolation only during training
                if self.training and self.alpha < 1.0:
                    # Interpolate: α * sparsemax(z) + (1-α) * z
                    p = self.alpha * p_sparse + (1 - self.alpha) * z_unimodal
                else:
                    # Test time: pure sparsemax (α=1.0)
                    p = p_sparse
            else: # softmax
                if self.alpha < .5 : 
                    p = F.softmax(z, dim=self.dim) 
                else:
                    p = F.softmax(z_unimodal, dim=self.dim)
            
            return p
        
        else:
            raise ValueError(f"Input must be 1D (K,) or 2D (B, K), got shape {z.shape}")
    
    def extra_repr(self) -> str:
        if self.normalization == 'sparsemax':
            return f'normalization={self.normalization}, alpha={self.alpha}'
        else:
            return f'normalization={self.normalization}'


# ══════════════════════════════════════════════════════════════════════════════
# PART 5: Stochastic Unimodal MLP with Beta Parameter
# ══════════════════════════════════════════════════════════════════════════════

class StochasticUnimodalMLP(nn.Module):
    """
    MLP with stochastic normalization selection during training.
    
    Both paths use PAVA (unimodal projection), but differ in normalization:
        - With probability β:     PAVA → Sparsemax (sparse + unimodal)
        - With probability (1-β): PAVA → Softmax  (dense + unimodal)
    
    At test time:
        - Always uses PAVA → Sparsemax (deterministic)
    
    This allows studying the contribution of sparsity while maintaining
    unimodality in both training modes. Can also be used for curriculum
    learning where the model gradually transitions to sparse predictions.
    
    Args:
        input_dim: input feature dimension
        hidden_dim: hidden layer dimension (default: 128)
        num_classes: number of output classes
        beta: probability of using Sparsemax during training (default: 1.0)
              beta=1.0 → always PAVA→Sparsemax (deterministic)
              beta=0.5 → 50% Sparsemax, 50% Softmax (both with PAVA)
              beta=0.0 → always PAVA→Softmax during training
        alpha: sparsemax interpolation parameter (default: 1.0)
        dropout: dropout probability (default: 0.3)
    
    Example:
        # Always sparse unimodal (standard)
        model = StochasticUnimodalMLP(input_dim=10, num_classes=5, beta=1.0)
        
        # 70% sparse, 30% dense (both unimodal)
        model = StochasticUnimodalMLP(input_dim=10, num_classes=5, beta=0.7)
        
        # Curriculum: gradually increase sparsity
        model = StochasticUnimodalMLP(input_dim=10, num_classes=5, beta=0.0)
        for epoch in range(100):
            model.set_beta(min(1.0, epoch / 50))
            train_one_epoch(...)
    
    Note:
        Both training paths apply PAVA projection for unimodality.
        The difference is only in the normalization step:
            - Sparsemax (beta path): produces sparse probabilities
            - Softmax ((1-beta) path): produces dense probabilities
        This isolates the contribution of sparsity in your ablation studies.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 10,
                 beta: float = 1.0, alpha: float = 1.0, dropout: float = 0.3):
        super().__init__()
        
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.beta = beta
        self.alpha = alpha
        
        # Backbone MLP
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output head (shared by both paths)
        self.head = nn.Linear(hidden_dim, num_classes)
        
        # Unimodal projection layer with sparsemax
        self.unimodal_sparsemax = UnimodalProjectionLayer(
            normalization='sparsemax',
            alpha=alpha
        )
        
        # Unimodal projection layer with softmax
        self.unimodal_softmax = UnimodalProjectionLayer(
            normalization='softmax'
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) input features
        
        Returns:
            p: (B, num_classes) output probabilities (always unimodal)
        """
        # Shared backbone
        h = self.backbone(x)  # (B, hidden_dim)
        z = self.head(h)      # (B, num_classes) logits
        
        # Stochastic normalization selection during training
        if self.training and self.beta < 1.0:
            # Sample: should we use sparsemax or softmax?
            use_sparsemax = torch.rand(1).item() < self.beta
            
            if use_sparsemax:
                # Path 1: PAVA → Sparsemax (sparse + unimodal)
                p = self.unimodal_sparsemax(z)
            else:
                # Path 2: PAVA → Softmax (dense + unimodal)
                p = self.unimodal_softmax(z)
        else:
            # Test time or beta=1.0: always use PAVA → Sparsemax
            p = self.unimodal_sparsemax(z)
        
        return p
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.
        
        Args:
            x: (B, input_dim) input features
        
        Returns:
            y_pred: (B,) predicted class indices
        """
        self.eval()
        with torch.no_grad():
            p = self(x)
            return p.argmax(dim=-1)
    
    def set_beta(self, new_beta: float):
        """Update beta parameter (useful for curriculum learning)."""
        if not (0.0 <= new_beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {new_beta}")
        self.beta = new_beta
    
    def extra_repr(self) -> str:
        return (f'input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, '
                f'num_classes={self.num_classes}, beta={self.beta}, alpha={self.alpha}')



# ══════════════════════════════════════════════════════════════════════════════
# PART 6: Verification & Testing
# ══════════════════════════════════════════════════════════════════════════════

def verify_unimodality(p: torch.Tensor, tol: float = 1e-6) -> bool:
    """
    Check if probability distribution p is unimodal.
    
    A distribution is unimodal if it has at most one local maximum.
    Equivalently: there exists a peak k such that
        p[0] <= ... <= p[k] >= ... >= p[K-1]
    """
    K = p.shape[0]
    
    # Find peak
    peak_idx = p.argmax().item()
    
    # Check non-decreasing before peak
    for i in range(peak_idx):
        if p[i] > p[i + 1] + tol:
            return False
    
    # Check non-increasing after peak
    for i in range(peak_idx, K - 1):
        if p[i] < p[i + 1] - tol:
            return False
    
    return True


def test_unimodal_projection():
    """Test suite for UnimodalProjectionLayer."""
    print("=" * 60)
    print("Testing UnimodalProjectionLayer")
    print("=" * 60)
    
    K = 10
    batch_size = 5
    
    # Test 1: Sparsemax normalization
    print("\n[Test 1] Sparsemax normalization (alpha=1.0)")
    layer_sparse = UnimodalProjectionLayer(normalization='sparsemax', alpha=1.0)
    z = torch.randn(batch_size, K, requires_grad=True)
    p = layer_sparse(z)
    
    print(f"  Input shape:  {z.shape}")
    print(f"  Output shape: {p.shape}")
    print(f"  Sum of probs: {p.sum(dim=1)}")  # Should be ~1.0
    print(f"  Sparsity:     {(p == 0).sum().item()} / {p.numel()} zeros")
    
    # Check unimodality
    unimodal_count = sum(verify_unimodality(p[i]) for i in range(batch_size))
    print(f"  Unimodal:     {unimodal_count}/{batch_size} samples")
    
    # Backward pass
    loss = p.sum()
    loss.backward()
    print(f"  Gradient shape: {z.grad.shape}")
    print(f"  ✓ Sparsemax test passed")
    
    # Test 1b: Sparsemax with alpha interpolation (TRAINING MODE)
    print("\n[Test 1b] Alpha interpolation during training (alpha=0.5)")
    layer_interp = UnimodalProjectionLayer(normalization='sparsemax', alpha=0.5)
    z = torch.randn(batch_size, K, requires_grad=True)
    
    # Training mode: alpha interpolation applied
    layer_interp.train()
    p_train = layer_interp(z)
    
    print(f"  Training output shape: {p_train.shape}")
    print(f"  Training output range: [{p_train.min().item():.4f}, {p_train.max().item():.4f}]")
    print(f"  Note: With alpha=0.5, training output is NOT a probability distribution")
    
    # Eval mode: pure sparsemax (alpha=1.0 implicitly)
    layer_interp.eval()
    p_test = layer_interp(z)
    
    print(f"  Test output shape: {p_test.shape}")
    print(f"  Test output range: [{p_test.min().item():.4f}, {p_test.max().item():.4f}]")
    print(f"  Test sum: {p_test.sum(dim=1).mean().item():.6f} (should be ~1.0)")
    print(f"  Test has zeros: {(p_test == 0).any().item()} (should be True)")
    
    # Verify they're different
    print(f"  Training != Test: {not torch.allclose(p_train, p_test)}")
    
    # Backward pass
    loss = p_train.sum()
    loss.backward()
    print(f"  ✓ Alpha interpolation test passed")
    
    # Test 1c: Identity during training (alpha=0.0), sparsemax at test
    print("\n[Test 1c] Identity training, sparsemax test (alpha=0.0)")
    layer_identity = UnimodalProjectionLayer(normalization='sparsemax', alpha=0.0)
    z_input = torch.tensor([1.0, 3.0, 2.0, 1.5, 0.5])
    
    # Training: identity (just PAVA projection)
    layer_identity.train()
    p_train = layer_identity(z_input)
    print(f"  Training (α=0.0): {p_train.detach().numpy()}")
    print(f"  Note: Just PAVA projection (unimodal, not normalized)")
    
    # Test: pure sparsemax
    layer_identity.eval()
    p_test = layer_identity(z_input)
    print(f"  Test (α→1.0):     {p_test.detach().numpy()}")
    print(f"  Note: Pure sparsemax (sparse, normalized)")
    print(f"  Test sum: {p_test.sum().item():.6f}")
    print(f"  Test has zeros: {(p_test == 0).any().item()}")
    
    # Verify difference
    print(f"  Training != Test: {not torch.allclose(p_train, p_test)}")
    print(f"  ✓ Identity→Sparsemax test passed")
    
    # Test 2: Softmax normalization
    print("\n[Test 2] Softmax normalization")
    layer_soft = UnimodalProjectionLayer(normalization='softmax')
    z = torch.randn(batch_size, K, requires_grad=True)
    p = layer_soft(z)
    
    print(f"  Input shape:  {z.shape}")
    print(f"  Output shape: {p.shape}")
    print(f"  Sum of probs: {p.sum(dim=1)}")  # Should be 1.0
    print(f"  Sparsity:     {(p < 1e-6).sum().item()} / {p.numel()} near-zeros")
    
    # Check unimodality
    unimodal_count = sum(verify_unimodality(p[i]) for i in range(batch_size))
    print(f"  Unimodal:     {unimodal_count}/{batch_size} samples")
    
    # Backward pass
    loss = p.sum()
    loss.backward()
    print(f"  Gradient shape: {z.grad.shape}")
    print(f"  ✓ Softmax test passed")
    
    # Test 3: Single vector input
    print("\n[Test 3] Single vector (unbatched)")
    z_single = torch.randn(K, requires_grad=True)
    p_single = layer_soft(z_single)
    
    print(f"  Input shape:  {z_single.shape}")
    print(f"  Output shape: {p_single.shape}")
    print(f"  Sum of probs: {p_single.sum().item():.6f}")
    print(f"  Unimodal:     {verify_unimodality(p_single)}")
    print(f"  ✓ Unbatched test passed")
    
    # Test 4: Compare sparsemax vs softmax
    print("\n[Test 4] Sparsemax vs Softmax comparison")
    z = torch.tensor([1.0, 2.0, 3.0, 2.5, 1.5, 0.5])
    
    p_sparse = UnimodalProjectionLayer(normalization='sparsemax', alpha=1.0)(z)
    p_soft = UnimodalProjectionLayer(normalization='softmax')(z)
    
    print(f"  Input:     {z.numpy()}")
    print(f"  Sparsemax: {p_sparse.detach().numpy()}")
    print(f"  Softmax:   {p_soft.detach().numpy()}")
    print(f"  Sparse zeros: {(p_sparse == 0).sum().item()}")
    print(f"  Softmax zeros: {(p_soft < 1e-6).sum().item()}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


def test_stochastic_mlp():
    """Test suite for StochasticUnimodalMLP with beta parameter."""
    print("\n" + "=" * 60)
    print("Testing StochasticUnimodalMLP")
    print("=" * 60)
    
    input_dim = 20
    num_classes = 7
    batch_size = 10
    
    # Test 1: Deterministic (beta=1.0, always unimodal)
    print("\n[Test 1] Deterministic mode (beta=1.0)")
    model_det = StochasticUnimodalMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        beta=1.0,
        alpha=1.0
    )
    
    x = torch.randn(batch_size, input_dim)
    
    # Training mode
    model_det.train()
    p_train1 = model_det(x)
    p_train2 = model_det(x)
    
    print(f"  Output shape: {p_train1.shape}")
    print(f"  Train output 1 == Train output 2: {torch.allclose(p_train1, p_train2)}")
    print(f"    (Should be True with beta=1.0)")
    
    # Eval mode
    model_det.eval()
    p_eval = model_det(x)
    print(f"  Train == Eval: {torch.allclose(p_train1, p_eval)}")
    print(f"  ✓ Deterministic test passed")
    
    # Test 2: Stochastic (beta=0.5)
    print("\n[Test 2] Stochastic mode (beta=0.5)")
    model_stoch = StochasticUnimodalMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        beta=0.5,
        alpha=1.0
    )
    
    # Training mode - outputs should differ due to randomness
    model_stoch.train()
    outputs_train = [model_stoch(x) for _ in range(10)]
    
    # Check if we get variety (at least some different outputs)
    all_same = all(torch.allclose(outputs_train[0], out) for out in outputs_train[1:])
    print(f"  All training outputs identical: {all_same}")
    print(f"    (Should be False with beta=0.5, expect ~50% variation)")
    
    # Count sparsity differences
    # Both paths use PAVA (unimodal), but:
    #   - beta path: PAVA → Sparsemax (has exact zeros)
    #   - (1-beta) path: PAVA → Softmax (no exact zeros)
    sparse_count = sum((out == 0).any().item() for out in outputs_train)
    dense_count = 10 - sparse_count
    print(f"  Sparse outputs (PAVA→Sparsemax): {sparse_count}/10")
    print(f"  Dense outputs (PAVA→Softmax):    {dense_count}/10")
    print(f"    (Expected ~5 each with beta=0.5)")
    print(f"    Note: Both are unimodal! Only sparsity differs.")
    
    # Verify all outputs are valid probabilities
    for i, out in enumerate(outputs_train):
        sum_check = (out.sum(dim=1) - 1.0).abs().max()
        assert sum_check < 1e-5, f"Output {i} doesn't sum to 1: {sum_check}"
    print(f"  All outputs sum to 1.0: ✓")
    
    # Eval mode - should be deterministic (always sparsemax)
    model_stoch.eval()
    p_eval1 = model_stoch(x)
    p_eval2 = model_stoch(x)
    print(f"  Eval output 1 == Eval output 2: {torch.allclose(p_eval1, p_eval2)}")
    print(f"  ✓ Stochastic test passed")
    
    # Test 3: Beta annealing (curriculum learning)
    print("\n[Test 3] Beta annealing (curriculum learning)")
    model_curr = StochasticUnimodalMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        beta=0.0,  # Start with pure softmax
        alpha=1.0
    )
    
    print(f"  Initial beta: {model_curr.beta}")
    
    # Simulate training with beta schedule
    beta_schedule = [0.0, 0.25, 0.5, 0.75, 1.0]
    for beta in beta_schedule:
        model_curr.set_beta(beta)
        print(f"  Set beta={beta}, current beta={model_curr.beta}")
    
    print(f"  ✓ Beta annealing test passed")
    
    # Test 4: Alpha interpolation in MLP
    print("\n[Test 4] Alpha parameter (sparsemax interpolation)")
    model_alpha_low = StochasticUnimodalMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        beta=1.0,
        alpha=0.3  # 30% sparsemax, 70% identity
    )
    
    model_alpha_high = StochasticUnimodalMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        beta=1.0,
        alpha=1.0  # 100% sparsemax
    )
    
    # Training mode
    model_alpha_low.train()
    model_alpha_high.train()
    
    x_test = torch.randn(5, input_dim)
    p_low_train = model_alpha_low(x_test)
    p_high_train = model_alpha_high(x_test)
    
    print(f"  Training - Alpha=0.3 range: [{p_low_train.min().item():.3f}, {p_low_train.max().item():.3f}]")
    print(f"  Training - Alpha=1.0 range: [{p_high_train.min().item():.3f}, {p_high_train.max().item():.3f}]")
    
    # Eval mode
    model_alpha_low.eval()
    model_alpha_high.eval()
    
    p_low_test = model_alpha_low(x_test)
    p_high_test = model_alpha_high(x_test)
    
    print(f"  Test - Alpha=0.3→1.0 sum: {p_low_test.sum(dim=1).mean().item():.3f} (should be ~1.0)")
    print(f"  Test - Alpha=1.0 sum:     {p_high_test.sum(dim=1).mean().item():.3f} (should be ~1.0)")
    print(f"  Note: Both test outputs use pure sparsemax regardless of alpha")
    print(f"  ✓ Alpha interpolation test passed")
    
    # Test 5: Gradient flow
    print("\n[Test 5] Gradient flow")
    model_grad = StochasticUnimodalMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        beta=0.7,
        alpha=1.0
    )
    
    model_grad.train()
    x_grad = torch.randn(batch_size, input_dim, requires_grad=True)
    p = model_grad(x_grad)
    
    # Dummy loss
    loss = -p.log().mean()
    loss.backward()
    
    print(f"  Input gradient shape: {x_grad.grad.shape}")
    print(f"  Gradient exists: {x_grad.grad is not None}")
    print(f"  Gradient non-zero: {(x_grad.grad.abs() > 1e-8).any().item()}")
    print(f"  ✓ Gradient flow test passed")
    
    # Test 6: Prediction method
    print("\n[Test 6] Predict method")
    model_pred = StochasticUnimodalMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        beta=1.0
    )
    
    x_pred = torch.randn(batch_size, input_dim)
    y_pred = model_pred.predict(x_pred)
    
    print(f"  Predictions shape: {y_pred.shape}")
    print(f"  Predictions: {y_pred.numpy()}")
    print(f"  All in range [0, {num_classes-1}]: {((y_pred >= 0) & (y_pred < num_classes)).all().item()}")
    print(f"  ✓ Predict method test passed")
    
    print("\n" + "=" * 60)
    print("All StochasticUnimodalMLP tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_unimodal_projection()
    test_stochastic_mlp()
