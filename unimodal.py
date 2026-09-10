import torch
import torch.nn as nn

#UNIMODAL UN-NORMALIZED PROJECTION: projects in the unimodal cone but does not normalize to probabilities.

# ==========================================================
# Safe isotonic regression (no autograd issues)
# ==========================================================

def isotonic_pava_batched(y: torch.Tensor) -> torch.Tensor:
    B, N = y.shape
    device = y.device

    # --------------------------------------------------
    # 1) Get block structure (no gradients)
    # --------------------------------------------------
    with torch.no_grad():
        means = torch.zeros((B, N), device=device, dtype=y.dtype)
        weights = torch.zeros((B, N), device=device, dtype=y.dtype)
        starts = torch.zeros((B, N), device=device, dtype=torch.long)

        top = torch.full((B,), -1, device=device, dtype=torch.long)
        b = torch.arange(B, device=device)

        for i in range(N):
            top = top + 1
            idx = top

            means[b, idx] = y[:, i]
            weights[b, idx] = 1.0
            starts[b, idx] = i

            while True:
                valid = top >= 1
                if not torch.any(valid):
                    break

                cur = top.clamp(min=0)
                prev = (top - 1).clamp(min=0)

                prev_mean = means[b, prev]
                cur_mean = means[b, cur]

                viol = valid & (prev_mean > cur_mean)
                if not torch.any(viol):
                    break

                bb = b[viol]
                pidx = prev[viol]
                cidx = cur[viol]

                w1 = weights[bb, pidx]
                w2 = weights[bb, cidx]
                v1 = means[bb, pidx]
                v2 = means[bb, cidx]

                new_w = w1 + w2
                new_m = (w1 * v1 + w2 * v2) / new_w

                means[bb, pidx] = new_m
                weights[bb, pidx] = new_w
                top[viol] -= 1

        # build block index per position
        k = torch.arange(N, device=device).view(1, N).expand(B, N)
        starts_filled = torch.where(
            k <= top.view(B, 1),
            starts,
            torch.full_like(starts, N)
        )

        pos = torch.arange(N, device=device).view(1, N).expand(B, N)

        block_idx = torch.searchsorted(
            starts_filled.contiguous(),
            pos.contiguous(),
            right=True
        ) - 1

        block_idx = block_idx.clamp(min=0)

    # --------------------------------------------------
    # 2) Recompute block means (WITH gradients!)
    # --------------------------------------------------

    # Sum per block
    block_sums = torch.zeros((B, N), device=device, dtype=y.dtype)
    block_counts = torch.zeros((B, N), device=device, dtype=y.dtype)

    ones = torch.ones_like(y)

    block_sums.scatter_add_(1, block_idx, y)
    block_counts.scatter_add_(1, block_idx, ones)

    block_means = block_sums / block_counts.clamp_min(1)

    # assign mean to each element
    x = torch.gather(block_means, 1, block_idx)

    return x


def antitonic_pava_batched(y: torch.Tensor) -> torch.Tensor:
    return -isotonic_pava_batched(-y)


# ==========================================================
# Unimodal projection (free mode, exact)
# ==========================================================

class unimodalProjection(nn.Module):
    """
    Exact unimodal projection (mode not fixed).
    Tries all peak locations and selects optimal one.
    """

    def __init__(self):
        super().__init__()
        #self.n = n

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        #if z.shape[-1] != self.n:
        #    raise ValueError(f"Expected last dim = {self.n}")
        self.n = z.shape[-1]
        shape = z.shape
        x = z.reshape(-1, self.n)
        B, N = x.shape

        best_x = None
        best_loss = None

        for m in range(N):
            # left: increasing
            left = isotonic_pava_batched(x[:, :m+1])

            # right: decreasing
            right = antitonic_pava_batched(x[:, m:])

            # combine
            candidate = torch.cat([
                left[:, :-1],
                ((left[:, -1] + right[:, 0]) / 2).unsqueeze(1),
                right[:, 1:]
            ], dim=1)

            loss = torch.sum((candidate - x) ** 2, dim=1)

            if best_loss is None:
                best_loss = loss
                best_x = candidate
            else:
                mask = loss < best_loss
                best_x = torch.where(mask.unsqueeze(1), candidate, best_x)
                best_loss = torch.where(mask, loss, best_loss)

        return best_x.reshape(shape)


#import cvxpy as cp
import numpy as np

def unimodal_projection_cvx(z_np):
    """
    Solve exact unimodal projection with CVXPY (mode free).
    z_np: (n,) numpy array
    """
    n = len(z_np)

    best_x = None
    best_obj = None

    for m in range(n):
        x = cp.Variable(n)

        objective = cp.Minimize(0.5 * cp.sum_squares(x - z_np))

        constraints = []

        # increasing part
        for i in range(m):
            constraints.append(x[i] <= x[i + 1])

        # decreasing part
        for i in range(m, n - 1):
            constraints.append(x[i] >= x[i + 1])

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False)

        if best_obj is None or problem.value < best_obj:
            best_obj = problem.value
            best_x = x.value

    return best_x

# ==========================================================
# Test
# ==========================================================

def check_unimodal(x, tol=1e-6):
    diffs = x[:, 1:] - x[:, :-1]
    increasing = diffs[:, :-1] >= -tol
    decreasing = diffs[:, 1:] <= tol
    # at most one sign change
    return bool(torch.all((increasing | decreasing)))


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    B, N = 4, 32
    layer = unimodalProjection(N).to(device)

    z = torch.randn(B, N, device=device, requires_grad=True)
    
    x = layer(z)

    print("Output shape:", x, x.shape)

    loss = (x ** 2).mean()
    loss.backward()

    print("Grad OK:", torch.isfinite(z.grad).all().item())
    print("Unimodal (approx check):", check_unimodal(x.detach()))

def test_compare():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    B, N = 400, 7  # keep small (CVXPY is slow)

    layer = unimodalProjection(N).to(device)

    z = torch.randn(B, N, device=device, requires_grad=True)

    x_torch = layer(z)
    x_torch_np = x_torch.detach().cpu().numpy()

    max_diff = 0.0
    max_obj_diff = 0.0

    print("\nComparing PyTorch vs CVXPY\n")

    for b in range(B):
        z_np = z[b].detach().cpu().numpy()

        x_cvx = unimodal_projection_cvx(z_np)

        diff = np.max(np.abs(x_cvx - x_torch_np[b]))
        max_diff = max(max_diff, diff)

        obj_torch = np.sum((x_torch_np[b] - z_np) ** 2)
        obj_cvx = np.sum((x_cvx - z_np) ** 2)

        obj_diff = abs(obj_torch - obj_cvx)
        max_obj_diff = max(max_obj_diff, obj_diff)

        print(f"Sample {b}:")
        print(f"  max |x_torch - x_cvx| = {diff:.3e}")
        print(f"  obj diff = {obj_diff:.3e}")

    print("\nSummary:")
    print(f"  MAX value diff: {max_diff:.3e}")
    print(f"  MAX objective diff: {max_obj_diff:.3e}")

    # backward check
    loss = x_torch.pow(2).mean()
    loss.backward()

    print("\nGradient check:")
    print("  Grad OK:", torch.isfinite(z.grad).all().item())

if __name__ == "__main__":
    test_compare()