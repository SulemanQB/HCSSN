"""Point metrics (MSE/MAE/RMSE/MAPE) plus gate / scale-separation diagnostics."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Standard point-forecast metrics
# ═══════════════════════════════════════════════════════════════════════════

def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target).sqrt()


def mape(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-2
) -> torch.Tensor:
    """Masked MAPE: ignores targets with |target| < eps to avoid blow-up."""
    mask = target.abs() > eps
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return ((pred[mask] - target[mask]).abs() / target[mask].abs()).mean() * 100.0


def compute_metrics(
    pred: torch.Tensor, target: torch.Tensor
) -> Dict[str, float]:
    """Return dict of standard metrics (float values)."""
    return {
        "mse": mse(pred, target).item(),
        "mae": mae(pred, target).item(),
        "rmse": rmse(pred, target).item(),
        "mape": mape(pred, target).item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Horizon-specific metrics  (short / medium / long)
# ═══════════════════════════════════════════════════════════════════════════

def horizon_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    short_end: int = 24,
    mid_end: int = 72,
) -> Dict[str, float]:
    """MSE broken down by horizon bands.

    pred, target: (B, H, D)
    short  : [0, short_end)
    medium : [short_end, mid_end)
    long   : [mid_end, H)
    """
    H = pred.size(1)
    out: Dict[str, float] = {}

    slices = {
        "mse_short": (0, min(short_end, H)),
        "mse_medium": (min(short_end, H), min(mid_end, H)),
        "mse_long": (min(mid_end, H), H),
    }
    for name, (a, b) in slices.items():
        if b > a:
            out[name] = F.mse_loss(pred[:, a:b, :], target[:, a:b, :]).item()
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Scale Separation Quality  (SSQ)
# ═══════════════════════════════════════════════════════════════════════════

def scale_separation_quality(
    s_slow: torch.Tensor,
    s_medium: torch.Tensor,
    s_fast: torch.Tensor,
) -> Dict[str, float]:
    """Spectral centroid analysis per scale.

    Each scale's output is FFT'd; we report the normalised centroid
    frequency.  Well-separated scales should have:
        centroid_slow  <  centroid_medium  <  centroid_fast

    Also reports a scalar SSQ = (c_fast − c_slow) / c_fast  ∈ (0, 1].
    """
    def _centroid(s: torch.Tensor) -> float:
        # s: (B, T, H) — take batch/channel average power spectrum
        S = torch.fft.rfft(s, dim=1)
        power = (S.real ** 2 + S.imag ** 2).mean(dim=(0, 2))  # (n_freq,)
        freqs = torch.arange(len(power), device=s.device, dtype=torch.float32)
        total = power.sum() + 1e-8
        return (freqs * power).sum().item() / total.item()

    c_s = _centroid(s_slow)
    c_m = _centroid(s_medium)
    c_f = _centroid(s_fast)
    ssq = (c_f - c_s) / (c_f + 1e-8) if c_f > 1e-8 else 0.0

    return {
        "centroid_slow": c_s,
        "centroid_medium": c_m,
        "centroid_fast": c_f,
        "ssq": ssq,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Causality Preservation Score  (CPS) — lightweight proxy
# ═══════════════════════════════════════════════════════════════════════════

def causality_preservation_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_lag: int = 5,
) -> float:
    """Lightweight proxy for Granger-like causal consistency.

    For each variable pair (i, j) we check whether lagged correlation
    structure of *pred* matches *target*.  Returns a score in [0, 1]
    (1 = perfect causal structure match).

    Uses first 2 variables if D > 2 for efficiency.
    """
    # flatten batch:  (B, H, D) → (B*H, D)
    B, H, D = pred.shape
    if D < 2:
        return 1.0  # single variable — trivially preserved

    # use at most 2 variable pairs for speed
    idx = min(D, 2)

    def _lag_corr(series: torch.Tensor, lag: int) -> torch.Tensor:
        """Cross-correlation between vars 0 and 1 at given lag."""
        x = series[:, 0]
        y = series[:, 1]
        if lag > 0:
            x, y = x[:-lag], y[lag:]
        elif lag < 0:
            x, y = x[-lag:], y[:lag]
        x = x - x.mean()
        y = y - y.mean()
        denom = x.norm() * y.norm() + 1e-8
        return (x * y).sum() / denom

    # compute lag profiles for pred and target (flatten over batch)
    pred_flat = pred.reshape(-1, D)[:, :idx]
    targ_flat = target.reshape(-1, D)[:, :idx]

    lags = list(range(-max_lag, max_lag + 1))
    pred_profile = torch.stack([_lag_corr(pred_flat, l) for l in lags])
    targ_profile = torch.stack([_lag_corr(targ_flat, l) for l in lags])

    # cosine similarity of lag profiles
    sim = F.cosine_similarity(pred_profile.unsqueeze(0), targ_profile.unsqueeze(0))
    return max(0.0, sim.item())
