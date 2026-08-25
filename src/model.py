"""HCSSN: multi-scale S4D forecasting with gated fusion and ablation switches."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Reversible Instance Normalisation  (Kim et al., ICLR 2022)
# ═══════════════════════════════════════════════════════════════════════════
class RevIN(nn.Module):
    """Per-instance, per-variable normalisation with learnable affine."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """x: (B,T,D) → (x_norm, (mean, std))  where mean,std are (B,D)."""
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()
        x_norm = (x - mean) / std
        if self.weight is not None:
            x_norm = x_norm * self.weight + self.bias
        return x_norm, (mean.squeeze(1), std.squeeze(1))

    def denorm(
        self, y: torch.Tensor, stats: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """y: (B,H,D) → de-normalised (B,H,D)."""
        mean, std = stats
        if self.weight is not None:
            y = (y - self.bias) / (self.weight + self.eps)
        return y * std.unsqueeze(1) + mean.unsqueeze(1)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  S4D Kernel — diagonal state-space convolution kernel
# ═══════════════════════════════════════════════════════════════════════════
class S4DKernel(nn.Module):
    r"""Causal convolution kernel from a diagonal SSM.

    Continuous-time dynamics (per channel h, state dim n):

    .. math::
        x'(t) = A_h x(t) + B_h u(t), \quad y(t) = \operatorname{Re}[C_h x(t)]

    with  :math:`A_h = \operatorname{diag}(\lambda_1, \ldots, \lambda_N)`,
    :math:`\operatorname{Re}(\lambda_n) < 0`.

    HiPPO-LegS initialisation:
        :math:`\lambda_n = -(n+0.5) + i\,n\pi` for n = 0 … N−1

    Kernel:
        :math:`K(\ell) = 2\,\operatorname{Re}\!\bigl[\sum_n C_n B_n \Delta\,
        \bar A_n^{\ell}\bigr]` for ℓ = 0 … L−1.
    """

    def __init__(
        self,
        d_model: int,
        N: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model, self.N = d_model, N

        # -- eigenvalue init (HiPPO-LegS) -----------------------------------
        log_A_real = torch.log(0.5 + torch.arange(N, dtype=torch.float32))
        A_imag = math.pi * torch.arange(N, dtype=torch.float32)
        self.log_A_real = nn.Parameter(
            log_A_real.unsqueeze(0).expand(d_model, -1).clone()
        )
        self.A_imag = nn.Parameter(
            A_imag.unsqueeze(0).expand(d_model, -1).clone()
        )

        # -- B (real), C (complex stored as [re, im]) -----------------------
        self.B = nn.Parameter(torch.randn(d_model, N) * N**-0.5)
        self.C = nn.Parameter(torch.randn(d_model, N, 2) * N**-0.5)

        # -- learnable step size Δ -------------------------------------------
        log_dt = (
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        self.log_dt = nn.Parameter(log_dt)

    def forward(self, L: int) -> torch.Tensor:
        """Return causal kernel of shape (d_model, L)."""
        dt = F.softplus(self.log_dt)                              # (H,)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag       # (H,N)
        dtA = A * dt.unsqueeze(-1)                                # (H,N)
        C = torch.view_as_complex(self.C)                         # (H,N)

        # Vandermonde product  K(ℓ) = Σ_n C_n B_n dt  Ā_n^ℓ
        arange = torch.arange(L, device=A.device, dtype=torch.float32)
        powers = torch.exp(
            dtA.unsqueeze(-1) * arange.unsqueeze(0).unsqueeze(0)
        )                                                         # (H,N,L)
        CB = C * self.B.to(C.dtype) * dt.unsqueeze(-1).to(C.dtype)
        K = torch.einsum("hn, hnl -> hl", CB, powers)            # (H,L)
        return 2.0 * K.real


# ═══════════════════════════════════════════════════════════════════════════
# 3.  S4D Block — one structured state-space layer
# ═══════════════════════════════════════════════════════════════════════════
class S4DBlock(nn.Module):
    """u → [causal FFT-conv + D·u skip] → GLU → dropout → +residual → LN"""

    def __init__(
        self,
        d_model: int,
        N: int = 64,
        dropout: float = 0.1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.kernel = S4DKernel(d_model, N, dt_min, dt_max)
        self.D = nn.Parameter(torch.randn(d_model))
        self.out_proj = nn.Linear(d_model, 2 * d_model)   # for GLU
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, L, H) → (B, L, H)"""
        residual = u
        B_sz, L, H = u.shape

        K = self.kernel(L)                                    # (H, L)
        u_T = u.transpose(1, 2)                               # (B, H, L)

        # causal FFT convolution (zero-pad to 2L for linear conv)
        fft_n = 2 * L
        K_f = torch.fft.rfft(K, n=fft_n)                     # (H, F)
        u_f = torch.fft.rfft(u_T, n=fft_n)                   # (B, H, F)
        y = torch.fft.irfft(K_f.unsqueeze(0) * u_f, n=fft_n) # (B,H,fft_n)
        y = y[..., :L]                                        # trim

        # skip from D
        y = y + self.D[None, :, None] * u_T
        y = y.transpose(1, 2)                                 # (B, L, H)

        # GLU + dropout + residual + norm
        y = self.drop(F.glu(self.out_proj(y), dim=-1))
        return self.norm(y + residual)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Scale-specific SSM  (downsample → S4D stack → upsample)
# ═══════════════════════════════════════════════════════════════════════════
class ScaleSSM(nn.Module):
    """Multi-layer S4D at one temporal scale.

    1. Stride-downsample (causal: keep every *stride*-th step)
    2. *n_layers* S4DBlocks
    3. Repeat-interleave upsample back to original length
    """

    def __init__(
        self,
        d_model: int,
        N: int = 64,
        stride: int = 1,
        n_layers: int = 2,
        dropout: float = 0.1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        use_film: bool = False,
    ):
        super().__init__()
        self.stride = stride
        self.layers = nn.ModuleList(
            [S4DBlock(d_model, N, dropout, dt_min, dt_max) for _ in range(n_layers)]
        )
        self.film = nn.Linear(d_model, 2 * d_model) if use_film else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, H) → (B, T, H). Optional cond FiLM-modulates the input."""
        if cond is not None and self.film is not None:
            gb = self.film(cond)
            gamma, beta = gb.chunk(2, dim=-1)
            x = x * (1.0 + torch.tanh(gamma)) + beta
        T = x.size(1)
        h = x[:, :: self.stride, :] if self.stride > 1 else x

        for layer in self.layers:
            h = layer(h)

        if self.stride > 1:
            h = h.repeat_interleave(self.stride, dim=1)[:, :T, :]
        return h


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Cross-scale gated conditioning
# ═══════════════════════════════════════════════════════════════════════════
class CrossScaleGate(nn.Module):
    r"""Learned element-wise gate mixing base signal *x* with conditioning
    *c* from a slower scale:

    .. math::
        g = \sigma(W_g\,[x \| c]),\quad
        \mathrm{out} = g \odot c + (1-g) \odot x
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([x, c], dim=-1)))
        return g * c + (1.0 - g) * x


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Gated Fusion — soft-max gating over K scales
# ═══════════════════════════════════════════════════════════════════════════
class GatedFusion(nn.Module):
    r"""Softmax-gated aggregation of K scale outputs.

    .. math::
        g = \mathrm{softmax}(\mathrm{MLP}([s_1 \| \cdots \| s_K]))\;,
        \quad z = \sum_k g_k \odot s_k
    """

    def __init__(self, d_model: int, n_scales: int = 3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * n_scales, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_scales),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, *scale_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Each input: (B,T,H).  Returns fused (B,T,H) & gates (B,T,K)."""
        cat = torch.cat(scale_outputs, dim=-1)
        gates = torch.softmax(self.gate(cat), dim=-1)           # (B,T,K)
        stacked = torch.stack(scale_outputs, dim=-1)            # (B,T,H,K)
        fused = (stacked * gates.unsqueeze(2)).sum(dim=-1)      # (B,T,H)
        return self.norm(fused), gates


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Forecast Decoder
# ═══════════════════════════════════════════════════════════════════════════
class ForecastDecoder(nn.Module):
    """MLP readout: last-step (or pre-pooled) latent → (H, D) point forecast."""

    def __init__(
        self, d_model: int, n_vars: int, horizon: int, dropout: float = 0.1
    ):
        super().__init__()
        self.horizon, self.n_vars = horizon, n_vars
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_vars * horizon),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, H) or (B, H) → (B, horizon, n_vars)"""
        if z.ndim == 3:
            z = z[:, -1, :]
        return self.mlp(z).view(-1, self.horizon, self.n_vars)


# ═══════════════════════════════════════════════════════════════════════════
# 8.  HCSSN — main model
# ═══════════════════════════════════════════════════════════════════════════
class HCSSN(nn.Module):
    """Hierarchical Causal State-Space Network.

    Parameters
    ----------
    n_vars        : number of input / output variables
    horizon       : forecast length
    embed_dim     : embedding width
    hidden_dim    : SSM channel width
    ssm_state_dim : state dimension N inside each S4D kernel
    n_ssm_layers  : S4DBlocks per scale
    dropout       : dropout probability
    slow_stride   : temporal stride for slow SSM  (default 24)
    medium_stride : temporal stride for medium SSM (default 6)
    fast_stride   : temporal stride for fast SSM   (default 1)
    use_revin     : enable/disable RevIN
    use_gating    : True=softmax gates, False=additive fusion (ablation)
    use_hierarchy : True=3 scales, False=single stacked S4D (ablation)
    """

    def __init__(
        self,
        n_vars: int,
        horizon: int = 96,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        ssm_state_dim: int = 64,
        n_ssm_layers: int = 2,
        dropout: float = 0.1,
        slow_stride: int = 24,
        medium_stride: int = 6,
        fast_stride: int = 1,
        use_revin: bool = True,
        use_gating: bool = True,
        use_hierarchy: bool = True,
        chain_gates: bool = False,
        inject_film: bool = False,
        decode_all_scales: bool = False,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.use_revin = use_revin
        self.use_gating = use_gating
        self.use_hierarchy = use_hierarchy
        self.chain_gates = chain_gates
        self.inject_film = inject_film
        self.decode_all_scales = decode_all_scales and use_hierarchy

        # RevIN
        self.revin = RevIN(n_vars) if use_revin else None

        # Embedding
        self.embed = nn.Sequential(
            nn.Linear(n_vars, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Hierarchical or single-scale SSMs
        if use_hierarchy:
            self.slow_ssm = ScaleSSM(
                hidden_dim, ssm_state_dim, slow_stride, n_ssm_layers, dropout,
                dt_min=0.01, dt_max=1.0,
            )
            self.medium_ssm = ScaleSSM(
                hidden_dim, ssm_state_dim, medium_stride, n_ssm_layers, dropout,
                dt_min=0.001, dt_max=0.1, use_film=inject_film,
            )
            self.fast_ssm = ScaleSSM(
                hidden_dim, ssm_state_dim, fast_stride, n_ssm_layers, dropout,
                dt_min=0.0001, dt_max=0.01, use_film=inject_film,
            )
            self.gate_slow_to_med = CrossScaleGate(hidden_dim)
            self.gate_med_to_fast = CrossScaleGate(hidden_dim)
            self.fusion = GatedFusion(hidden_dim, 3) if use_gating else None
        else:
            self.single_ssm = ScaleSSM(
                hidden_dim, ssm_state_dim, 1, n_ssm_layers * 3, dropout,
                dt_min=0.001, dt_max=0.1,
            )

        # Decoder. Default: last fused timestep. `--decode_all_scales`:
        # Linear mix of each scale's last token so the head cannot ignore
        # slow/medium by putting all fusion mass on the fast channel.
        self.scale_readout = (
            nn.Linear(hidden_dim * 3, hidden_dim) if self.decode_all_scales else None
        )
        self.decoder = ForecastDecoder(hidden_dim, n_vars, horizon, dropout)

    # ===================================================================
    #  Forward
    # ===================================================================
    def forward(
        self,
        x: torch.Tensor,
        return_decomposition: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        x : (B, T, D) — look-back window
        return_decomposition : if True, also yield per-scale + gate info

        Returns  pred (B, H, D) [, decomp dict]
        """
        # RevIN
        if self.revin is not None:
            x_norm, stats = self.revin(x)
        else:
            x_norm, stats = x, None

        # Embed
        z0 = self.embed(x_norm)       # (B, T, embed_dim)
        z = self.input_proj(z0)       # (B, T, hidden_dim)

        # SSMs
        if self.use_hierarchy:
            s_slow = self.slow_ssm(z)
            z_med = self.gate_slow_to_med(z, s_slow)
            s_medium = self.medium_ssm(
                z_med, cond=s_slow if self.inject_film else None
            )
            fast_base = z_med if self.chain_gates else z
            z_fast = self.gate_med_to_fast(fast_base, s_medium)
            s_fast = self.fast_ssm(
                z_fast, cond=s_medium if self.inject_film else None
            )

            if self.fusion is not None:
                fused, gates = self.fusion(s_slow, s_medium, s_fast)
            else:
                fused = s_slow + s_medium + s_fast
                gates = None
        else:
            fused = self.single_ssm(z)
            s_slow = s_medium = s_fast = gates = None

        # Decode
        if self.decode_all_scales:
            feat = torch.cat(
                [s_slow[:, -1, :], s_medium[:, -1, :], s_fast[:, -1, :]], dim=-1
            )
            pred = self.decoder(self.scale_readout(feat))
        else:
            pred = self.decoder(fused)

        # De-RevIN
        if self.revin is not None:
            pred = self.revin.denorm(pred, stats)

        if return_decomposition:
            return pred, {
                "slow": s_slow, "medium": s_medium,
                "fast": s_fast, "gates": gates,
            }
        return pred

    # ===================================================================
    #  Regularisation losses
    # ===================================================================
    def scale_separation_loss(
        self, s_slow: torch.Tensor, s_med: torch.Tensor, s_fast: torch.Tensor,
    ) -> torch.Tensor:
        """Minimise cosine similarity of normalised power spectra across
        scales, encouraging each scale to capture different frequencies."""
        def _profile(s: torch.Tensor) -> torch.Tensor:
            S = torch.fft.rfft(s, dim=1)
            pw = (S.real ** 2 + S.imag ** 2).mean(dim=(0, 2))
            return F.normalize(pw.unsqueeze(0), dim=-1).squeeze(0)

        ps, pm, pf = _profile(s_slow), _profile(s_med), _profile(s_fast)
        sim = (
            F.cosine_similarity(ps.unsqueeze(0), pm.unsqueeze(0))
            + F.cosine_similarity(pm.unsqueeze(0), pf.unsqueeze(0))
            + F.cosine_similarity(ps.unsqueeze(0), pf.unsqueeze(0))
        )
        return sim / 3.0

    def orthogonality_loss(
        self, s_slow: torch.Tensor, s_med: torch.Tensor, s_fast: torch.Tensor,
    ) -> torch.Tensor:
        """Penalise cosine similarity of latent representations."""
        def _cs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return F.cosine_similarity(
                a.reshape(-1, a.size(-1)), b.reshape(-1, b.size(-1)), dim=-1,
            ).abs().mean()

        return (_cs(s_slow, s_med) + _cs(s_med, s_fast)
                + _cs(s_slow, s_fast)) / 3.0

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        decomp: Optional[Dict] = None,
        lambda_sep: float = 0.01,
        lambda_orth: float = 0.01,
        lambda_entropy: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """MSE + λ_sep · scale_separation + λ_orth · orthogonality [+ entropy]."""
        mse = F.mse_loss(pred, target)
        comps: Dict[str, float] = {"mse": mse.item()}
        loss = mse

        if decomp is not None and decomp.get("slow") is not None:
            s_s, s_m, s_f = decomp["slow"], decomp["medium"], decomp["fast"]
            if lambda_sep > 0:
                sep = self.scale_separation_loss(s_s, s_m, s_f)
                loss = loss + lambda_sep * sep
                comps["scale_sep"] = sep.item()
            if lambda_orth > 0:
                orth = self.orthogonality_loss(s_s, s_m, s_f)
                loss = loss + lambda_orth * orth
                comps["orthogonality"] = orth.item()
            if lambda_entropy > 0 and decomp.get("gates") is not None:
                g = decomp["gates"].clamp(min=1e-8)
                # Maximise fusion entropy to discourage collapse onto one scale.
                ent = -(g * g.log()).sum(dim=-1).mean()
                loss = loss - lambda_entropy * ent
                comps["gate_entropy"] = ent.item()

        comps["total"] = loss.item()
        return loss, comps

    # ===================================================================
    #  Utility
    # ===================================================================
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        return (
            f"HCSSN  n_vars={self.n_vars}  horizon={self.horizon}  "
            f"hidden={self.hidden_dim}\n"
            f"  hierarchy={self.use_hierarchy}  gating={self.use_gating}  "
            f"revin={self.use_revin}  decode_all_scales={self.decode_all_scales}\n"
            f"  trainable parameters: {self.num_params():,}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Factory helper
# ═══════════════════════════════════════════════════════════════════════════
def build_hcssn(
    n_vars: int,
    horizon: int = 96,
    preset: str = "default",
    **overrides,
) -> HCSSN:
    """Convenience constructor with sensible presets.

    Presets: default, paper, small, ablation_single, ablation_nogating
    """
    configs = {
        "default": dict(
            embed_dim=64, hidden_dim=128, ssm_state_dim=64,
            n_ssm_layers=2, dropout=0.1,
            slow_stride=24, medium_stride=6, fast_stride=1,
        ),
        "paper": dict(
            embed_dim=64, hidden_dim=128, ssm_state_dim=64,
            n_ssm_layers=2, dropout=0.1,
            slow_stride=60, medium_stride=10, fast_stride=1,
        ),
        "small": dict(
            embed_dim=32, hidden_dim=64, ssm_state_dim=32,
            n_ssm_layers=1, dropout=0.0,
            slow_stride=12, medium_stride=4, fast_stride=1,
        ),
        "ablation_single": dict(
            embed_dim=64, hidden_dim=128, ssm_state_dim=64,
            n_ssm_layers=2, dropout=0.1,
            use_hierarchy=False,
        ),
        "ablation_nogating": dict(
            embed_dim=64, hidden_dim=128, ssm_state_dim=64,
            n_ssm_layers=2, dropout=0.1,
            slow_stride=24, medium_stride=6, fast_stride=1,
            use_gating=False,
        ),
    }
    cfg = configs.get(preset, configs["default"])
    cfg.update(overrides)
    return HCSSN(n_vars=n_vars, horizon=horizon, **cfg)
