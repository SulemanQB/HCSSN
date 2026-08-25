#!/usr/bin/env python3
"""Train HCSSN. Example: python train.py --data ../data/ETTh1.csv --epochs 50"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import optim

from src.data import create_dataloaders
from src.metrics import (
    causality_preservation_score,
    compute_metrics,
    horizon_metrics,
    scale_separation_quality,
)
from src.model import HCSSN, build_hcssn


# ═══════════════════════════════════════════════════════════════════════════
# Training logic
# ═══════════════════════════════════════════════════════════════════════════
def train(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"[train] device={device}")

    # ---- data -----------------------------------------------------------
    train_loader, val_loader, test_loader, n_features = create_dataloaders(
        csv_path=args.data,
        batch_size=args.batch_size,
        context_len=args.context,
        pred_len=args.horizon,
        stride=args.stride,
    )

    # ---- model ----------------------------------------------------------
    overrides = dict(
        use_revin=not args.no_revin,
        use_gating=not args.no_gating,
        use_hierarchy=not args.no_hierarchy,
    )
    if args.hidden_dim is not None:
        overrides["hidden_dim"] = args.hidden_dim
    if args.n_ssm_layers is not None:
        overrides["n_ssm_layers"] = args.n_ssm_layers
    overrides["chain_gates"] = args.chain_gates
    overrides["inject_film"] = args.inject_film
    overrides["decode_all_scales"] = args.decode_all_scales
    model = build_hcssn(
        n_vars=n_features,
        horizon=args.horizon,
        preset=args.preset,
        **overrides,
    ).to(device)
    print(model.summary())

    # ---- optimiser + scheduler ------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ---- training loop --------------------------------------------------
    best_val_loss = float("inf")
    best_val_epoch = 0
    patience_counter = 0
    ckpt_path = Path(args.save_dir) / "best_model.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # — train —
        model.train()
        train_loss_sum, train_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred, decomp = model(x, return_decomposition=True)
            loss, comps = model.compute_loss(
                pred, y, decomp,
                lambda_sep=args.lambda_sep,
                lambda_orth=args.lambda_orth,
                lambda_entropy=args.lambda_entropy,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            train_loss_sum += comps["total"]
            train_batches += 1

        scheduler.step()
        avg_train = train_loss_sum / max(train_batches, 1)

        # — validate —
        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss_sum += torch.nn.functional.mse_loss(pred, y).item()
                val_batches += 1
        avg_val = val_loss_sum / max(val_batches, 1)

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={avg_train:.5f}  val={avg_val:.5f}  "
            f"lr={lr_now:.2e}  time={elapsed:.1f}s"
        )

        # — early stopping —
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_val_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[train] early stopping at epoch {epoch}")
                break

    # ---- load best & test -----------------------------------------------
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print("[test] loaded best checkpoint")

    model.eval()
    all_pred, all_target = [], []
    gate_sums = None
    gate_n = 0
    ssq_acc = None
    ssq_n = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred, decomp = model(x, return_decomposition=True)
            all_pred.append(pred.cpu())
            all_target.append(y.cpu())
            if decomp is not None and decomp.get("gates") is not None:
                g = decomp["gates"].cpu()
                means = g.mean(dim=(0, 1))
                gate_sums = means if gate_sums is None else gate_sums + means
                gate_n += 1
            if decomp is not None and decomp.get("slow") is not None:
                ssq = scale_separation_quality(
                    decomp["slow"].cpu(), decomp["medium"].cpu(), decomp["fast"].cpu()
                )
                if ssq_acc is None:
                    ssq_acc = {k: float(v) for k, v in ssq.items()}
                else:
                    for k, v in ssq.items():
                        ssq_acc[k] += float(v)
                ssq_n += 1

    pred_cat = torch.cat(all_pred, dim=0)
    targ_cat = torch.cat(all_target, dim=0)

    # ---- aggregate metrics ---------------------------------------------
    results = compute_metrics(pred_cat, targ_cat)
    results.update(horizon_metrics(pred_cat, targ_cat))
    results["cps"] = causality_preservation_score(pred_cat, targ_cat)

    if ssq_acc is not None and ssq_n > 0:
        results.update({k: v / ssq_n for k, v in ssq_acc.items()})

    if gate_sums is not None and gate_n > 0:
        gmean = gate_sums / gate_n
        results["gate_slow_mean"] = float(gmean[0])
        results["gate_medium_mean"] = float(gmean[1])
        results["gate_fast_mean"] = float(gmean[2])
        results["gate_collapsed"] = bool(float(gmean.max()) > 0.90)

    print("\n═══ TEST RESULTS ═══")
    for k, v in results.items():
        print(f"  {k:20s}: {v:.6f}")

    # ---- save results ---------------------------------------------------
    results["dataset"] = Path(args.data).stem
    results["horizon"] = args.horizon
    results["context"] = args.context
    results["preset"] = args.preset
    results["params"] = model.num_params()
    results["hierarchy"] = not args.no_hierarchy
    results["gating"] = not args.no_gating
    results["revin"] = not args.no_revin
    results["seed"] = args.seed
    results["hidden_dim"] = int(model.hidden_dim)
    results["decode_all_scales"] = bool(getattr(model, "decode_all_scales", False))
    results["best_val_epoch"] = int(best_val_epoch)
    results["best_val_loss"] = float(best_val_loss)
    results["lr"] = float(args.lr)

    out_path = Path(args.save_dir) / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {ckpt_path}")
    print(f"[saved] {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    p = argparse.ArgumentParser(description="Train HCSSN")

    # data
    p.add_argument(
        "--data", type=str,
        default=str(Path(__file__).resolve().parent / "data" / "ETTh1.csv"),
        help="Path to CSV dataset",
    )
    p.add_argument("--context", type=int, default=336, help="Look-back length")
    p.add_argument("--horizon", type=int, default=96, help="Forecast horizon")
    p.add_argument("--stride", type=int, default=1, help="Sliding-window stride")

    # training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0, help="Gradient clip norm")
    p.add_argument("--patience", type=int, default=10, help="Early-stop patience")

    # regularisation
    p.add_argument("--lambda_sep", type=float, default=0.01,
                    help="Scale-separation loss weight")
    p.add_argument("--lambda_orth", type=float, default=0.01,
                    help="Orthogonality loss weight")
    p.add_argument("--lambda_entropy", type=float, default=0.0,
                    help="Fusion-gate entropy bonus (discourages collapse; 0=off)")
    p.add_argument("--chain_gates", action="store_true",
                    help="Condition fast scale on medium mix, not raw embedding")
    p.add_argument("--inject_film", action="store_true",
                    help="FiLM-modulate next ScaleSSM input with slower-scale output")
    p.add_argument("--decode_all_scales", action="store_true",
                    help="Decode from concat of last tokens of all three scales")

    # model
    p.add_argument("--preset", type=str, default="default",
                    choices=["default", "paper", "small",
                             "ablation_single", "ablation_nogating"])

    # ablation switches
    p.add_argument("--no_hierarchy", action="store_true",
                    help="Ablation: use single-scale SSM")
    p.add_argument("--no_gating", action="store_true",
                    help="Ablation: additive fusion instead of gated")
    p.add_argument("--no_revin", action="store_true",
                    help="Ablation: disable RevIN")
    p.add_argument("--hidden_dim", type=int, default=None,
                    help="Override SSM hidden width (for matched-capacity flat S4D)")
    p.add_argument("--n_ssm_layers", type=int, default=None,
                    help="Override number of S4D blocks per scale")

    # misc
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    # reproducibility
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train(args)


if __name__ == "__main__":
    main()
