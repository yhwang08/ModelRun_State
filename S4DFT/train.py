# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Yihan Wang @ University of Oklahoma
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import argparse, os, sys, time, random

os.environ["PYTHONUNBUFFERED"] = "1"

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# --- add ../common to path (same as your LSTM script) ---
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
from common.regional_data import RegionalDataset, SlidingWindows, DYN_DIM
from common.utils import (
    list_wsids, save_hparams, load_npz_for_wsid,
    write_timeseries_txt, write_stats_table
)
from common.plotting import plot_ts
from common.metrics import compute_all_metrics
from common.models import HOPE  # S4D/SSM core


def log(msg): print(msg, flush=True)


def format_hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def save_ckpt(path: Path, epoch: int, model: nn.Module, opt: optim.Optimizer, va_loss: float,
              extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": opt.state_dict(),
        "val_loss": float(va_loss),
    }
    if extra: payload.update(extra)
    torch.save(payload, path)


# ---------- Loss (same semantics as LSTM script: y + q_std in SAME space) ----------
class NSELoss(torch.nn.Module):
    """Calculate (batch-wise) NSE Loss with 1/(std+eps)^2 weights."""

    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor):
        se = (y_pred.view(-1) - y_true.view(-1)) ** 2
        qs = q_stds.view(-1)
        weights = 1.0 / (qs + self.eps) ** 2
        return torch.mean(weights * se)


def make_loader(ds, batch, shuffle, num_workers):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


# ---------- Cosine schedule with linear warmup (no papercode dependency) ----------
class WarmupCosine:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float, min_lr: float):
        self.optimizer = optimizer
        self.warm = max(0, int(warmup_epochs))
        self.total = max(1, int(total_epochs))
        self.base = float(base_lr)
        self.min = float(min_lr)
        self.t = 0
        self._apply_lr(self._lr_at(0))

    def _lr_at(self, t):
        if t < self.warm and self.warm > 0:
            return self.min + (self.base - self.min) * (t / self.warm)
        # cosine from base -> min over [warm..total]
        tt = min(max(t - self.warm, 0), max(self.total - self.warm, 1))
        cos = 0.5 * (1 + np.cos(np.pi * tt / max(self.total - self.warm, 1)))
        return self.min + (self.base - self.min) * cos

    def step(self):
        self.t += 1
        self._apply_lr(self._lr_at(self.t))

    def _apply_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr


def setup_optimizer(model: nn.Module, lr: float, weight_decay: float, epochs: int, warmup_epochs: int, lr_min: float):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = WarmupCosine(opt, warmup_epochs=warmup_epochs, total_epochs=epochs, base_lr=lr, min_lr=lr_min)
    return opt, sched


# ---------- Adapter so training loop mirrors LSTM head usage ----------
class S4DFTAdapter(nn.Module):
    """
    Wrap HOPE (S4D/SSM) to behave like the LSTM head:
      - forward(x) -> [B, L] so we can index [:, -1] if needed.
      - Internally HOPE returns a pooled [B, 1] prediction for the window.
    """

    def __init__(self, input_dim: int, d_model: int, n_layers: int, d_state: int, dropout: float,
                 lr: float, lr_min: float, lr_dt: float, min_dt: float, max_dt: float, cfr: float, cfi: float,
                 wd: float):
        super().__init__()
        cfg = dict(
            lr_min=lr_min, lr=lr, lr_dt=lr_dt, d_state=d_state,
            min_dt=min_dt, max_dt=max_dt, cfr=cfr, cfi=cfi, wd=wd
        )
        self.core = HOPE(d_input=input_dim, d_output=1, d_model=d_model,
                         n_layers=n_layers, dropout=dropout, cfg=cfg)

    def forward(self, x):  # x: [B, L, F]
        y_last = self.core(x).squeeze(-1)  # [B]
        return y_last.unsqueeze(1).repeat(1, x.size(1))  # [B, L]


@torch.no_grad()
def eval_epoch(model, loader, device, loss_fn):
    model.eval()
    total = 0.0
    n = 0
    for xb, yb, qstd in loader:
        xb = xb.to(device)
        yb = yb.to(device).view(-1)
        qstd = qstd.to(device).view(-1)
        y = model(xb)[:, -1].contiguous().view(-1)  # adapter returns [B,L]
        loss = loss_fn(y, yb, qstd)
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state_dir",
                    default=r"C:\Users\yihan\OneDrive - University of Oklahoma\OU\code\runs\Pennsylvania")
    ap.add_argument("--seq_len", type=int, default=365)

    # --------- S4D / SSM hyperparameters (keep your S4D setup) ----------
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-2)  # base LR for AdamW
    ap.add_argument("--lr_min", type=float, default=1e-3)  # min LR at cosine floor
    ap.add_argument("--warmup", type=int, default=10)  # warmup epochs for scheduler
    ap.add_argument("--weight_decay", type=float, default=1e-2)

    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--d_state", type=int, default=64)
    ap.add_argument("--ssm_dropout", type=float, default=0.15)

    # SSM internals passed to HOPE cfg (kept for parity; not all may be used)
    ap.add_argument("--lr_dt", type=float, default=0.0)
    ap.add_argument("--min_dt", type=float, default=1e-3)
    ap.add_argument("--max_dt", type=float, default=1.0)
    ap.add_argument("--cfr", type=float, default=10.0)
    ap.add_argument("--cfi", type=float, default=10.0)
    ap.add_argument("--wd", type=float, default=0.0)

    # data/scaler flags identical to LSTM script
    ap.add_argument("--normalize_y", action="store_true", default=True,
                    help="Normalize targets using TRAIN-union stats (per state).")
    ap.add_argument("--no_norm_precip", action="store_true", default=False,
                    help="If True, DO NOT normalize precipitation.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=200)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    state_dir = Path(args.state_dir)
    model_input = state_dir / "model_input"
    out_dir = state_dir / "S4DFT output"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    wsids = list_wsids(str(model_input))
    if not wsids:
        raise RuntimeError(f"No *.npz in {model_input}")

    # ----------------------------- DATASETS (same as LSTM script) -----------------------------
    ds_union = RegionalDataset(
        str(model_input), wsids, split="train", seq_len=args.seq_len,
        scaler=None, compute_state_scaler=True,
        normalize_y=args.normalize_y, no_norm_precip=args.no_norm_precip
    )
    scaler = ds_union.scaler

    g = torch.Generator().manual_seed(args.seed)
    n_val = int(round(0.10 * len(ds_union)))
    n_tr = len(ds_union) - n_val
    ds_tr, ds_va = random_split(ds_union, [n_tr, n_val], generator=g)

    ds_te = RegionalDataset(
        str(model_input), wsids, split="test", seq_len=args.seq_len,
        scaler=scaler, compute_state_scaler=False,
        normalize_y=args.normalize_y, no_norm_precip=args.no_norm_precip
    )

    dl_tr = make_loader(ds_tr, args.batch, True, args.num_workers)
    dl_va = make_loader(ds_va, args.batch, False, args.num_workers)
    dl_te = make_loader(ds_te, args.batch, False, args.num_workers)

    log(f"[info] wsids: {len(wsids)} -> {wsids[:5]}{'...' if len(wsids) > 5 else ''}")
    log(f"[info] samples: union={len(ds_union)}  train={len(ds_tr)}  val={len(ds_va)}  test={len(ds_te)}")
    log(f"[info] batches: train={len(dl_tr)}  val={len(dl_va)}  test={len(dl_te)}")
    log(f"[info] normalize_y={args.normalize_y}  no_norm_precip={args.no_norm_precip}")

    # ----------------------------- MODEL + OPTIM -----------------------------
    input_dim = ds_union.x.shape[-1]
    model = S4DFTAdapter(
        input_dim=input_dim,
        d_model=args.d_model, n_layers=args.n_layers, d_state=args.d_state, dropout=args.ssm_dropout,
        lr_min=args.lr_min, lr=args.lr, lr_dt=args.lr_dt, min_dt=args.min_dt, max_dt=args.max_dt,
        cfr=args.cfr, cfi=args.cfi, wd=args.wd
    ).to(device)

    nse_loss = NSELoss(eps=0.1)
    opt, sched = setup_optimizer(model, lr=args.lr, weight_decay=args.weight_decay,
                                 epochs=args.epochs, warmup_epochs=args.warmup, lr_min=args.lr_min)

    save_hparams({
        "seed": args.seed,
        "model": "S4DFT",
        "input_dim": input_dim,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "d_state": args.d_state,
        "dropout": args.ssm_dropout,
        "seq_len": args.seq_len,
        "batch": args.batch,
        "epochs": args.epochs,
        "optimizer": "AdamW",
        "lr": args.lr,
        "lr_min": args.lr_min,
        "warmup": args.warmup,
        "weight_decay": args.weight_decay,
        "normalize_y": args.normalize_y,
        "inputs_normalized": True,
        "no_norm_precip": args.no_norm_precip,
        "loss": "NSELoss(eps=0.1) with q_std in same space as y",
        "scaler": "per-state on (idx_tr ∪ idx_va) UNION",
        "val_split": "deterministic 90/10 random_split on union",
        "test": "idx_te with same scaler"
    }, out_dir)

    # ----------------- TRAIN -----------------
    best_va = float("inf")
    best_state = None
    start_all = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_start = time.time()
        total_loss = 0.0
        total_count = 0

        log(f"\n=== Epoch {ep}/{args.epochs} (lr={opt.param_groups[0]['lr']:.3e}) ===")
        for bi, (xb, yb, qstd) in enumerate(dl_tr, 1):
            xb = xb.to(device)
            yb = yb.to(device).view(-1)
            qstd = qstd.to(device).view(-1)

            opt.zero_grad()
            y_hat = model(xb)[:, -1].contiguous().view(-1)  # [B]
            loss = nse_loss(y_hat, yb, qstd)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()

            total_loss += loss.item() * xb.size(0)
            total_count += xb.size(0)

            if (bi % args.log_every) == 0 or bi == len(dl_tr):
                elapsed = time.time() - ep_start
                avg_train = total_loss / max(total_count, 1)
                log(f"[train] epoch {ep} batch {bi}/{len(dl_tr)} "
                    f"avg_loss={avg_train:.4f} lr={opt.param_groups[0]['lr']:.2e} elapsed={format_hms(elapsed)}")

        va_loss = eval_epoch(model, dl_va, device, nse_loss)
        if sched is not None: sched.step()
        ep_time = time.time() - ep_start
        avg_train = total_loss / max(total_count, 1)
        log(f"[summary] Epoch {ep:03d} | train {avg_train:.4f} | val {va_loss:.4f} | time {format_hms(ep_time)}")

        # --- save per-epoch checkpoint ---
        save_ckpt(ckpt_dir / f"epoch_{ep:03d}.pt", ep, model, opt, va_loss,
                  extra={"train_loss": avg_train, "seed": args.seed})

        if va_loss < best_va:
            best_va = va_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            save_ckpt(ckpt_dir / "best.pt", ep, model, opt, va_loss, extra={"train_loss": avg_train, "seed": args.seed})
            log(f"[checkpoint] new best val {best_va:.4f} (epoch {ep})")

    # save last
    save_ckpt(ckpt_dir / "last.pt", args.epochs, model, opt, va_loss, extra={"seed": args.seed})

    if best_state is not None:
        model.load_state_dict(best_state)

    total_time = time.time() - start_all
    log(f"\n=== Training done in {format_hms(total_time)}. Best val: {best_va:.4f} ===")

    # --------------- EXPORT (PHYSICAL, CONTINUOUS MODELING WINDOW) ---------------
    # Matches your LSTM export pathing & slicing
    model.eval()
    for wid in wsids:
        d = load_npz_for_wsid(model_input, wid)
        dates_full = d["dates"]
        y_phys_full = d["y"].astype(np.float32)

        idx_tr = d["idx_tr"];
        idx_va = d["idx_va"];
        idx_te = d["idx_te"]
        if (idx_tr.size + idx_va.size + idx_te.size) == 0:
            continue
        idx_all = np.unique(np.concatenate([idx_tr, idx_va, idx_te]))
        lo, hi = int(idx_all.min()), int(idx_all.max())

        dates = dates_full[lo:hi + 1]
        y_phys = y_phys_full[lo:hi + 1]

        win = SlidingWindows(d["X"], d["y"], idx_all, args.seq_len)

        preds_phys = {}
        with torch.no_grad():
            for j in range(len(win)):
                xr_raw, _, t = win.get(j)  # raw features -> normalize like training
                xr = xr_raw.copy().astype(np.float32)
                # dynamic
                xr[:, :DYN_DIM] = (xr[:, :DYN_DIM]
                                   - scaler.dyn_mean[None, :]) / scaler.dyn_std[None, :]
                # static if present
                if xr.shape[1] > DYN_DIM and (scaler.sta_mean is not None) and (scaler.sta_std is not None):
                    xr[:, DYN_DIM:] = (xr[:, DYN_DIM:]
                                       - scaler.sta_mean[None, :]) / scaler.sta_std[None, :]

                xbt = torch.from_numpy(xr[None, ...]).float().to(device)
                yhat_norm = model(xbt)[:, -1].item()

                # invert y normalization if used in training
                if args.normalize_y and (scaler.y_std is not None) and (scaler.y_mean is not None):
                    yhat_phys = yhat_norm * float(scaler.y_std) + float(scaler.y_mean)
                else:
                    yhat_phys = yhat_norm

                # clamp negatives
                yhat_phys = max(float(yhat_phys), 0.0)
                preds_phys[int(t)] = yhat_phys

        # continuous series over [lo..hi]
        sim_all = np.full_like(y_phys, np.nan, dtype=float)
        for tt, val in preds_phys.items():
            if lo <= tt <= hi:
                sim_all[tt - lo] = val

        out_txt_all = out_dir / f"{wid}_all.csv"
        write_timeseries_txt(out_txt_all, dates, y_phys, sim_all)
        plot_ts(dates, y_phys, sim_all, f"S4DFT {wid} (train+val+test)", out_dir / f"{wid}_all.png")

        # per-split (truncated to [lo..hi])
        def sliced_series(idx_split):
            mask = np.isin(np.arange(lo, hi + 1), idx_split)
            arr = np.full_like(y_phys, np.nan, dtype=float)
            arr[mask] = sim_all[mask]
            return arr

        for split_name, idx_split in [("train", idx_tr), ("val", idx_va), ("test", idx_te)]:
            sim_split = sliced_series(idx_split)
            out_txt = out_dir / f"{wid}_{split_name}.txt"
            write_timeseries_txt(out_txt, dates, y_phys, sim_split)
            plot_ts(dates, y_phys, sim_split, f"S4DFT {wid} ({split_name})", out_dir / f"{wid}_{split_name}.png")

    # --------------- SUMMARY METRICS (PHYSICAL) ---------------
    for split in ("train", "val", "test"):
        rows = []
        for wid in wsids:
            d = load_npz_for_wsid(model_input, wid)
            y_phys_full = d["y"].astype(np.float32)
            idx_tr, idx_va, idx_te = d["idx_tr"], d["idx_va"], d["idx_te"]
            idx_all = np.unique(np.concatenate([idx_tr, idx_va, idx_te])) \
                if (idx_tr.size + idx_va.size + idx_te.size) > 0 else np.arange(len(y_phys_full))
            if idx_all.size == 0:
                continue
            lo, hi = int(idx_all.min()), int(idx_all.max())
            y_phys = y_phys_full[lo:hi + 1]

            import pandas as pd
            csv_all = out_dir / f"{wid}_all.csv"
            if not csv_all.exists():
                continue
            df_all = pd.read_csv(csv_all)
            sim_all = df_all["simulated"].to_numpy()

            idx_split = d[f"idx_{split[:2]}"]
            mask = np.isin(np.arange(lo, hi + 1), idx_split)
            obs_split = y_phys[mask]
            sim_split = sim_all[mask]
            if obs_split.size == 0:
                continue

            m = compute_all_metrics(obs_split, sim_split)
            rows.append((wid, [m["nse"], "kge" in m and m["kge"], "kge_r" in m and m["kge_r"],
                               "fhv_2" in m and m["fhv_2"], "flv" in m and m["flv"], m.get("bias", np.nan)]))

        if rows:
            # sanitize None placeholders to floats
            cleaned = []
            for wid, vals in rows:
                cleaned.append((wid, [
                    float(v) if isinstance(v, (int, float, np.floating)) else (float(v) if v is not None else np.nan)
                    for v in vals]))
            write_stats_table(out_dir / f"stats_{split}.csv",
                              ["nse", "kge", "kge_r", "fhv_2", "flv", "bias"], cleaned)


if __name__ == "__main__":
    main()
