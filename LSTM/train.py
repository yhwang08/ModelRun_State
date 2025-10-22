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

# add ../common to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
from common.regional_data import RegionalDataset, SlidingWindows, DYN_DIM
from common.utils import (
    list_wsids, save_hparams, load_npz_for_wsid,
    write_timeseries_txt, write_stats_table
)
from common.plotting import plot_ts
from common.metrics import compute_all_metrics
from common.models import LSTM_Model

def log(msg): print(msg, flush=True)

def format_hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def save_ckpt(path: Path, epoch: int, model: nn.Module, opt: optim.Optimizer, va_loss: float, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": opt.state_dict(),
        "val_loss": float(va_loss),
    }
    if extra: payload.update(extra)
    torch.save(payload, path)

# ---------- NSELoss (weighted by q_std in PHYSICAL units) ----------
class NSELoss(torch.nn.Module):
    """Calculate (batch-wise) NSE Loss with 1/(std+eps)^2 weights."""
    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor):
        se = (y_pred - y_true)**2
        if q_stds.dim() == 2:
            q_stds = q_stds.view(-1)
        weights = 1 / (q_stds + self.eps)**2
        return torch.mean(weights * se)

def make_loader(ds, batch, shuffle, num_workers):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

@torch.no_grad()
def eval_epoch(model, loader, device, nse_loss):
    model.eval()
    total = 0.0
    n = 0
    for xb, yb, qstd in loader:
        xb = xb.to(device)
        yb = yb.to(device).view(-1)
        qstd = qstd.to(device).view(-1)
        y = model(xb)[0].view(-1)
        loss  = nse_loss(y, yb, qstd)
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
    ap.add_argument("--state_dir", default=r"C:\Users\yihan\OneDrive - University of Oklahoma\OU\code\runs\Pennsylvania")
    ap.add_argument("--seq_len", type=int, default=365)

    # Strong defaults to match the “good” baseline capacity
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)  # initial LR; manual schedule below
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.4)

    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--normalize_y", action="store_true", default=True,
                    help="Normalize targets using TRAIN-union stats (per state).")
    ap.add_argument("--no_norm_precip", action="store_true", default=False,
                    help="If True, DO NOT normalize precipitation.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=500, help="Print every N batches.")
    ap.add_argument("--seed", type=int, default=200)
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)
    state_dir   = Path(args.state_dir)
    model_input = state_dir / "model_input"
    out_dir     = state_dir / "LSTM output"
    ckpt_dir    = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    wsids = list_wsids(str(model_input))
    if not wsids:
        raise RuntimeError(f"No *.npz in {model_input}")

    # -----------------------------
    # DATASETS (no explicit 'val' in dataset layer)
    #   - ds_union: split='train' => idx_tr ∪ idx_va, computes per-state scaler on UNION
    #   - random_split ds_union into train/val (90/10) deterministically
    #   - ds_te:     split='test'  => idx_te, reuses ds_union.scaler
    # -----------------------------
    ds_union = RegionalDataset(
        str(model_input), wsids, split="train", seq_len=args.seq_len,
        scaler=None, compute_state_scaler=True,
        normalize_y=args.normalize_y, no_norm_precip=args.no_norm_precip
    )
    scaler = ds_union.scaler

    g = torch.Generator().manual_seed(args.seed)
    n_val = int(round(0.10 * len(ds_union)))
    n_tr  = len(ds_union) - n_val
    ds_tr, ds_va = random_split(ds_union, [n_tr, n_val], generator=g)

    ds_te = RegionalDataset(
        str(model_input), wsids, split="test", seq_len=args.seq_len,
        scaler=scaler, compute_state_scaler=False,
        normalize_y=args.normalize_y, no_norm_precip=args.no_norm_precip
    )

    dl_tr = make_loader(ds_tr, args.batch, True,  args.num_workers)
    dl_va = make_loader(ds_va, args.batch, False, args.num_workers)
    dl_te = make_loader(ds_te, args.batch, False, args.num_workers)

    log(f"[info] wsids: {len(wsids)} -> {wsids[:5]}{'...' if len(wsids)>5 else ''}")
    log(f"[info] samples: union={len(ds_union)}  train={len(ds_tr)}  val={len(ds_va)}  test={len(ds_te)}")
    log(f"[info] batches: train={len(dl_tr)}  val={len(dl_va)}  test={len(dl_te)}")

    input_dim = ds_union.x.shape[-1]
    print(input_dim)
    model = LSTM_Model(
        input_size_dyn=input_dim,
        hidden_size=args.hidden,
        initial_forget_bias=3,   # hydrology default (2–5)
        dropout=args.dropout,
        concat_static=(input_dim > 5),
        no_static=(input_dim == 5)
    ).to(device)

    nse_loss = NSELoss(eps=0.1)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    save_hparams({
        "seed": args.seed,
        "model": "LSTM",
        "input_dim": input_dim,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "seq_len": args.seq_len,
        "batch": args.batch,
        "epochs": args.epochs,
        "lr_schedule": "1e-3 (1..10), 5e-4 (11..25), 1e-4 (26..)",
        "normalize_y": args.normalize_y,
        "inputs_normalized": True,
        "no_norm_precip": args.no_norm_precip,
        "loss": "NSELoss(eps=0.1) with q_std in physical space",
        "scaler": "per-state on (idx_tr ∪ idx_va) UNION",
        "val_split": "deterministic 90/10 random_split on union",
        "test": "idx_te with same scaler"
    }, out_dir)

    # ----------------- TRAIN -----------------
    best_va = float("inf")
    best_state = None
    start_all = time.time()
    lr = args.lr

    for ep in range(1, args.epochs + 1):
        # manual LR schedule to mirror the good code
        if ep == 11:
            lr = 5e-4
            for pg in opt.param_groups: pg['lr'] = lr
        if ep == 26:
            lr = 1e-4
            for pg in opt.param_groups: pg['lr'] = lr

        model.train()
        ep_start = time.time()
        total_loss = 0.0
        total_count = 0

        log(f"\n=== Epoch {ep}/{args.epochs} (lr={lr:.1e}) ===")
        for bi, (xb, yb, qstd) in enumerate(dl_tr, 1):
            xb = xb.to(device)
            yb = yb.to(device).view(-1)
            qstd = qstd.to(device).view(-1)

            opt.zero_grad()
            y_hat = model(xb)[0].view(-1)
            loss  = nse_loss(y_hat, yb, qstd)
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            opt.step()

            total_loss  += loss.item() * xb.size(0)
            total_count += xb.size(0)

            if (bi % args.log_every) == 0 or bi == len(dl_tr):
                elapsed = time.time() - ep_start
                avg_train = total_loss / max(total_count, 1)
                log(f"[train] epoch {ep} batch {bi}/{len(dl_tr)} "
                    f"avg_loss={avg_train:.4f} lr={opt.param_groups[0]['lr']:.2e} elapsed={format_hms(elapsed)}")

        va_loss = eval_epoch(model, dl_va, device, nse_loss)
        ep_time = time.time() - ep_start
        avg_train = total_loss / max(total_count, 1)
        log(f"[summary] Epoch {ep:03d} | train {avg_train:.4f} | val {va_loss:.4f} | time {format_hms(ep_time)}")

        # --- save per-epoch checkpoint ---
        save_ckpt(ckpt_dir / f"epoch_{ep:03d}.pt", ep, model, opt, va_loss, extra={"train_loss": avg_train, "seed": args.seed})

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
    # Normalize inputs with the TRAIN scaler before inference.
    model.eval()
    for wid in wsids:
        d = load_npz_for_wsid(model_input, wid)
        dates_full   = d["dates"]
        y_phys_full  = d["y"].astype(np.float32)

        idx_tr = d["idx_tr"]; idx_va = d["idx_va"]; idx_te = d["idx_te"]
        if (idx_tr.size + idx_va.size + idx_te.size) == 0:
            continue
        idx_all = np.unique(np.concatenate([idx_tr, idx_va, idx_te]))
        lo, hi = int(idx_all.min()), int(idx_all.max())

        dates = dates_full[lo:hi+1]
        y_phys = y_phys_full[lo:hi+1]

        # windows over ALL indices within the modeling span that are valid targets
        win = SlidingWindows(d["X"], d["y"], idx_all, args.seq_len)

        preds_phys = {}
        with torch.no_grad():
            for j in range(len(win)):
                xr_raw, _, t = win.get(j)  # xr_raw is UNnormalized raw features
                # --- normalize with train scaler ---
                xr = xr_raw.copy().astype(np.float32)
                # dynamic part
                xr[:, :DYN_DIM] = (xr[:, :DYN_DIM]
                                   - scaler.dyn_mean[None, :]) / scaler.dyn_std[None, :]
                # static part (if any)
                if xr.shape[1] > DYN_DIM and (scaler.sta_mean is not None) and (scaler.sta_std is not None):
                    xr[:, DYN_DIM:] = (xr[:, DYN_DIM:]
                                       - scaler.sta_mean[None, :]) / scaler.sta_std[None, :]

                xbt = torch.from_numpy(xr[None, ...]).float().to(device)
                yhat_norm = model(xbt)[0].view(-1)

                # invert normalization of y if model trained in normalized space
                if args.normalize_y and (scaler.y_std is not None) and (scaler.y_mean is not None):
                    yhat_phys = yhat_norm * float(scaler.y_std) + float(scaler.y_mean)
                else:
                    yhat_phys = yhat_norm
                preds_phys[int(t)] = float(yhat_phys)

        # continuous series over [lo..hi]
        sim_all = np.full_like(y_phys, np.nan, dtype=float)
        for tt, val in preds_phys.items():
            if lo <= tt <= hi:
                sim_all[tt - lo] = val

        # write ONE csv and plot for the continuous window
        out_txt_all = out_dir / f"{wid}_all.csv"
        write_timeseries_txt(out_txt_all, dates, y_phys, sim_all)
        plot_ts(dates, y_phys, sim_all, f"LSTM {wid} (train+val+test)", out_dir / f"{wid}_all.png")

        # per-split (truncated to [lo..hi]) — still based on original CAMELS idx for reporting
        def sliced_series(idx_split):
            mask = np.isin(np.arange(lo, hi+1), idx_split)
            arr = np.full_like(y_phys, np.nan, dtype=float)
            arr[mask] = sim_all[mask]
            return arr

        for split_name, idx_split in [("train", idx_tr), ("val", idx_va), ("test", idx_te)]:
            sim_split = sliced_series(idx_split)
            out_txt = out_dir / f"{wid}_{split_name}.txt"
            write_timeseries_txt(out_txt, dates, y_phys, sim_split)
            plot_ts(dates, y_phys, sim_split, f"LSTM {wid} ({split_name})", out_dir / f"{wid}_{split_name}.png")

    # --------------- SUMMARY METRICS ---------------
    # For reporting, we still slice by original CAMELS splits.
    for split in ("train", "val", "test"):
        rows = []
        for wid in wsids:
            d = load_npz_for_wsid(model_input, wid)
            dates_full   = d["dates"]
            y_phys_full  = d["y"].astype(np.float32)
            idx_split = d[f"idx_{split[:2]}"]

            idx_all = np.unique(np.concatenate([d["idx_tr"], d["idx_va"], d["idx_te"]])) \
                      if (d["idx_tr"].size + d["idx_va"].size + d["idx_te"].size) > 0 else np.arange(len(y_phys_full))
            if idx_all.size == 0:
                continue
            lo, hi = int(idx_all.min()), int(idx_all.max())
            y_phys = y_phys_full[lo:hi+1]

            import pandas as pd
            csv_all = out_dir / f"{wid}_all.csv"
            if not csv_all.exists():
                continue
            df_all = pd.read_csv(csv_all)
            sim_all = df_all["simulated"].to_numpy()

            mask = np.isin(np.arange(lo, hi+1), idx_split)
            obs_split = y_phys[mask]
            sim_split = sim_all[mask]
            if obs_split.size == 0:
                continue

            m = compute_all_metrics(obs_split, sim_split)
            rows.append((wid, [m["nse"], m["kge"], m["kge_r"], m["fhv_2"], m["flv"], m["bias"]]))

        if rows:
            write_stats_table(out_dir / f"stats_{split}.csv",
                              ["nse", "kge", "kge_r", "fhv_2", "flv", "bias"], rows)

if __name__ == "__main__":
    main()
