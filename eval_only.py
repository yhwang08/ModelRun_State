# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Yihan Wang @ University of Oklahoma
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------- repo paths ----------
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))

from common.regional_data import RegionalDataset, SlidingWindows, DYN_DIM
from common.utils import list_wsids, load_npz_for_wsid, write_stats_table
from common.metrics import compute_all_metrics
from common.models import LSTM_Model, MassConservingLSTM_MR


def log(msg: str) -> None:
    print(msg, flush=True)


def load_hparams(out_dir: Path) -> dict:
    hp_path = out_dir / "hparams.json"
    if not hp_path.exists():
        raise FileNotFoundError(f"No hparams.json in {out_dir}")
    with hp_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ckpt(ckpt_dir: Path, which: str):
    if which in ("best", "last"):
        path = ckpt_dir / f"{which}.pt"
    else:
        path = ckpt_dir / (which if which.endswith(".pt") else f"{which}.pt")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def write_timeseries_csv(path: Path, dates, observed: np.ndarray, simulated: np.ndarray):
    """Write time series to CSV with columns: date, observed, simulated."""
    df = pd.DataFrame({
        "date": pd.to_datetime(dates).astype(str),
        "observed": observed.astype(float),
        "simulated": simulated.astype(float)
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state_dir",
                    default=r"C:\Users\yihan\OneDrive - University of Oklahoma\OU\code\runs\Pennsylvania")
    ap.add_argument("--ckpt", default="best", help="'best', 'last', or 'epoch_XXX'")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    state_dir   = Path(args.state_dir)
    model_input = state_dir / "model_input"
    out_dir     = state_dir / "MCRLSTM output"   # matches train.py
    ckpt_dir    = out_dir / "checkpoints"

    # ---- load hparams (to mirror training flags) ----
    hp = load_hparams(out_dir)
    log(f"[hparams] {hp}")
    normalize_y     = bool(hp.get("normalize_y", True))
    no_norm_precip  = bool(hp.get("no_norm_precip", False))
    seq_len         = int(hp.get("seq_len", 365))
    hidden          = int(hp.get("hidden", 256))
    dropout         = float(hp.get("dropout", 0.4))

    # ---- build TRAIN dataset to get scaler (exactly like training) ----
    wsids = list_wsids(str(model_input))
    if not wsids:
        raise RuntimeError(f"No *.npz in {model_input}")

    ds_tr = RegionalDataset(
        str(model_input), wsids, split="train", seq_len=seq_len,
        scaler=None, compute_state_scaler=True,
        normalize_y=normalize_y, no_norm_precip=no_norm_precip
    )
    scaler = ds_tr.scaler

    # TEST dataset uses the same scaler (for consistency)
    ds_te = RegionalDataset(
        str(model_input), wsids, split="test", seq_len=seq_len,
        scaler=scaler, compute_state_scaler=False,
        normalize_y=normalize_y, no_norm_precip=no_norm_precip
    )

    # ---- model + checkpoint (same construction as train.py) ----
    input_dim = ds_tr.x.shape[-1]  # actual feature dim in the prepared files
    model = LSTM_Model(
        input_size_dyn=input_dim,           # mirrors your train.py
        hidden_size=hidden,
        initial_forget_bias=3,
        dropout=dropout,
        concat_static=(input_dim > 5),
        no_static=(input_dim == 5)
    ).to(device)

    model = MassConservingLSTM_MR()

    ckpt = load_ckpt(ckpt_dir, args.ckpt)
    missing = model.load_state_dict(ckpt["state_dict"], strict=False)
    log(f"[load_state_dict] missing/unexpected: {missing}")
    model.eval()

    # ---- evaluate on TEST and write per-basin CSVs + stats_test.csv ----
    rows = []
    for wid in wsids:
        d = load_npz_for_wsid(model_input, wid)
        dates_full   = d["dates"]
        y_phys_full  = d["y"].astype(np.float32)
        idx_tr = d["idx_tr"]; idx_va = d["idx_va"]; idx_te = d["idx_te"]

        if idx_te.size == 0:
            continue

        # Evaluate strictly on TEST span
        idx_all = idx_te
        lo, hi = int(idx_all.min()), int(idx_all.max())
        dates = dates_full[lo:hi+1]
        y_phys = y_phys_full[lo:hi+1]

        # Build windows for TEST indices
        win = SlidingWindows(d["X"], d["y"], idx_all, seq_len)
        sim_all_phys = np.full_like(y_phys, np.nan, dtype=float)

        with torch.no_grad():
            for j in range(len(win)):
                xr_raw, _, t = win.get(j)           # (L, F) raw (unnormalized) features
                xr = xr_raw.astype(np.float32, copy=True)

                # --- normalize inputs exactly as in training ---
                # dynamic features
                xr[:, :DYN_DIM] = (xr[:, :DYN_DIM] - scaler.dyn_mean[None, :]) / scaler.dyn_std[None, :]
                # static features (if present)
                if xr.shape[1] > DYN_DIM and (scaler.sta_mean is not None) and (scaler.sta_std is not None):
                    xr[:, DYN_DIM:] = (xr[:, DYN_DIM:] - scaler.sta_mean[None, :]) / scaler.sta_std[None, :]

                xbt = torch.from_numpy(xr[None, ...]).float().to(device)
                # model returns [B, 1, L] or similar; train script uses [0].view(-1)
                yhat_norm = model(xbt)[0].view(-1)   # last-step target for the window
                yhat_norm = yhat_norm[-1].item()

                # de-normalize y if training used normalized targets
                if normalize_y and (scaler.y_std is not None) and (scaler.y_mean is not None):
                    yhat_phys = yhat_norm * float(scaler.y_std) + float(scaler.y_mean)
                else:
                    yhat_phys = yhat_norm

                # clamp negatives (match "good code")
                if yhat_phys < 0.0:
                    yhat_phys = 0.0

                if lo <= t <= hi:
                    sim_all_phys[t - lo] = float(yhat_phys)

        # mask continuous window down to TEST indices
        mask_test = np.isin(np.arange(lo, hi+1), idx_te)
        obs_te = y_phys[mask_test]
        sim_te = sim_all_phys[mask_test]
        sim_te = np.maximum(sim_te, 0.0)  # safety clamp

        # metrics per basin
        m = compute_all_metrics(obs_te, sim_te)
        rows.append((wid, [m["nse"], m["kge"], m["kge_r"], m["fhv_2"], m["flv"], m["bias"]]))

        # write per-basin CSV for the continuous TEST window
        out_csv = out_dir / f"{wid}_test_evalonly.csv"
        write_timeseries_csv(out_csv, dates, y_phys, np.where(mask_test, sim_all_phys, np.nan))

    # summary table across basins
    write_stats_table(out_dir / "stats_test.csv",
                      ["nse", "kge", "kge_r", "fhv_2", "flv", "bias"], rows)
    log(f"[done] Wrote {len(rows)} rows to {out_dir / 'stats_test.csv'}")


if __name__ == "__main__":
    main()
