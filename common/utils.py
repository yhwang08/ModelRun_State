# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Yihan Wang @ University of Oklahoma
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from __future__ import annotations
import glob, os, json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

def list_wsids(model_input_dir: str) -> List[str]:
    files = glob.glob(os.path.join(model_input_dir, "*.npz"))
    wsids = [Path(f).stem for f in files]
    wsids.sort()
    return wsids

def save_hparams(hp: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "hparams.json", "w", encoding="utf-8") as f:
        json.dump(hp, f, indent=2)

def load_npz_for_wsid(model_input_dir: Path, wsid: str):
    npz = np.load(model_input_dir / f"{wsid}.npz", allow_pickle=True)
    return {
        "dates": np.array([np.datetime64(d) for d in npz["dates"]]),
        "X": npz["X"],
        "y": npz["y"].reshape(-1),
        "idx_tr": npz["idx_tr"].astype(int),
        "idx_va": npz["idx_va"].astype(int),
        "idx_te": npz["idx_te"].astype(int),
        "feature_names": npz["feature_names"],
    }

def reconstruct_daily_from_windows(y_full_len: int, idx_split: np.ndarray, seq_len: int, preds_laststep: Dict[int,float]) -> np.ndarray:
    """
    Reconstruct a per-day series for the split indices:
      - We expect preds_laststep[t] to hold the model's prediction for day t using window ending at t.
      - For days in idx_split that lack a prediction (e.g., not enough history), set to NaN.
    """
    out = np.full(y_full_len, np.nan, dtype=float)
    for t in idx_split:
        if t in preds_laststep:
            out[t] = preds_laststep[t]
    return out

def write_timeseries_txt(path: Path, dates, obs, sim):
    """
    Excel-friendly CSV (txt): first row headers; first col dates (YYYY-MM-DD)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("date,observed,simulated\n")
        for d, o, s in zip(dates, obs, sim):
            f.write(f"{str(d)[:10]},{'' if np.isnan(o) else float(o)},{'' if np.isnan(s) else float(s)}\n")

def write_timeseries_txt_3model(path: Path, dates, obs, sim_lstm, sim_mcr, sim_s4):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("date,observed,LSTM simulated,MCRLSTM simulated,S4DFT simulated\n")
        for i in range(len(dates)):
            d = str(dates[i])[:10]
            def fmt(v): return "" if (v is None or np.isnan(v)) else float(v)
            f.write(f"{d},{fmt(obs[i])},{fmt(sim_lstm[i])},{fmt(sim_mcr[i])},{fmt(sim_s4[i])}\n")

def write_stats_table(path: Path, header_stats: List[str], rows: List[Tuple[str, List[float]]]):
    """
    writes: first row = stat names; first col = watershed ids; rows of values
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("watershed," + ",".join(header_stats) + "\n")
        for wid, vals in rows:
            vals_str = ",".join("" if (v is None or np.isnan(v)) else f"{v:.6f}" for v in vals)
            f.write(f"{wid},{vals_str}\n")
