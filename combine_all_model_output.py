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
import glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.utils import load_npz_for_wsid, list_wsids, write_timeseries_txt_3model
from common.plotting import plot_ts

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--state_dir", default="runs/Pennsylvania")
    ap.add_argument("--split", choices=["train","val","test"], default="test")
    ap.add_argument("--seq_len", type=int, default=270)
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    model_input = state_dir / "model_input"
    out_dir = state_dir / "combined_all_model_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    lstm_dir = state_dir / "LSTM output"
    mcr_dir  = state_dir / "MCRLSTM output"
    s4_dir   = state_dir / "S4DFT output"

    wsids = list_wsids(str(model_input))
    for wid in wsids:
        # read original dates/obs
        d = load_npz_for_wsid(model_input, wid)
        dates, y, idx = d["dates"], d["y"], d[f"idx_{args.split[:2]}"]

        def read_sim(model_dir: Path) -> np.ndarray:
            f = model_dir / f"{wid}_{args.split}.txt"
            if not f.exists():
                return np.full_like(y, np.nan, dtype=float)
            df = pd.read_csv(f)
            arr = df["simulated"].to_numpy()
            # align length if needed
            if len(arr) != len(y):  # safety
                arr2 = np.full_like(y, np.nan, dtype=float)
                arr2[:min(len(arr), len(y))] = arr[:min(len(arr), len(y))]
                return arr2
            return arr

        sim_lstm = read_sim(lstm_dir)
        sim_mcr  = read_sim(mcr_dir)
        sim_s4   = read_sim(s4_dir)

        # save combined text
        write_timeseries_txt_3model(out_dir / f"{wid}_{args.split}.txt",
                                    dates, y, sim_lstm, sim_mcr, sim_s4)

        # combined plot
        plt.figure(figsize=(12,4))
        plt.plot(dates, y, label="Observed")
        if np.isfinite(sim_lstm).any(): plt.plot(dates, sim_lstm, label="LSTM")
        if np.isfinite(sim_mcr).any():  plt.plot(dates, sim_mcr,  label="MCRLSTM")
        if np.isfinite(sim_s4).any():   plt.plot(dates, sim_s4,   label="S4DFT")
        plt.xlabel("Date"); plt.ylabel("Streamflow (mm/day)")
        plt.title(f"All models — {wid} ({args.split})"); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"{wid}_{args.split}.png", dpi=200)
        plt.close()

if __name__ == "__main__":
    main()
