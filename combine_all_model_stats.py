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
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

def mean_row(df):
    # skip first col (watershed)
    vals = df.iloc[:,1:].apply(pd.to_numeric, errors="coerce")
    return vals.mean(axis=0, skipna=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state_dir", default="runs/Pennsylvania")
    ap.add_argument("--split", choices=["train","val","test"], default="test")
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    out_dir = state_dir / "combined_all_model_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # load per-model stats
    def load_stats(which: str) -> pd.DataFrame | None:
        f = state_dir / f"{which} output" / f"stats_{args.split}.csv"
        return pd.read_csv(f) if f.exists() else None

    lstm_df = load_stats("LSTM")
    mcr_df  = load_stats("MCRLSTM")
    s4_df   = load_stats("S4DFT")

    # compute averages
    rows = []
    cols = None
    if lstm_df is not None:
        m = mean_row(lstm_df)
        cols = list(m.index)
        rows.append(("LSTM", m.values))
    if mcr_df is not None:
        m = mean_row(mcr_df)
        cols = list(m.index)
        rows.append(("MCRLSTM", m.values))
    if s4_df is not None:
        m = mean_row(s4_df)
        cols = list(m.index)
        rows.append(("S4DFT", m.values))

    # write combined matrix: first col=stat names; first row=model names
    # (we'll store a friendly table with models as columns)
    if not rows or cols is None:
        print("No stats found to combine.")
        return

    out = pd.DataFrame(
        {name: vals for (name, vals) in rows},
        index=cols
    )
    # requirement format: (a) first column = statistic names; (b) first row = model names
    # Save as CSV (Excel-friendly text)
    out_path = out_dir / f"avg_stats_{args.split}.csv"
    out.to_csv(out_path, index_label="statistic")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
