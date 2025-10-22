# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Yihan Wang @ University of Oklahoma
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


#Prepare per-state inputs from CAMELS using a stations-with-states CSV.

import argparse
import os
import pdb
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pdb

# -----------------------
# Constants & helpers
# -----------------------

INVALID_ATTR = [
    'gauge_name', 'area_geospa_fabric', 'geol_1st_class', 'glim_1st_class_frac',
    'geol_2nd_class', 'glim_2nd_class_frac', 'dom_land_cover_frac', 'dom_land_cover',
    'high_prec_timing', 'low_prec_timing', 'huc_02', 'q_mean', 'runoff_ratio', 'stream_elas',
    'slope_fdc', 'baseflow_index', 'hfd_mean', 'q5', 'q95', 'high_q_freq', 'high_q_dur',
    'low_q_freq', 'low_q_dur', 'zero_q_freq', 'geol_porostiy', 'root_depth_50',
    'root_depth_99', 'organic_frac', 'water_frac', 'other_frac', 'gauge_lat', 'gauge_lon'
]

DYN_COLS = ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]

# Fixed split dates
TRAIN_START = pd.Timestamp("1999-10-01")
TRAIN_END   = pd.Timestamp("2008-09-30")
TEST_START  = pd.Timestamp("1989-10-01")
TEST_END    = pd.Timestamp("1999-09-30")

def norm_path(p: str) -> Path:
    return Path(p).expanduser().resolve()

def ensure_dirs(base: Path, rels: List[str]):
    for r in rels:
        (base / r).mkdir(parents=True, exist_ok=True)

def read_camels_attributes(camels_root: Path) -> pd.DataFrame:
    """Merge all camels_*.txt tables keyed by 'gauge_id'."""
    for attr_dirname in ["camels_attributes_v2.0", "camels_attributes_v1.2", "camels_attributes_v1.1", "camels_attributes_v1.0"]:
        attr_dir = camels_root / attr_dirname
        if attr_dir.exists():
            break
    else:
        raise RuntimeError(f"Could not find a camels_attributes_* folder under {camels_root}")

    frames = []
    for f in attr_dir.glob("camels_*.txt"):
        try:
            df = pd.read_csv(f, sep=';', header=0, dtype={'gauge_id': str}).set_index('gauge_id')
            frames.append(df)
        except Exception as e:
            print(f"[warn] failed reading {f}: {e}")
    if not frames:
        raise RuntimeError("No attribute tables readable.")
    df_all = pd.concat(frames, axis=1)

    # standardize HUC if present
    if "huc_02" in df_all.columns and "huc" not in df_all.columns:
        df_all["huc"] = df_all["huc_02"].apply(lambda x: str(x).zfill(2))
    return df_all

def load_station_state_csv(stations_states_csv: Path, state_name: str, basin_list: Optional[Path]) -> List[str]:
    """Return list of 8-digit gauge_ids in the specified state (left-padded with zeros)."""
    import pandas as pd
    import re
    import math

    def _norm_id(x) -> Optional[str]:
        if pd.isna(x):
            return None
        s = str(x).strip()

        # 1) Prefer numeric parse to handle '10023000.0', '1.0023E7', etc.
        try:
            v = float(s)
            # if it's effectively an integer, use its integer digits
            if math.isfinite(v) and abs(v - round(v)) < 1e-6:
                s_digits = str(int(round(v)))
            else:
                # fall back to stripping non-digits
                s_digits = re.sub(r'\D', '', s)
        except Exception:
            s_digits = re.sub(r'\D', '', s)

        if not s_digits:
            return None

        # 2) Enforce 8 digits by LEFT-padding (never append on the right)
        if len(s_digits) > 8:
            # Extremely rare; keep the last 8 (CAMELS IDs are 8-digit)
            s_digits = s_digits[-8:]
        return s_digits.zfill(8)

    df = pd.read_csv(stations_states_csv)

    # ensure ID column exists
    if "gauge_id" not in df.columns:
        if "basin" in df.columns:
            df["gauge_id"] = df["basin"]
        else:
            raise RuntimeError("CSV must include 'gauge_id' or 'basin'.")

    if "STATE_NAME" not in df.columns:
        raise RuntimeError("CSV must include 'STATE_NAME'.")

    # normalize IDs
    df["gauge_id"] = df["gauge_id"].map(_norm_id)
    df = df[df["gauge_id"].notna()].copy()

    # filter by state (case-insensitive)
    target = state_name.strip().lower()
    df = df[df["STATE_NAME"].astype(str).str.strip().str.lower() == target].copy()

    # optional intersect with basin_list (normalize those too)
    if basin_list and basin_list.exists():
        ids = [line.strip() for line in basin_list.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = [i for i in (_norm_id(x) for x in ids) if i is not None]
        df = df[df["gauge_id"].isin(set(ids))].copy()

    return sorted(df["gauge_id"].unique().tolist())


def list_forcing_files(forcing_dir: Path) -> List[Path]:
    return list(forcing_dir.glob("**/*_forcing_leap.txt"))

def find_forcing_file(forcing_dir: Path, basin_id: str) -> Path:
    cands = [f for f in list_forcing_files(forcing_dir) if f.name.startswith(basin_id)]
    if not cands:
        raise FileNotFoundError(f"No forcing file for {basin_id} under {forcing_dir}")
    return cands[0]

def read_forcing_and_area(forcing_path: Path) -> Tuple[pd.DataFrame, int]:
    df = pd.read_csv(forcing_path, sep=r"\s+", header=3)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")
    # area on header line ~2 (0-based)
    with open(forcing_path, "r") as fp:
        header = fp.readlines()
    try:
        area_km2 = int(header[2].strip())
    except Exception:
        area_km2 = int(re.sub(r"[^0-9]", "", header[2]))
    return df, area_km2

def find_streamflow_file(camels_root: Path, basin_id: str) -> Path:
    q_dir = camels_root / "usgs_streamflow"
    cands = list(q_dir.glob("**/*_streamflow_qc.txt"))
    for f in cands:
        if f.name.startswith(basin_id):
            return f
    raise FileNotFoundError(f"No streamflow file for {basin_id} under {q_dir}")

def read_qobs_mm_per_day(streamflow_path: Path, area_m2: int) -> pd.Series:
    cols = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    df = pd.read_csv(streamflow_path, sep=r"\s+", header=None, names=cols)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # same conversion as your reference
    q_mm_day = 28.316846592 * df.QObs * 86400.0 / area_m2
    return q_mm_day

def make_splits(dates: pd.DatetimeIndex, val_fraction: float, val_seed: int):
    """
    Build idx_tr, idx_va, idx_te per your spec:
      - TEST:  1989-10-01 .. 1999-09-30
      - TRAIN: 1999-10-01 .. 2008-09-30
      - VAL:   random 10% of TRAIN (disjoint), reproducible
    """
    dates = pd.to_datetime(dates)
    mask_test = (dates >= TEST_START) & (dates <= TEST_END)
    mask_train = (dates >= TRAIN_START) & (dates <= TRAIN_END)

    idx_te = np.where(mask_test)[0]
    idx_pool = np.where(mask_train)[0]

    rng = np.random.default_rng(val_seed)
    n_val = int(round(len(idx_pool) * val_fraction))
    if n_val > 0:
        idx_va = np.sort(rng.choice(idx_pool, size=n_val, replace=False))
    else:
        idx_va = np.array([], dtype=int)
    # train = pool \ val
    val_set = set(idx_va.tolist())
    idx_tr = np.array(sorted([i for i in idx_pool if i not in val_set]), dtype=int)
    return idx_tr, idx_va, idx_te

def prepare_state(camels_root: Path,
                  forcing_name: str,
                  stations_states_csv: Path,
                  state_name: str,
                  out_base: Path,
                  basin_list: Optional[Path],
                  val_seed: int,
                  val_fraction: float = 0.10):

    # Output dirs
    state_dir = out_base / state_name
    ensure_dirs(state_dir, [
        "raw_data/dynamic_features",
        "raw_data/static_features",
        "raw_data/streamflow",
        "model_input"
    ])

    # Attributes (for static features)
    df_attr = read_camels_attributes(camels_root)
    numeric_cols = df_attr.select_dtypes(include=[np.number]).columns.tolist()
    static_keep = [c for c in numeric_cols if c not in INVALID_ATTR]

    # Basin selection by STATE_NAME from CSV
    basins = load_station_state_csv(stations_states_csv, state_name, basin_list)
    if not basins:
        raise RuntimeError(f"No stations found for state '{state_name}' in {stations_states_csv}")

    # Forcing/streamflow paths
    forcing_dir = camels_root / "basin_mean_forcing" / forcing_name
    if not forcing_dir.exists():
        raise RuntimeError(f"Forcing dir not found: {forcing_dir}")
    if not (camels_root / "usgs_streamflow").exists():
        raise RuntimeError(f"USGS streamflow dir not found under {camels_root}")

    prepared = 0
    for wsid in basins:

        try:
            fpath = find_forcing_file(forcing_dir, wsid)
            forc_df, area_m2 = read_forcing_and_area(fpath)
            if any(c not in forc_df.columns for c in DYN_COLS):
                missing = [c for c in DYN_COLS if c not in forc_df.columns]
                raise RuntimeError(f"Forcing missing columns: {missing}")
            qpath = find_streamflow_file(camels_root, wsid)
            q_mm_day = read_qobs_mm_per_day(qpath, area_m2)
            # align & filter
            df = pd.DataFrame(index=forc_df.index.copy())
            for c in DYN_COLS:
                df[c] = forc_df[c].astype(float)
            df["QObs(mm/d)"] = q_mm_day.reindex(df.index).astype(float)
            df = df[(df["QObs(mm/d)"].notna()) & (df["QObs(mm/d)"] >= 0.0)].copy()

            if len(df) < 10:
                print(f"[skip] {wsid}: not enough valid records.")
                continue

            # Save raw CSVs
            dyn = df.reset_index().rename(columns={"index":"date"})
            if "date" not in dyn.columns:
                dyn.rename(columns={dyn.columns[0]: "date"}, inplace=True)
            dyn["date"] = pd.to_datetime(dyn["date"]).dt.date.astype(str)
            (state_dir / "raw_data" / "dynamic_features" / f"{wsid}.csv").parent.mkdir(parents=True, exist_ok=True)
            dyn[["date"] + DYN_COLS].to_csv(state_dir / "raw_data" / "dynamic_features" / f"{wsid}.csv", index=False)

            qdf = df.reset_index()[["index", "QObs(mm/d)"]]
            qdf.rename(columns={"index":"date", "QObs(mm/d)":"streamflow"}, inplace=True)
            qdf["date"] = pd.to_datetime(qdf["date"]).dt.date.astype(str)
            qdf.to_csv(state_dir / "raw_data" / "streamflow" / f"{wsid}.csv", index=False)

            # Static features row
            if wsid in df_attr.index:
                st_row = df_attr.loc[wsid:wsid, static_keep].copy()
            else:
                st_row = pd.DataFrame({"area_m2":[area_m2]})
            st_row.to_csv(state_dir / "raw_data" / "static_features" / f"{wsid}.csv", index=False)

            # model_input
            dates = pd.to_datetime(df.index)
            y = df["QObs(mm/d)"].values.astype(np.float32)
            X_dyn = df[DYN_COLS].values.astype(np.float32)

            sta_vec = st_row.select_dtypes(include=[np.number]).iloc[0].values.astype(np.float32)
            if sta_vec.size:
                X_sta = np.repeat(sta_vec[None, :], len(df), axis=0)
                X = np.concatenate([X_dyn, X_sta], axis=1)
                feat_names = DYN_COLS + [f"static_{i}" for i in range(sta_vec.size)]
            else:
                X = X_dyn
                feat_names = DYN_COLS

            idx_tr, idx_va, idx_te = make_splits(dates, val_fraction=val_fraction, val_seed=val_seed)

            np.savez_compressed(
                state_dir / "model_input" / f"{wsid}.npz",
                dates=dates.astype(str).values,
                X=X, y=y,
                idx_tr=idx_tr, idx_va=idx_va, idx_te=idx_te,
                feature_names=np.array(feat_names, dtype=object)
            )

            prepared += 1
            print(f"[ok] {wsid} → raw_data + model_input")

        except Exception as e:
            print(f"[warn] {wsid}: {e}")

    print(f"\nState: {state_name}  basins prepared: {prepared}  → {state_dir}")
    if prepared == 0:
        print("No basins prepared. Check the CSV/paths and state spelling.")
# -----------------------
# CLI
# -----------------------

ap = argparse.ArgumentParser()
ap.add_argument("--camels_root", default=r"C:\Users\yihan\OneDrive - University of Oklahoma\CAMELS_US\basin_timeseries_v1p2_metForcing_obsFlow\basin_dataset_public_v1p2",
                help="Path to CAMELS 'basin_dataset_public_v1p2' root.")
ap.add_argument("--forcing_name", default="nldas_extended",
                help="Subfolder under basin_mean_forcing (e.g., nldas_extended).")
ap.add_argument("--stations_states_csv", default="stations_with_states.csv",
                help="Path to stations_with_states.csv (must include gauge_id and STATE_NAME).")
ap.add_argument("--state_name",  default="Pennsylvania",
                help="State name used for filtering and output folder.")
ap.add_argument("--out_base", default=r"C:\Users\yihan\OneDrive - University of Oklahoma\OU\code\runs")
ap.add_argument("--basin_list", default=None, help="Optional path to basin_list.txt to restrict stations.")
ap.add_argument("--val_seed", type=int, default=123, help="RNG seed for 10%% validation split within TRAIN.")
args = ap.parse_args()

prepare_state(
    camels_root=norm_path(args.camels_root),
    forcing_name=args.forcing_name,
    stations_states_csv=norm_path(args.stations_states_csv),
    state_name=args.state_name,
    out_base=norm_path(args.out_base),
    basin_list=norm_path(args.basin_list) if args.basin_list else None,
    val_seed=args.val_seed,
)
