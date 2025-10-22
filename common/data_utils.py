# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Yihan Wang @ University of Oklahoma
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# /common/data_utils.py
import os, glob
import numpy as np
import pandas as pd

REQ_DYNAMIC_COLS = None  # if you want to enforce a subset, set list of names

def ensure_dirs(base, subdirs):
    for s in subdirs:
        os.makedirs(os.path.join(base, s), exist_ok=True)

def _read_csv_or_fail(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df

def gather_watersheds(raw_dir):
    dyn_dir = os.path.join(raw_dir, "dynamic_features")
    st_dir  = os.path.join(raw_dir, "static_features")
    q_dir   = os.path.join(raw_dir, "streamflow")
    ids = sorted([os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(q_dir, "*.csv"))])
    # keep only ids that have all three files
    wsids = []
    for wid in ids:
        if (os.path.exists(os.path.join(dyn_dir, f"{wid}.csv")) and
            os.path.exists(os.path.join(st_dir,  f"{wid}.csv")) and
            os.path.exists(os.path.join(q_dir,   f"{wid}.csv"))):
            wsids.append(wid)
    return wsids

def train_val_test_split(dates, train_end=None, val_end=None):
    """Date-based split. Defaults to 70/15/15 if dates are missing/naive."""
    dates = pd.to_datetime(dates)
    n = len(dates)
    if n < 10 or train_end is None or val_end is None:
        i_tr = int(0.7*n); i_va = int(0.85*n)
        idx_tr = np.arange(0, i_tr)
        idx_va = np.arange(i_tr, i_va)
        idx_te = np.arange(i_va, n)
        return idx_tr, idx_va, idx_te
    mask_tr = dates <= pd.to_datetime(train_end)
    mask_va = (dates > pd.to_datetime(train_end)) & (dates <= pd.to_datetime(val_end))
    mask_te = dates > pd.to_datetime(val_end)
    return np.where(mask_tr)[0], np.where(mask_va)[0], np.where(mask_te)[0]
