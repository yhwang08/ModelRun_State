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
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Dynamic feature count (order must be: PRCP, SRAD, Tmax, Tmin, Vp)
DYN_DIM = 5  # ["PRCP(mm/day)", "SRAD(W/m2)", "Tmax(C)", "Tmin(C)", "Vp(Pa)"]

@dataclass
class Scaler:
    no_norm_precip: bool = True
    dyn_mean: Optional[np.ndarray] = None  # shape [5]
    dyn_std:  Optional[np.ndarray] = None  # shape [5]
    sta_mean: Optional[np.ndarray] = None  # shape [S] or None
    sta_std:  Optional[np.ndarray] = None  # shape [S] or None
    y_mean:   Optional[float] = None
    y_std:    Optional[float] = None

    def apply_X(self, X: np.ndarray) -> np.ndarray:
        """Normalize a [N,F] or [N*L,F] matrix in-place-ish (returns new array)."""
        X = X.astype(np.float32, copy=True)
        if self.dyn_mean is not None and self.dyn_std is not None:
            X[:, :DYN_DIM] = (X[:, :DYN_DIM] - self.dyn_mean[None, :]) / self.dyn_std[None, :]
        if self.sta_mean is not None and self.sta_std is not None and X.shape[1] > DYN_DIM:
            X[:, DYN_DIM:] = (X[:, DYN_DIM:] - self.sta_mean[None, :]) / self.sta_std[None, :]
        return X

    def apply_y(self, y: np.ndarray) -> np.ndarray:
        if self.y_mean is None or self.y_std is None:
            return y.astype(np.float32, copy=True)
        return ((y - self.y_mean) / self.y_std).astype(np.float32)


class SlidingWindows:
    """
    Build (seq_len -> next-step) windows, where each window's target t must be in `idx`.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.idx = np.asarray(idx, dtype=int)
        self.seq_len = int(seq_len)

        targets = []
        if len(self.idx) > 0 and len(X) > 0:
            present = set(range(len(X)))
            for t in self.idx:
                t0 = t - (self.seq_len - 1)
                if t0 >= 0 and all((t - k) in present for k in range(self.seq_len)):
                    targets.append(t)
        self.targets = np.array(targets, dtype=int)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def get(self, j: int) -> Tuple[np.ndarray, float, int]:
        t = int(self.targets[j])
        t0 = t - (self.seq_len - 1)
        return self.X[t0:t+1, :], float(self.y[t]), t


class RegionalDataset(Dataset):
    """
    Regional dataset for a state.

    Modes:
      - split="train": uses idx_union = idx_tr ∪ idx_va for ALL basins.
                       Computes a per-state scaler from UNION rows.
                       (You will random-split this dataset into train/val during training.)
      - split="test" : uses idx_te and REUSES the scaler computed on train.

    RETURNS:
      (x[L,F], y[1], q_std[1])
      where q_std is **per-basin std of physical y** computed from the UNION (idx_tr ∪ idx_va),
      to mirror the good code’s train-period std behavior.
    """
    def __init__(
        self,
        model_input_dir: str,
        wsids: List[str],
        split: str,                       # "train" (== union train∪val) OR "test"
        seq_len: int,
        scaler: Optional[Scaler],         # None for TRAIN to compute; pass TRAIN's scaler for TEST
        compute_state_scaler: bool,       # True only for TRAIN
        normalize_y: bool = False,
        no_norm_precip: bool = True,
    ):
        assert split in ("train", "test")
        self.seq_len = int(seq_len)
        self.normalize_y = bool(normalize_y)
        self.model_input_dir = model_input_dir
        self.wsids = list(wsids)

        xs: List[np.ndarray] = []
        ys: List[float] = []
        sample_wsids: List[str] = []
        metas: List[Tuple[str, int]] = []

        # For computing the state-wide scaler on the TRAIN UNION only
        dyn_rows: List[np.ndarray] = []
        sta_rows: List[np.ndarray] = []
        y_rows:   List[np.ndarray] = []

        feat_dim = None

        # -----------------------
        # Build samples
        # -----------------------
        for wid in self.wsids:
            path = os.path.join(model_input_dir, f"{wid}.npz")
            if not os.path.exists(path):
                continue
            npz = np.load(path, allow_pickle=True)
            X_full = npz["X"]              # [T,F]
            y_full = npz["y"].reshape(-1)  # [T]
            idx_tr = npz["idx_tr"].astype(int)
            idx_va = npz["idx_va"].astype(int)
            idx_te = npz["idx_te"].astype(int)

            feat_dim = feat_dim or int(X_full.shape[1])
            sta_dim = max(0, feat_dim - DYN_DIM)

            if split == "train":
                idx = np.unique(np.concatenate([idx_tr, idx_va], axis=0))  # UNION
            else:  # "test"
                idx = idx_te

            if idx.size == 0:
                continue

            win = SlidingWindows(X_full, y_full, idx, self.seq_len)
            for j in range(len(win)):
                xr, yr, t = win.get(j)
                xs.append(xr.astype(np.float32))
                ys.append(yr)
                sample_wsids.append(wid)
                metas.append((wid, int(t)))

            # Collect stats ONLY when we're building the TRAIN UNION dataset
            if compute_state_scaler and split == "train":
                union_idx = idx  # already tr ∪ va
                dyn_rows.append(X_full[union_idx, :DYN_DIM])
                if sta_dim > 0:
                    sta_rows.append(X_full[union_idx, DYN_DIM:])
                if self.normalize_y:
                    y_rows.append(y_full[union_idx])

        if len(xs) == 0:
            raise RuntimeError(f"No samples for split='{split}'. Check seq_len/wsids/files.")

        self.x = np.stack(xs, axis=0).astype(np.float32)  # [N,L,F]
        self.y = np.asarray(ys, dtype=np.float32)         # [N]
        self._sample_wsids = sample_wsids                 # length N
        self.meta = metas
        self.feat_dim = feat_dim
        self.sta_dim = max(0, feat_dim - DYN_DIM)

        # -----------------------
        # Compute or set scaler
        # -----------------------
        if scaler is None:
            if not compute_state_scaler or split != "train":
                raise RuntimeError("TEST must be given TRAIN's scaler; TRAIN must compute it.")
            if len(dyn_rows) == 0:
                raise RuntimeError("No rows to compute per-state scaler from UNION (train∪val).")

            Xdyn = np.concatenate(dyn_rows, axis=0).astype(np.float64)
            dyn_mean = np.nanmean(Xdyn, axis=0).astype(np.float32)
            dyn_std  = np.nanstd (Xdyn, axis=0).astype(np.float32)
            dyn_std[dyn_std == 0] = 1.0
            if no_norm_precip:
                dyn_mean[0] = 0.0
                dyn_std[0]  = 1.0

            if self.sta_dim > 0 and len(sta_rows) > 0:
                Xsta = np.concatenate(sta_rows, axis=0).astype(np.float64)
                sta_mean = np.nanmean(Xsta, axis=0).astype(np.float32)
                sta_std  = np.nanstd (Xsta, axis=0).astype(np.float32)
                sta_std[sta_std == 0] = 1.0
            else:
                sta_mean = None
                sta_std  = None

            if self.normalize_y and len(y_rows) > 0:
                yy = np.concatenate(y_rows).astype(np.float64)
                y_mean = float(np.nanmean(yy))
                y_std  = float(np.nanstd (yy))
                if y_std == 0.0:
                    y_std = 1.0
            else:
                y_mean = None
                y_std  = None

            self.scaler = Scaler(
                no_norm_precip=no_norm_precip,
                dyn_mean=dyn_mean, dyn_std=dyn_std,
                sta_mean=sta_mean, sta_std=sta_std,
                y_mean=y_mean, y_std=y_std
            )
        else:
            self.scaler = scaler

        # -----------------------
        # Apply normalization
        # -----------------------
        N, L, F = self.x.shape
        X2D = self.x.reshape(N * L, F)
        X2D = self.scaler.apply_X(X2D)
        self.x = X2D.reshape(N, L, F).astype(np.float32)
        if self.normalize_y:
            self.y = self.scaler.apply_y(self.y)

        # -----------------------
        # Per-basin q_std in PHYSICAL space (computed from UNION)
        # -----------------------
        qstd_by_basin_phys: dict[str, float] = {}
        for wid in self.wsids:
            p = os.path.join(model_input_dir, f"{wid}.npz")
            if not os.path.exists(p):
                continue
            npz = np.load(p, allow_pickle=True)
            y_full = npz["y"].reshape(-1).astype(np.float32)
            idx_tr = npz["idx_tr"].astype(int)
            idx_va = npz["idx_va"].astype(int)
            idx_union = np.unique(np.concatenate([idx_tr, idx_va], axis=0))
            if idx_union.size > 0:
                s_phys = float(np.std(y_full[idx_union]))
            else:
                s_phys = float(np.std(y_full))
            qstd_by_basin_phys[wid] = max(s_phys, 1e-8)

        self.qstd = np.array([qstd_by_basin_phys[w] for w in self._sample_wsids], dtype=np.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        x_t = torch.from_numpy(self.x[i])                                # [L,F]
        y_t = torch.tensor(self.y[i], dtype=torch.float32).unsqueeze(0)  # [1]
        q_t = torch.tensor(self.qstd[i], dtype=torch.float32).unsqueeze(0)  # [1]
        return x_t, y_t, q_t
