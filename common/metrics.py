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
import numpy as np

def nse(y_obs: np.ndarray, y_sim: np.ndarray) -> float:
    y_obs = np.asarray(y_obs); y_sim = np.asarray(y_sim)
    denom = np.sum((y_obs - np.mean(y_obs))**2)
    if denom <= 0: return np.nan
    return 1.0 - np.sum((y_obs - y_sim)**2) / denom

def kge(y_obs: np.ndarray, y_sim: np.ndarray) -> tuple[float,float]:
    """Return (KGE, r). Kling-Gupta Efficiency with Pearson r."""
    y_obs = np.asarray(y_obs); y_sim = np.asarray(y_sim)
    if len(y_obs) < 2: return np.nan, np.nan
    r = np.corrcoef(y_obs, y_sim)[0,1]
    alpha = np.std(y_sim) / (np.std(y_obs) + 1e-12)
    beta  = (np.mean(y_sim) + 1e-12) / (np.mean(y_obs) + 1e-12)
    kge_val = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
    return float(kge_val), float(r)

def pct_bias(y_obs: np.ndarray, y_sim: np.ndarray) -> float:
    m = np.mean(y_obs)
    if abs(m) < 1e-12: return np.nan
    return 100.0 * (np.mean(y_sim) - m) / m

def fhv_2(y_obs: np.ndarray, y_sim: np.ndarray) -> float:
    """
    High-flow bias on the top 2% flows (approximate).
    """
    if len(y_obs) == 0: return np.nan
    thr = np.quantile(y_obs, 0.98)
    mask = y_obs >= thr
    if not np.any(mask): return np.nan
    obs_hv = np.sum(y_obs[mask])
    if obs_hv <= 0: return np.nan
    sim_hv = np.sum(y_sim[mask])
    return 100.0 * (sim_hv - obs_hv) / obs_hv

def flv(y_obs: np.ndarray, y_sim: np.ndarray) -> float:
    """
    Low-flow bias on bottom 30% flows (approximate).
    """
    if len(y_obs) == 0: return np.nan
    thr = np.quantile(y_obs, 0.30)
    mask = y_obs <= thr
    obs_lv = np.sum(y_obs[mask])
    if abs(obs_lv) < 1e-12: return np.nan
    sim_lv = np.sum(y_sim[mask])
    return 100.0 * (sim_lv - obs_lv) / (obs_lv + 1e-12)

def compute_all_metrics(y_obs: np.ndarray, y_sim: np.ndarray) -> dict:
    k, r = kge(y_obs, y_sim)
    return {
        "nse": nse(y_obs, y_sim),
        "kge": k,
        "kge_r": r,
        "fhv_2": fhv_2(y_obs, y_sim),
        "flv": flv(y_obs, y_sim),
        "bias": pct_bias(y_obs, y_sim),
    }
