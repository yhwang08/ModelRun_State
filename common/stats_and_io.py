# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Yihan Wang @ University of Oklahoma
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# <StateName>/common/stats_and_io.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _safe(arr): return np.asarray(arr, dtype=float)

def nse(obs, sim):
    obs, sim = _safe(obs), _safe(sim)
    m = np.nanmean(obs)
    den = np.nansum((obs - m) ** 2)
    if den <= 0: return np.nan
    return 1.0 - (np.nansum((obs - sim) ** 2) / den)

def pearson_r(obs, sim):
    obs, sim = _safe(obs), _safe(sim)
    o = obs - np.nanmean(obs); s = sim - np.nanmean(sim)
    den = np.sqrt(np.nansum(o**2) * np.nansum(s**2))
    return (np.nansum(o*s) / den) if den>0 else np.nan

def bias_ratio(obs, sim):
    mu_o = np.nanmean(obs); mu_s = np.nanmean(sim)
    return mu_s / mu_o if mu_o != 0 else np.nan

def var_ratio(obs, sim):
    so = np.nanstd(obs); ss = np.nanstd(sim)
    return ss / so if so != 0 else np.nan

def kge(obs, sim):
    r = pearson_r(obs, sim); beta = bias_ratio(obs, sim); gamma = var_ratio(obs, sim)
    if np.isnan(r) or np.isnan(beta) or np.isnan(gamma): return np.nan
    return 1 - np.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)

def compute_stats(obs, sim):
    m = {
        "NSE": nse(obs, sim),
        "KGE": kge(obs, sim),
        "COR": pearson_r(obs, sim),
    }
    return m

def save_timeseries_csv(dates, obs, sim, out_csv):
    df = pd.DataFrame({"date": pd.to_datetime(dates), "observed": obs, "simulated": sim})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

def plot_series(dates, obs, sim, title, out_png):
    plt.figure(figsize=(12,4))
    plt.plot(dates, obs, label="Observed", linewidth=1.2)
    plt.plot(dates, sim, label="Simulated", linewidth=1.0, alpha=0.9)
    plt.title(title)
    plt.xlabel("Date"); plt.ylabel("Streamflow")
    plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close()
