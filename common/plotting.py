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
import matplotlib.pyplot as plt
import numpy as np

def plot_ts(dates, obs, sim, title, out_png):
    plt.figure(figsize=(12,4))
    plt.plot(dates, obs, label="Observed")
    plt.plot(dates, sim, label="Simulated", alpha=0.8)
    plt.xlabel("Date"); plt.ylabel("Streamflow (mm/day)")
    plt.title(title); plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
