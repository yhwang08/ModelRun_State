This repository contains the codes for LSTM, MCR-LSTM, and S4D-FT model runs for each U.S. state using CAMELS dataset.

MCR-LSTM is introduced in:
Wang, Y., Zhang, L., Erichson, N. B., & Yang, T. (2025). A Mass Conservation Relaxed (MCR) LSTM Model for Streamflow Simulation across CONUS. Water Resources Research, 61(8), e2024WR039131.
https://doi.org/10.1029/2024WR039131

S4D-FT is presented in:
Wang, Y., Zhang, L., Yu, A., Erichson, N. B., & Yang, T. (2025). A Deep State Space Model for Rainfall–Runoff Simulations. arXiv preprint arXiv:2501.14980.
https://arxiv.org/abs/2501.14980

For regional train across CONUS on GPU, refer to the shared repo specified in the above two papers. 

## How to Run the Code

1. **Prepare state shapefiles**  
   Create a folder named `state_shps` in the project directory.  
   Place the shapefiles for your target state inside a subfolder named `[statename]_shp` (e.g., `kansas_shp`).

2. **Download the CAMELS dataset**  
   Download and extract the CAMELS dataset to a local directory.

3. **Preprocess data**  
   Run the following script to prepare the state-level inputs:
   ```bash
   python common/prepare_from_camels_by_state_geo.py


Integrate results
After training is complete, merge outputs and statistics:

python combine_all_model_output.py
python combine_all_model_stats.py

