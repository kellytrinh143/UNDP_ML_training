# ML training -- Practical sessions


---

## Overview

Thi practical sessions show how to apply machine learning to wheat yield prediction using real multiscale climate and soil data from Western Australia (1989–2021). 

| Session | Notebook | Model | Key Concepts |
|---------|----------|-------|--------------|
| 1 | `Session1_Multiple_regression.ipynb` | Linear Regression | EDA, missing data, correlation, train/test split, RMSE/R², residuals |
| 2 | `Session2_RandomForest.ipynb` | Random Forest | Bagging, decision trees, feature importance, hyperparameter tuning, temporal CV |
| 3 | `Session3_XGBoost.ipynb` | XGBoost | Boosting, early stopping, SHAP explanations, full model comparison |
| 4 | `Session4_NeuralNetworks.ipynb` | MLP + LSTM | Activation functions, backpropagation, sequence modelling, extrapolation |

---

## Data Description

Each row = one **grid cell × one growing year**.

**Growth windows:**
- `W1_estab` — Establishment
- `W2_veg` — Vegetative growth
- `W3_preAnth` — Pre-anthesis (critical for yield)
- `W4_grainFill` — Grain filling
- `W5_matur` — Maturation
--- 

| Column | Unit | Description |
|--------|------|-------------|
| `lat` | °S | Latitude of grid cell |
| `lon` | °E | Longitude of grid cell |
| `pawc_0_30_mm` | mm | Plant Available Water Capacity, 0–30 cm depth |
| `ph_0_30` | — | Depth-weighted mean soil pH, 0–30 cm |
| `minN_0_30` | mg/kg | Mineral nitrogen (NO₃ + NH₄), 0–30 cm |
| `wheat_yield` | t/ha | APSIM-simulated yield (target variable) |
| `year` | — | Calendar year |

---

### Crop-Seasonal & Monthly Feature Variables
The same feature are computed for every temporal slot (monthly, and crop-seasonal)
| Variable | Unit | Description |
|----------|------|-------------|
| `rain_sum` | mm | Total rainfall. Water input for the soil water balance. |
| `rain_days` | days | Number of days with rainfall ≥ 1 mm. Rainfall frequency pattern. |
| `rad_mean` | MJ/m²/d | Mean daily solar radiation. Drives biomass accumulation via RUE. |
| `rad_sum` | MJ/m² | Cumulative radiation. Total energy available for photosynthesis. |
| `tmean` | °C | Mean daily temperature (Tmax + Tmin) / 2 |
| `tmax_max` | °C | Maximum of daily Tmax. Detects heat extremes (>34 °C). |
| `tmin_min` | °C | Minimum of daily Tmin. Detects frost events (<0 °C). |
| `diurnal` | °C | Mean diurnal temperature range (Tmax − Tmin). |
| `heat_days` | days | Count of days with Tmax > 34 °C. Heat damage |
| `frost_days` | days | Count of days with Tmin < 0 °C. Frost damage potential. |
| `vpd_mean` | kPa | Mean vapour pressure deficit.|
| `fasw_mean` | 0–2 | Mean fraction of available soil water: SM / PAWC. |
| `fw_photo` | 0–1 | Mean photosynthesis water stress: min(FASW / 0.5, 1). |
---
| Slot name | Meaning |
|-----------|---------|
| `mpre6` … `mpre1` | 6 calendar months before the sowing month |
| `m0` | Sowing month (only post-sowing days included) |
| `m1` … `m8` | Months 1–8 after sowing month |
---
## Questions by Session

### Session 1 — Data & Linear Regression
1. What is in the dataset? (Q1.1, Q1.2)
2. What drives yield most — rainfall or temperature? (Q2.2, Q2.3)
3. How do you interpret regression coefficients? (Q3.1)
4. What do residual plots reveal? (Q4.1)
5. When does linear regression fail? (Q4.2)

### Session 2 — Random Forest
1. Why do single trees overfit? (Q1.1, Q1.2)
2. How does Random Forest fix the overfitting problem? (concept)
3. How many trees are enough? 
4. What is permutation importance and why is it more reliable? (Q3.1)
5. Temporal vs random validation — which is more realistic? (Q4.1)

### Session 3 — XGBoost
1. What is the difference between bagging and boosting? (Part 1)
2. What does each XGBoost hyperparameter control?
3. How do SHAP values explain individual predictions? 
4. What are the limitations of tree-based models? 

### Session 4 — Neural Networks
1. Why do we need activation functions? 
2. How does the training loop work? 
3. When does LSTM outperform MLP? 
4. How do all models behave when extrapolating? 
5. Which model should you use for which task?

---

## Key ML Concepts Covered

| Concept | Where Covered |
|---------|--------------|
| Train/test split | Session 1 |
| Feature scaling (StandardScaler) | Session 1 |
| Overfitting & generalisation | Sessions 1, 2, 4 |
| Cross-validation | Sessions 1, 2 |
| Bias-variance trade-off | Session 2 |
| Feature importance | Sessions 2, 3 |
| Hyperparameter tuning | Sessions 2, 3 |
| SHAP explanations | Session 3 |
| Early stopping | Sessions 3, 4 |
| Temporal validation | Sessions 2, 3 |
| Extrapolation limitations | Session 4 |
| Uncertainty quantification | Session 4 (exercise) |

---

