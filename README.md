# ML Training -- Practical Sessions


---

## Overview

These practical sessions show how to apply machine learning to wheat yield prediction using real multiscale climate and soil data from Western Australia (1989–2020).

| Session | Notebook | Model | Key Concepts |
|---------|----------|-------|--------------|
| 1 | `Session1_Multiple_regression.ipynb` | Linear Regression | EDA, missing data, correlation, train/test split, RMSE/R², residuals |
| 2 | `Session2_RandomForest.ipynb` | Random Forest | Bagging, decision trees, feature importance, hyperparameter tuning, temporal CV |
| 3 | `Session3_XGBoost.ipynb` | XGBoost | Boosting, early stopping, SHAP explanations, full model comparison |
| 4 | `Session4_NeuralNetworks.ipynb` | MLP + LSTM | Activation functions, backpropagation, sequence modelling, extrapolation |

---

## Data Description

Each row = one **grid cell × one growing year** for Western Australia. Each grid cell has a resolution of 5 km × 5 km. Data covers 1989–2020 (32 years).

---

### Static Features

| Column | Unit | Description |
|--------|------|-------------|
| `lat` | °S | Latitude of grid cell |
| `lon` | °E | Longitude of grid cell |
| `year` | — | Calendar year |
| `sowing_doy` | day | Day-of-year of sowing (NaN if not sown) |
| `wheat_yield` | t/ha | wheat yield (**target variable**) |
| `pawc_0_30_mm` | mm | Plant Available Water Capacity, 0–30 cm depth |
| `ph_0_30` | — | Depth-weighted mean soil pH, 0–30 cm |

---

### Seasonal Rainfall Aggregates

| Column | Unit | Description |
|--------|------|-------------|
| `rain_preseason` | mm | Total rainfall Feb–Apr. |
| `rain_early` | mm | Total rainfall May–Aug. Captures pre-flowering rainfall (sowing through vegetative growth). |
| `rain_late` | mm | Total rainfall Sep–Nov. Captures post-flowering rainfall (grain filling period). |

---

### Temperature & Stress Aggregates

| Column | Unit | Description |
|--------|------|-------------|
| `gdd_may_nov` | °Cd | Cumulative growing degree days May–Nov (base 0 °C). Calculated as sum of max((Tmax + Tmin) / 2, 0) over the period. |
| `frost_days_aug_oct` | days | Count of days with Tmin < 2 °C during Aug–Oct (flowering and grain fill window). |
| `heat_days_aug_oct` | days | Count of days with Tmax > 32 °C during Aug–Oct (flowering and grain fill window). |
| `rad_aug_oct` | MJ/m² | Captures light energy available during the critical flowering and grain fill period|
| `fasw_mean_aug_oct` | -| Captures soil water availability during the stress-sensitive reproductive window.
---

### Monthly Rainfall (15 months)

Total rainfall per calendar month, relative to the sowing month.
| Column | Description |
|--------|-------------|
| `rain_mpre6` … `rain_mpre1` | Monthly total rainfall, 6 months before sowing month |
| `rain_m0` | Monthly total rainfall in the sowing month (post-sowing days only) |
| `rain_m1` … `rain_m8` | Monthly total rainfall, 1–8 months after sowing month |

---

### Monthly Post-Sowing Features (9 months each)

These variables are computed for each month from sowing onward (`m0` through `m8`), covering the full crop cycle.

| Variable prefix | Unit | Description |
|-----------------|------|-------------|
| `rad_sum_m0` … `rad_sum_m8` | MJ/m² | Monthly cumulative solar radiation|
| `fasw_mean_m0` … `fasw_mean_m8` | 0–2 | Monthly mean fraction of available soil water (SM / PAWC). Values > 1 indicate above field capacity. |
| `gdd_m0` … `gdd_m8` | °Cd | Monthly growing degree days (base 0 °C). Tracks thermal accumulation through the crop cycle. |

---
  
### Feature Count Summary

| Group | Columns | Count |
|-------|---------|-------|
| Static / identifiers | lat, lon, year, sowing_doy, wheat_yield, pawc_0_30_mm, ph_0_30 | 7 |
| Monthly rain (pre-sow) | rain_mpre6 … rain_mpre1 | 6 |
| Monthly rain (post-sow) | rain_m0 … rain_m8 | 9 |
| Monthly radiation (post-sow) | rad_sum_m0 … rad_sum_m8 | 9 |
| Monthly FASW (post-sow) | fasw_mean_m0 … fasw_mean_m8 | 9 |
| Monthly GDD (post-sow) | gdd_m0 … gdd_m8 | 9 |
| Seasonal aggregates | rain_preseason, rain_early, rain_late, gdd_may_nov, frost_days_aug_oct, heat_days_aug_oct, rad_aug_oct, fasw_mean_aug_oct | 8 |
| **Total** | | **57** |

---

## Questions by Session

### Session 1 — Data & Linear Regression
1. What is in the dataset?
2. What drives yield most — rainfall or temperature?
3. How do you interpret regression coefficients?
4. What do residual plots reveal?
5. When does linear regression fail?

### Session 2 — Random Forest
1. Why do single trees overfit?
2. How does Random Forest fix the overfitting problem?
3. How many trees are enough?
4. What is permutation importance and why is it more reliable?
5. Temporal vs random validation — which is more realistic?

### Session 3 — XGBoost
1. What is the difference between bagging and boosting?
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
