"""
ML Wheat Yield — Multi-Scale Features + Spatial Clustering + GPU
=================================================================
Key design decisions:
  - FEATURE SELECTION: Choose temporal scale via FEATURE_SCALE env var:
      "weekly"  → ~100-150 curated weekly features
      "monthly" → ~100-150 curated monthly features
      "crop"    → ~80-100 crop-seasonal features
      "all"     → all scales combined (~1000+)
  - SPATIAL: lat/lon replaced by K-means clusters (forecast-ready)
  - SHAP: Group importance NORMALISED by group size (mean per feature)
  - GPU: XGBoost 3.x compatible (device="cuda")

Output: ml_{scale}_results/

All yields in t/ha.
"""
from pathlib import Path
import os, json, time
import numpy as np
import pandas as pd
import matplotlib  
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data"))
MS_DIR    = DATA_ROOT / "processed_multiscale"
OUT_ROOT  = Path(os.environ.get("OUT_ROOT", "/output"))

START_YEAR      = int(os.environ.get("START_YEAR", 1989))
END_YEAR        = int(os.environ.get("END_YEAR", 2021))
FIRST_TEST_YEAR = 2017
SHAP_MAX_SAMPLES = int(os.environ.get("SHAP_MAX_SAMPLES", 10000))
SEED = 42
YIELD_COL = "wheat_yield"
N_CLUSTERS = int(os.environ.get("N_CLUSTERS", 25))

# Yield target mode:
#   "raw"    → train on observed yield directly (t/ha)
#   "anomaly"→ train on (yield - cell_mean_yield), predictions converted back
YIELD_MODE = os.environ.get("YIELD_MODE", "raw")
INCLUDE_CROP_EXTRAS = os.environ.get("INCLUDE_CROP_EXTRAS", "yes").lower() in ("yes","true","1")
 
# Run tag: labels the output folder. Set via env var to describe your experiment.
# Examples: "baseline", "no_extras", "weekly_rain_only", "test_v2"
# If empty, auto-generates from yield mode + extras setting.
RUN_TAG = os.environ.get("RUN_TAG", "")
 
if RUN_TAG:
    OUT_DIR = OUT_ROOT / f"ml_multiscale_{RUN_TAG}"
else:
    tag = f"ml_multiscale_results_{YIELD_MODE}"
    if not INCLUDE_CROP_EXTRAS: tag += "_noextras"
    OUT_DIR = OUT_ROOT / f"ml_{tag}"
 
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)
SHAP_DIR = OUT_DIR / "shap_per_fold"; SHAP_DIR.mkdir(exist_ok=True)
 
print(f"[CONFIG] {START_YEAR}-{END_YEAR} | Clusters: {N_CLUSTERS} | Yield: {YIELD_MODE}")
print(f"[CONFIG] Crop extras: {INCLUDE_CROP_EXTRAS} | Output: {OUT_DIR}")
 
# Feature scale selection: any combination separated by "+"
#   "weekly"              → weekly features only (~170)
#   "monthly"             → monthly features only (~150)
#   "crop"                → crop-seasonal features only (~106)
#   "monthly+crop"        → monthly + crop-seasonal (~250)
#   "weekly+crop"         → weekly + crop-seasonal (~270)
#   "weekly+monthly"      → weekly + monthly (~320)
#   "weekly+monthly+crop" → all three scales (~450)
#   "all"                 → same as "weekly+monthly+crop"
FEATURE_SCALE = os.environ.get("FEATURE_SCALE", "monthly+crop")

# ── GPU detection ──
def get_gpu_params():
    import xgboost as xgb
    ver = tuple(int(x) for x in xgb.__version__.split(".")[:2])
    has_gpu = False
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        if r.returncode == 0: has_gpu = True
    except: pass
    if has_gpu:
        if ver >= (2, 0):
            print(f"[GPU] XGBoost {xgb.__version__} device=cuda")
            return {"tree_method": "hist", "device": "cuda"}
        else:
            return {"tree_method": "gpu_hist", "gpu_id": 0}
    print(f"[CPU] XGBoost {xgb.__version__}")
    return {"tree_method": "hist"}

gpu_params = get_gpu_params()

# ── Metrics ──
def calc_lccc(o, p):
    if len(o)<2: return np.nan
    mo,mp=np.mean(o),np.mean(p); vo,vp=np.var(o,ddof=0),np.var(p,ddof=0)
    return 2*np.mean((o-mo)*(p-mp))/(vo+vp+(mo-mp)**2) if (vo+vp+(mo-mp)**2)>0 else np.nan

def calc_metrics(o, p):
    n=len(o)
    if n<2: return {}
    ssr=np.sum((o-p)**2); sst=np.sum((o-np.mean(o))**2); rmse=np.sqrt(np.mean((o-p)**2))
    return {"n":n,"R2":1-ssr/sst if sst>0 else np.nan,"RMSE_tha":rmse,
            "MAE_tha":np.mean(np.abs(o-p)),"Bias_tha":np.mean(p-o),
            "NSE":1-ssr/sst if sst>0 else np.nan,
            "NRMSE_pct":rmse/np.mean(o)*100 if np.mean(o)>0 else np.nan,
            "LCCC":calc_lccc(o,p)}

# ══════════════════════════════════════════════════════════════════════
# FEATURE SELECTION — PER-VARIABLE TEMPORAL SCALE + PRE/POST CONTROL
# ══════════════════════════════════════════════════════════════════════
#
# EDIT THIS DICTIONARY to control which temporal scale each variable uses.
# Each variable maps to a list of scale descriptors:
#
#   "weekly"       → all weekly slots (pre-sow + post-sow)
#   "weekly_pre"   → weekly pre-sow slots only (wpre4..wpre1)
#   "weekly_post"  → weekly post-sow slots only (w0..w34)
#   "monthly"      → all monthly slots (pre-sow + post-sow)
#   "monthly_pre"  → monthly pre-sow only (mpre3..mpre1)
#   "monthly_post" → monthly post-sow only (m0..m8)
#   "crop"         → crop-seasonal windows (W1..W5 + crop total)
#   []             → excluded
#
# Example rationale:
#   - fasw_mean: "weekly_pre" (pre-sow stored water matters at weekly scale)
#               + "weekly_post" (soil moisture changes rapidly during crop)
#               = same as "weekly" but written explicitly
#   - rain_sum: "weekly_post" (wet/dry spells during crop matter weekly)
#             + "monthly_pre" (pre-season cumulative rainfall at monthly is enough)
#   - rad_sum: "monthly_post" only (radiation before sowing is irrelevant)
#   - heat_days: "monthly_post" (heat waves only matter during crop growth)

VAR_SCALE_MAP = {
    # variable_name:  list of scale descriptors
    "rain_sum":       ["weekly_post", "weekly_pre"],    # weekly during crop, monthly pre-season
    "rain_days":      ["monthly_post"],                  # rain frequency during crop
    "rad_sum":        ["monthly_post"],                  # radiation only during crop
    "rad_mean":       ["crop"],                          # mean radiation per growth stage
    "tmean":          ["crop"],                          # temperature at crop-seasonal
    "tmax_max":       ["monthly_post"],                  # monthly heat extremes during crop
    "tmin_min":       ["monthly_post"],                  # monthly frost extremes during crop
    "diurnal":        ["crop"],                          # diurnal range per growth stage
    "tt_sum":         ["crop"],                          # thermal time defines the stages
    "fT_photo":       ["crop"],                          # temp stress is stage-dependent
    "heat_days":      ["monthly_post"],                  # heat waves only matter during crop
    "frost_days":     ["monthly_post"],                  # frost only matters during crop
    "vpd_mean":       ["monthly_post"],                  # atmospheric demand during crop
    "fasw_mean":      ["weekly_pre", "weekly_post"],     # soil water: pre-sow recharge + in-crop
    "fw_photo":       ["weekly_post", "crop"],           # short-term drought + stage-level impact
    "n_days":         [],                                # excluded
}

# ── Slot definitions ──

# Weekly: pre-sow and post-sow slots (edit to add/remove weeks)
WEEKLY_PRE_SLOTS  = ["wpre24", "wpre23", "wpre22", "wpre21",
                     "wpre20", "wpre19", "wpre18", "wpre17",
                     "wpre16", "wpre15", "wpre14", "wpre13",
                     "wpre12", "wpre11", "wpre10", "wpre9",
                     "wpre8", "wpre7", "wpre6", "wpre5",
                     "wpre4", "wpre3", "wpre2", "wpre1"]
WEEKLY_POST_SLOTS = ["w0","w1","w2","w3","w4","w5","w6","w7",
                     "w8","w9","w10","w11","w12","w13","w14","w15","w16","w17",
                     "w18","w19","w20","w21","w22","w23","w24","w25","w26","w27"
                      "w28","w29","w30","w31","w32","w33","w34"]
WEEKLY_SLOTS = WEEKLY_PRE_SLOTS + WEEKLY_POST_SLOTS  # combined for reference

# Monthly: pre-sow and post-sow slots
MONTHLY_PRE_SLOTS  = [f"mpre{i}" for i in range(6, 0, -1)]   # mpre3, mpre2, mpre1
MONTHLY_POST_SLOTS = [f"m{i}" for i in range(9)]              # m0..m8
MONTHLY_SLOTS = MONTHLY_PRE_SLOTS + MONTHLY_POST_SLOTS

# Crop-seasonal windows
CROP_WINDOWS = ["W1_estab","W2_veg","W3_preAnth","W4_grainFill","W5_matur","crop"]

# Extra crop-seasonal features always included
CROP_EXTRAS = [
    "fw_expan_mean_W2_veg","fw_expan_mean_W3_preAnth",
    "fw_photo_min_W2_veg","fw_photo_min_W3_preAnth","fw_photo_min_W4_grainFill",
    "drought_severe_W3_preAnth","drought_severe_W4_grainFill",
    "drought_moderate_W3_preAnth","drought_moderate_W4_grainFill",
    "cum_water_stress_crop","cum_water_stress_W3_preAnth","cum_water_stress_W4_grainFill",
    "FASW_at_sowing","FASW_W4_drying_rate",
    "vern_sum_W2_veg","fD_mean_W2_veg",
    "hg_mean_W4","heat_sen_W4","VPD_mean_crop","TE_mean_crop","TT_cum_crop",
]

# Scale descriptor → slot list mapping
SCALE_SLOT_MAP = {
    "weekly":       WEEKLY_SLOTS,
    "weekly_pre":   WEEKLY_PRE_SLOTS,
    "weekly_post":  WEEKLY_POST_SLOTS,
    "monthly":      MONTHLY_SLOTS,
    "monthly_pre":  MONTHLY_PRE_SLOTS,
    "monthly_post": MONTHLY_POST_SLOTS,
    "crop":         CROP_WINDOWS,
}


def select_features(all_cols):
    """Select features based on the per-variable VAR_SCALE_MAP configuration.
    Each variable can use different scales and pre/post-sow subsets."""
    selected = []

    # Always include static soil features
    for c in ["sowing_doy","pawc_0_30_mm","pawc_0_60_mm","ph_0_30","minN_0_30","profile_depth_cm"]:
        if c in all_cols: selected.append(c)

    # Per-variable, per-scale selection
    for var, scales in VAR_SCALE_MAP.items():
        for scale in scales:
            slots = SCALE_SLOT_MAP.get(scale, [])
            for slot in slots:
                c = f"{var}_{slot}"
                if c in all_cols: selected.append(c)

    # Extra crop-seasonal features
    for e in CROP_EXTRAS:
        if e in all_cols: selected.append(e)

    # Remove duplicates, preserve order
    seen = set(); unique = []
    for s in selected:
        if s not in seen: seen.add(s); unique.append(s)

    return unique

# ══════════════════════════════════════════════════════════════════════
# FEATURE GROUP DEFINITIONS (for SHAP analysis)
# ══════════════════════════════════════════════════════════════════════

def build_feature_groups(feature_cols):
    """Build feature groups for normalised SHAP analysis."""
    groups = {}
    groups["Rainfall"] = [c for c in feature_cols if c.startswith("rain_")]
    groups["Radiation"] = [c for c in feature_cols if c.startswith("rad_")]
    groups["Temperature"] = [c for c in feature_cols if any(c.startswith(p) for p in ["tmean_","tmax_","tmin_","diurnal_"])]
    groups["Thermal Time"] = [c for c in feature_cols if c.startswith("tt_sum")]
    groups["Temp Response (fT)"] = [c for c in feature_cols if c.startswith("fT_photo")]
    groups["Heat Stress"] = [c for c in feature_cols if c.startswith("heat_")]
    groups["Frost Stress"] = [c for c in feature_cols if c.startswith("frost_")]
    groups["VPD"] = [c for c in feature_cols if c.startswith("vpd_") or c.startswith("VPD_") or c.startswith("TE_")]
    groups["Water Stress (FASW)"] = [c for c in feature_cols if c.startswith("fasw_") or c.startswith("FASW")]
    groups["Water Stress (fw)"] = [c for c in feature_cols if c.startswith("fw_")]
    groups["Drought Events"] = [c for c in feature_cols if c.startswith("drought_") or c.startswith("cum_water")]
    groups["Soil Properties"] = [c for c in feature_cols if any(c.startswith(p) for p in ["pawc_","ph_","minN_","profile_"])]
    groups["Sowing & Phenology"] = [c for c in feature_cols if any(c.startswith(p) for p in ["sowing","vern_","fD_","TT_cum","hg_","n_days"])]
    groups["Spatial Clusters"] = [c for c in feature_cols if c.startswith("cluster_")]
    # Remove empty groups
    return {k:v for k,v in groups.items() if v}

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading multiscale parquets ...")
frames = []
for yr in range(START_YEAR, END_YEAR+1):
    p = MS_DIR / f"wheat_multiscale_features_{yr}.parquet"
    if not p.exists(): print(f"  [WARN] Missing {p.name}"); continue
    frames.append(pd.read_parquet(p))
    print(f"  {yr}: {len(frames[-1]):,}")
df_all = pd.concat(frames, ignore_index=True)
print(f"  Total: {len(df_all):,} x {len(df_all.columns)}")

# ── Select features by scale ──
all_cols = [c for c in df_all.columns if c not in
            {"point","latitude","longitude","lat","lon","year","wheat_yield",
             "no_sow","dominant_soil_id","soil_id"}]
feature_cols = select_features(all_cols)

# Print summary
n_w_pre = len([c for c in feature_cols if any(f"_{s}" in c for s in WEEKLY_PRE_SLOTS)])
n_w_post = len([c for c in feature_cols if any(f"_{s}" in c for s in WEEKLY_POST_SLOTS)])
n_m_pre = len([c for c in feature_cols if any(f"_{s}" in c for s in MONTHLY_PRE_SLOTS)])
n_m_post = len([c for c in feature_cols if any(f"_{s}" in c for s in MONTHLY_POST_SLOTS)])
n_crop = len([c for c in feature_cols if any(f"_{s}" in c for s in CROP_WINDOWS)])
print(f"  Selected: {len(feature_cols)} features")
print(f"    Weekly pre-sow:  {n_w_pre}   Weekly post-sow:  {n_w_post}")
print(f"    Monthly pre-sow: {n_m_pre}   Monthly post-sow: {n_m_post}")
print(f"    Crop-seasonal:   {n_crop}")
print(f"  Variable → scale mapping:")
for var, scales in VAR_SCALE_MAP.items():
    if scales: print(f"    {var:15s} → {', '.join(scales)}")

# ── Clean ──
df_c = df_all.dropna(subset=[YIELD_COL]).copy()
df_c = df_c[df_c[YIELD_COL] > 0].copy()
if "no_sow" in df_c.columns: df_c = df_c[df_c["no_sow"] != 1].copy()

# Drop features >95% NaN
nan_f = df_c[feature_cols].isnull().mean()
feature_cols = [c for c in feature_cols if nan_f.get(c, 0) <= 0.95]
nna = df_c[feature_cols].isnull().sum().sum()
if nna > 0: df_c[feature_cols] = df_c[feature_cols].fillna(df_c[feature_cols].median())

# ══════════════════════════════════════════════════════════════════════
# 2. SPATIAL CLUSTERING (replaces lat/lon)
# ══════════════════════════════════════════════════════════════════════
print(f"\n[STEP 2] Spatial clustering (K={N_CLUSTERS}) ...")

# Soil features for clustering
clust_soil = [c for c in ["pawc_0_30_mm","pawc_0_60_mm","ph_0_30","minN_0_30","profile_depth_cm"]
              if c in df_c.columns]

# Climate normals from training period
train_period = df_c[df_c["year"] < FIRST_TEST_YEAR]
clust_climate = [c for c in ["rain_sum_crop","rad_sum_crop","tmean_crop","vpd_mean_crop",
                              "tt_sum_crop","fasw_mean_crop","fw_photo_crop"]
                 if c in df_c.columns]
# Fallback: use monthly m0 if crop features not available
if not clust_climate:
    clust_climate = [c for c in ["rain_sum_m0","rad_sum_m0","tmean_m0"] if c in df_c.columns]

cell_norms = (train_period.groupby(["lat","lon"])[clust_climate]
              .mean().rename(columns={c:f"norm_{c}" for c in clust_climate}).reset_index())
norm_cols = [f"norm_{c}" for c in clust_climate]

# Merge normals
df_c["_lr"] = np.round(df_c["lat"].astype(float),3)
df_c["_lo"] = np.round(df_c["lon"].astype(float),3)
cell_norms["_lr"] = np.round(cell_norms["lat"].astype(float),3)
cell_norms["_lo"] = np.round(cell_norms["lon"].astype(float),3)
cell_norms.drop(columns=["lat","lon"], inplace=True)
nb = len(df_c)
df_c = df_c.merge(cell_norms, on=["_lr","_lo"], how="left")
assert len(df_c) == nb

# Cluster on unique cells
cells = df_c.drop_duplicates(subset=["_lr","_lo"])[["_lr","_lo"]+clust_soil+norm_cols].copy()
for c in clust_soil+norm_cols:
    if c in cells: cells[c] = cells[c].fillna(cells[c].median())

X_clust = cells[clust_soil+norm_cols].values.astype(np.float32)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clust)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
cells["cluster_id"] = kmeans.fit_predict(X_scaled)

# Save cluster model
json.dump({"scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist(),
           "centers":kmeans.cluster_centers_.tolist(),"n_clusters":N_CLUSTERS,
           "input_cols":clust_soil+norm_cols},
          open(OUT_DIR/"cluster_model.json","w"), indent=2)
cells[["_lr","_lo","cluster_id"]].rename(columns={"_lr":"lat","_lo":"lon"}).to_csv(
    OUT_DIR/"grid_cluster_assignments.csv", index=False)

# Merge clusters back
df_c = df_c.merge(cells[["_lr","_lo","cluster_id"]], on=["_lr","_lo"], how="left")
df_c.drop(columns=["_lr","_lo"]+norm_cols, inplace=True, errors="ignore")

# One-hot encode
for k in range(N_CLUSTERS):
    cn = f"cluster_{k}"
    df_c[cn] = (df_c["cluster_id"]==k).astype(np.float32)
    feature_cols.append(cn)

print(f"  Cluster sizes: {dict(cells['cluster_id'].value_counts().sort_index())}")
print(f"  Final features: {len(feature_cols)}")

# Cluster map
if "lat" in df_c.columns:
    cm = cells[["_lr","_lo","cluster_id"]].copy()
    fig,ax=plt.subplots(figsize=(12,8))
    ax.scatter(cm["_lo"],cm["_lr"],c=cm["cluster_id"],cmap="tab20",s=3,edgecolors="none")
    ax.set(xlabel="Longitude",ylabel="Latitude",title=f"Spatial Clusters (K={N_CLUSTERS})")
    ax.set_facecolor("#f0f0f0")
    fig.tight_layout(); fig.savefig(FIG_DIR/"cluster_map.png",dpi=200); plt.close(fig)

# ── Prepare arrays ──
X = df_c[feature_cols].values.astype(np.float32)
y_raw = df_c[YIELD_COL].values.astype(np.float32)
years = df_c["year"].values
lats = df_c["lat"].values.astype(np.float32) if "lat" in df_c.columns else None
lons = df_c["lon"].values.astype(np.float32) if "lon" in df_c.columns else None

# ── Yield mode: raw or anomaly ──
cell_mean = None  # will be set if anomaly mode
if YIELD_MODE == "anomaly":
    # Compute per-cell mean yield from TRAINING period only (avoid leakage)
    train_mask = df_c["year"] < FIRST_TEST_YEAR
    cell_means_df = (df_c[train_mask].groupby(["lat","lon"])[YIELD_COL]
                     .mean().rename("cell_mean_yield").reset_index())
    # Merge back to full dataset
    df_c = df_c.merge(cell_means_df, on=["lat","lon"], how="left")
    # For cells with no training data, use global mean
    global_mean = df_c.loc[train_mask, YIELD_COL].mean()
    df_c["cell_mean_yield"] = df_c["cell_mean_yield"].fillna(global_mean)
    cell_mean = df_c["cell_mean_yield"].values.astype(np.float32)
    y = y_raw - cell_mean  # anomaly = observed - cell mean
    print(f"  YIELD MODE: anomaly (y = yield - cell_mean)")
    print(f"  Anomaly range: {y.min():.2f} to {y.max():.2f} t/ha (mean={y.mean():.3f})")
    # Save cell means for converting predictions back
    cell_means_df.to_csv(OUT_DIR / "cell_mean_yields.csv", index=False)
else:
    y = y_raw
    print(f"  YIELD MODE: raw")

print(f"  X: {X.shape} | y: {y.mean():.2f}+/-{y.std():.2f} t/ha")

# ══════════════════════════════════════════════════════════════════════
# 3. EXPANDING WINDOW CV
# ══════════════════════════════════════════════════════════════════════
print(f"\n[STEP 3] Expanding CV ...")
import xgboost as xgb
try: import joblib
except: joblib=None

xgb_params = {"n_estimators":1500,"max_depth":6,"learning_rate":0.05,"subsample":0.8,
              "colsample_bytree":0.8,"min_child_weight":10,"reg_alpha":0.1,"reg_lambda":1.0,
              "random_state":SEED,"n_jobs":-1, **gpu_params}

fold_results,ppp,pypf,fold_models=[],[],[],{}
for ts in range(FIRST_TEST_YEAR, END_YEAR+1):
    te=ts-1; fn=f"train_{START_YEAR}-{te}_test_{ts}-{END_YEAR}"
    trm=years<=te; tem=(years>=ts)&(years<=END_YEAR)
    Xtr,ytr=X[trm],y[trm]; Xte,yte=X[tem],y[tem]
    if len(Xte)==0 or len(Xtr)==0: continue
    nv=max(100,int(len(Xtr)*0.1)); rng=np.random.RandomState(SEED)
    vs=rng.choice(len(Xtr),nv,replace=False); ts2=np.setdiff1d(np.arange(len(Xtr)),vs)
    mdl=xgb.XGBRegressor(**xgb_params)
    mdl.fit(Xtr[ts2],ytr[ts2],eval_set=[(Xtr[vs],ytr[vs])],verbose=False)
    yp_model = mdl.predict(Xte)  # model output (anomaly or raw depending on mode)

    # Convert back to raw yield for metrics and storage
    if YIELD_MODE == "anomaly" and cell_mean is not None:
        yp_raw = yp_model + cell_mean[tem]   # predicted anomaly + cell mean = predicted yield
        yte_raw = y_raw[tem]                  # actual raw yield for metrics
    else:
        yp_raw = yp_model
        yte_raw = yte

    fold_models[fn]=mdl
    if joblib: joblib.dump(mdl,OUT_DIR/f"model_{fn}.joblib")
    m=calc_metrics(yte_raw, yp_raw)
    m.update({"fold":fn,"train_years":f"{START_YEAR}-{te}","test_years":f"{ts}-{END_YEAR}","train_size":int(trm.sum())})
    fold_results.append(m)
    print(f"  {fn}: R2={m['R2']:.3f} RMSE={m['RMSE_tha']:.3f} Bias={m['Bias_tha']:.3f} LCCC={m['LCCC']:.3f}")
    tyrs=years[tem]
    for yr in sorted(np.unique(tyrs)):
        ym=tyrs==yr
        if ym.sum()==0: continue
        pm=calc_metrics(yte_raw[ym], yp_raw[ym])
        pm.update({"fold":fn,"train_years":f"{START_YEAR}-{te}","test_year":int(yr),"train_size":int(trm.sum())})
        pypf.append(pm)
        print(f"    {yr}: R2={pm['R2']:.3f} RMSE={pm['RMSE_tha']:.3f} Bias={pm['Bias_tha']:.3f} LCCC={pm['LCCC']:.3f}")
    pp=pd.DataFrame({"year":tyrs,"lat":lats[tem] if lats is not None else np.nan,
                      "lon":lons[tem] if lons is not None else np.nan,
                      "obs_tha":yte_raw,"pred_tha":yp_raw,"residual_tha":yp_raw-yte_raw,"fold":fn})
    ppp.append(pp)

fold_df=pd.DataFrame(fold_results); fold_df.to_csv(OUT_DIR/"fold_metrics.csv",index=False)
pypf_df=pd.DataFrame(pypf); pypf_df.to_csv(OUT_DIR/"per_year_per_fold_metrics.csv",index=False)
pred_df=pd.concat(ppp,ignore_index=True); pred_df.to_parquet(OUT_DIR/"per_point_predictions.parquet",index=False)
json.dump({"var_scale_map":{k:v for k,v in VAR_SCALE_MAP.items() if v},
           "yield_mode":YIELD_MODE,
           "features":feature_cols,"n_features":len(feature_cols),
           "n_clusters":N_CLUSTERS,"xgb_params":{k:v for k,v in xgb_params.items() if not callable(v)},
           "start_year":START_YEAR,"end_year":END_YEAR,"first_test_year":FIRST_TEST_YEAR,"seed":SEED},
          open(OUT_DIR/"run_config.json","w"),indent=2)

# ══════════════════════════════════════════════════════════════════════
# SAVE ALL INFERENCE ARTEFACTS
# ══════════════════════════════════════════════════════════════════════
# Everything needed to:
#   (a) reproduce SHAP plots without retraining
#   (b) run predictions on new/future climate data
#   (c) understand exactly what was trained
#
# Already saved above:
#   model_{fold}.joblib          — trained XGBoost models per fold
#   cluster_model.json           — scaler + k-means centroids for spatial clustering
#   grid_cluster_assignments.csv — cluster ID per grid cell
#   cell_mean_yields.csv         — cell means (anomaly mode only)
#   run_config.json              — feature list, params, var_scale_map
#
# Additional artefacts for standalone inference:

print("\n[SAVING] Inference artefacts ...")

# 1. Feature medians — for filling NaN in new data (same values as training)
feature_medians = {}
for i, c in enumerate(feature_cols):
    vals = X[:, i]
    feature_medians[c] = float(np.nanmedian(vals))
json.dump(feature_medians, open(OUT_DIR / "feature_medians.json", "w"), indent=2)

# 2. Feature column order — critical for prediction (model expects exact order)
#    Already in run_config.json["features"], but save separately for clarity
pd.DataFrame({"col_index": range(len(feature_cols)),
              "feature_name": feature_cols}).to_csv(
    OUT_DIR / "feature_column_order.csv", index=False)

# 3. Training data summary — for sanity checks on new data
train_mask_final = years < FIRST_TEST_YEAR
train_summary = {"n_train_samples": int(train_mask_final.sum()),
                 "n_test_samples": int((~train_mask_final).sum()),
                 "n_grid_cells": int(len(np.unique(
                     list(zip(lats[train_mask_final], lons[train_mask_final]))))),
                 "yield_mean_tha": float(y_raw[train_mask_final].mean()),
                 "yield_std_tha": float(y_raw[train_mask_final].std()),
                 "yield_min_tha": float(y_raw[train_mask_final].min()),
                 "yield_max_tha": float(y_raw[train_mask_final].max()),
                 "yield_mode": YIELD_MODE}
if YIELD_MODE == "anomaly":
    train_summary["anomaly_mean"] = float(y[train_mask_final].mean())
    train_summary["anomaly_std"] = float(y[train_mask_final].std())

# Per-feature stats (min/max/mean/std from training data)
feat_stats = []
for i, c in enumerate(feature_cols):
    v = X[train_mask_final, i]
    feat_stats.append({"feature": c, "mean": float(np.nanmean(v)),
                       "std": float(np.nanstd(v)),
                       "min": float(np.nanmin(v)),
                       "max": float(np.nanmax(v)),
                       "pct_nan": float(np.isnan(v).mean())})
pd.DataFrame(feat_stats).to_csv(OUT_DIR / "feature_stats_training.csv", index=False)

# 4. Weekly/monthly/crop slot definitions — so new data can be constructed identically
json.dump({"weekly_slots": WEEKLY_SLOTS,
           "monthly_slots": MONTHLY_SLOTS,
           "crop_windows": CROP_WINDOWS,
           "crop_extras": CROP_EXTRAS,
           "var_scale_map": {k:v for k,v in VAR_SCALE_MAP.items() if v},
           "cluster_soil_cols": clust_soil,
           "cluster_climate_cols": clust_climate,
           "train_summary": train_summary},
          open(OUT_DIR / "inference_config.json", "w"), indent=2)

print(f"  Saved: feature_medians.json, feature_column_order.csv,")
print(f"         feature_stats_training.csv, inference_config.json")
print(f"  Models: {len(fold_models)} fold models (.joblib)")
print(f"  To predict on new data, load model + run_config.json + cluster_model.json")
if YIELD_MODE == "anomaly":
    print(f"  + cell_mean_yields.csv (for converting anomaly → raw yield)")

pf=pred_df[pred_df["fold"]==fold_results[0]["fold"]]
ov=calc_metrics(pf["obs_tha"].values,pf["pred_tha"].values)
pd.DataFrame([ov]).to_csv(OUT_DIR/"overall_metrics.csv",index=False)
print(f"\n  Primary: R2={ov['R2']:.3f} RMSE={ov['RMSE_tha']:.3f} LCCC={ov['LCCC']:.3f} Bias={ov['Bias_tha']:.3f}")

# ══════════════════════════════════════════════════════════════════════
# 4. PLOTS
# ══════════════════════════════════════════════════════════════════════
print(f"\n[STEP 4] Plots ...")
obs_p,pred_p=pf["obs_tha"].values,pf["pred_tha"].values
fig,ax=plt.subplots(figsize=(8,8))  
ax.hexbin(obs_p,pred_p,gridsize=80,cmap="YlOrRd",mincnt=1); plt.colorbar(ax.collections[0],ax=ax)
lim=max(obs_p.max(),pred_p.max())*1.05
ax.plot([0,lim],[0,lim],"k--",lw=1.5); z=np.polyfit(obs_p,pred_p,1)
ax.plot([0,lim],[z[1],z[1]+z[0]*lim],"b-",lw=1,alpha=0.6)
ax.set(xlabel="APSIM (t/ha)",ylabel="ML (t/ha)",xlim=(0,lim),ylim=(0,lim),aspect="equal")
ax.set_title(f"Multiscale: R2={ov['R2']:.3f} RMSE={ov['RMSE_tha']:.3f} LCCC={ov['LCCC']:.3f}")
fig.tight_layout(); fig.savefig(FIG_DIR/"obs_vs_pred.png",dpi=200); plt.close(fig)

for mt,lb,cm in [("R2","R2","RdYlGn"),("RMSE_tha","RMSE","YlOrRd_r"),("Bias_tha","Bias","RdBu_r"),("LCCC","LCCC","RdYlGn")]:
    pv=pypf_df.pivot_table(index="train_years",columns="test_year",values=mt)
    if pv.empty: continue
    fig,ax=plt.subplots(figsize=(max(8,len(pv.columns)*1.2),max(4,len(pv)*0.8)))
    kw={"vmin":-max(abs(pv.min().min()),abs(pv.max().max()),0.01),"vmax":max(abs(pv.min().min()),abs(pv.max().max()),0.01)} if "Bias" in mt else {}
    im=ax.imshow(pv.values,cmap=cm,aspect="auto",**kw); plt.colorbar(im,ax=ax,label=lb,shrink=0.8)
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels([str(int(c)) for c in pv.columns],fontsize=9,rotation=45)
    ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index,fontsize=9); ax.set_title(f"{"Multiscale"}: {lb}")
    for i in range(len(pv.index)):
        for j in range(len(pv.columns)):
            v=pv.values[i,j]
            if not np.isnan(v): ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR/f"heatmap_{mt}.png",dpi=200,bbox_inches="tight"); plt.close(fig)

# Spatial bias
if lats is not None:
    sd=pred_df.groupby(["fold","year","lat","lon"]).agg(bias_tha=("residual_tha","mean"),n=("obs_tha","count")).reset_index()
    sd["train_years"]=sd["fold"].str.extract(r"train_(\d+-\d+)"); sd.to_csv(OUT_DIR/"spatial_bias.csv",index=False)
    nf=len(fold_results); fig,axes=plt.subplots(1,nf,figsize=(6*nf,5))
    if nf==1: axes=[axes]
    for i,fr in enumerate(fold_results):
        ax=axes[i]; fy=int(fr["test_years"].split("-")[0])
        sp=sd[(sd["fold"]==fr["fold"])&(sd["year"]==fy)]
        if len(sp)==0: continue
        vm=max(abs(sp["bias_tha"].quantile(0.05)),abs(sp["bias_tha"].quantile(0.95)),0.3)
        sc=ax.scatter(sp["lon"],sp["lat"],c=sp["bias_tha"],cmap="RdBu_r",s=3,vmin=-vm,vmax=vm,edgecolors="none")
        plt.colorbar(sc,ax=ax,shrink=0.7); ax.set_title(f"Train {fr['train_years']}->{fy}"); ax.set_facecolor("#f0f0f0")
    fig.suptitle(f"{"Multiscale"}: Spatial Bias",fontsize=13,y=1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR/"spatial_bias.png",dpi=200,bbox_inches="tight"); plt.close(fig)

# Yield level bias
p10,p90=np.percentile(y,10),np.percentile(y,90)
lo=["Low (<P10)","Medium","High (>P90)"]
pred_df["yield_level"]=pd.Categorical(np.where(pred_df["obs_tha"]<p10,lo[0],np.where(pred_df["obs_tha"]>p90,lo[2],lo[1])),categories=lo,ordered=True)
br=[]
for (fl,yr,lv),g in pred_df.groupby(["fold","year","yield_level"],observed=True):
    if len(g)<2: continue
    m=calc_metrics(g["obs_tha"].values,g["pred_tha"].values)
    m.update({"fold":fl,"test_year":int(yr),"yield_level":lv,"p10":p10,"p90":p90}); br.append(m)
pd.DataFrame(br).to_csv(OUT_DIR/"bias_by_yield_level.csv",index=False)

fig,ax=plt.subplots(figsize=(10,6))
ax.hexbin(obs_p,pred_p-obs_p,gridsize=60,cmap="RdBu_r",mincnt=1); ax.axhline(0,color="k")
plt.colorbar(ax.collections[0],ax=ax); ax.set(xlabel="APSIM (t/ha)",ylabel="Bias (t/ha)")
fig.tight_layout(); fig.savefig(FIG_DIR/"conditional_bias.png",dpi=200); plt.close(fig)

# ══════════════════════════════════════════════════════════════════════
# 5. PER-FOLD SHAP + NORMALISED GROUP IMPORTANCE
# ══════════════════════════════════════════════════════════════════════
print(f"\n[STEP 5] SHAP ...")
try: import shap; print(f"  SHAP {shap.__version__}")
except: shap=None; print("  [WARN] shap not installed")

if shap:
    groups = build_feature_groups(feature_cols)
    all_ranks = []

    for fr in fold_results:
        fn=fr["fold"]; fy=int(fr["test_years"].split("-")[0]); tl=fr["train_years"]
        mdl=fold_models[fn]; ym=years==fy; Xy=X[ym]
        ns=min(SHAP_MAX_SAMPLES,len(Xy))
        Xs=Xy[np.random.RandomState(SEED).choice(len(Xy),ns,replace=False)] if ns<len(Xy) else Xy
        print(f"  {fn} -> SHAP {fy} ({ns} samples)")
        ex=shap.TreeExplainer(mdl); sv=ex.shap_values(Xs)

        pd.DataFrame(sv,columns=feature_cols).to_parquet(SHAP_DIR/f"shap_values_{fn}_{fy}.parquet",index=False)
        ma=np.abs(sv).mean(axis=0); si=np.argsort(ma)[::-1]
        rd=pd.DataFrame({"rank":range(1,len(feature_cols)+1),"feature":[feature_cols[i] for i in si],
                         "mean_abs_shap_tha":ma[si],"fold":fn,"test_year":fy})
        rd.to_csv(SHAP_DIR/f"shap_ranking_{fn}_{fy}.csv",index=False); all_ranks.append(rd)

        # Bar
        tn=min(30,len(feature_cols)); ti=si[:tn]
        fig,ax=plt.subplots(figsize=(10,12))
        ax.barh(range(tn),ma[ti][::-1],color="#3498db")
        ax.set_yticks(range(tn)); ax.set_yticklabels([feature_cols[i] for i in ti][::-1],fontsize=8)
        ax.set_xlabel("Mean |SHAP| (t/ha)"); ax.set_title(f"Train {tl} -> Test {fy}")
        fig.tight_layout(); fig.savefig(SHAP_DIR/f"shap_bar_{fn}_{fy}.png",dpi=200); plt.close(fig)

        # Beeswarm
        fig=plt.figure(figsize=(12,14))
        shap.summary_plot(sv,Xs,feature_names=feature_cols,max_display=30,show=False,plot_size=None)
        plt.title(f"Train {tl} -> Test {fy}"); plt.tight_layout()
        plt.savefig(SHAP_DIR/f"shap_beeswarm_{fn}_{fy}.png",dpi=200,bbox_inches="tight"); plt.close("all")
        print(f"    Top5: {[feature_cols[i] for i in si[:5]]}")

    pd.concat(all_ranks,ignore_index=True).to_csv(OUT_DIR/"shap_rankings_all_folds.csv",index=False)

    # ── NORMALISED GROUP SHAP ──
    # Total group SHAP / number of features in group = mean SHAP per feature
    svp = list(SHAP_DIR.glob("shap_values_*"))
    if svp:
        sv0 = pd.read_parquet(svp[0]).values
        rows_total, rows_norm = [], []

        for gn, gc in groups.items():
            ci = [feature_cols.index(c) for c in gc if c in feature_cols]
            if not ci: continue
            total_shap = np.abs(sv0[:,ci]).sum(axis=1).mean()
            n_feats = len(ci)
            norm_shap = total_shap / n_feats  # normalised by group size

            rows_total.append({"group":gn, "total_shap_tha":total_shap, "n_features":n_feats})
            rows_norm.append({"group":gn, "norm_shap_tha_per_feat":norm_shap, "n_features":n_feats,
                              "total_shap_tha":total_shap})

        gdf_total = pd.DataFrame(rows_total).sort_values("total_shap_tha",ascending=False)
        gdf_norm  = pd.DataFrame(rows_norm).sort_values("norm_shap_tha_per_feat",ascending=False)
        gdf_total.to_csv(OUT_DIR/"shap_group_total.csv",index=False)
        gdf_norm.to_csv(OUT_DIR/"shap_group_normalised.csv",index=False)

        # Plot: side-by-side total vs normalised
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Left: total (raw) SHAP
        ax = axes[0]
        d = gdf_total
        cl = plt.cm.Set3(np.linspace(0,1,len(d)))
        ax.barh(range(len(d)), d["total_shap_tha"].values[::-1], color=cl[::-1])
        ax.set_yticks(range(len(d))); ax.set_yticklabels([f"{r['group']} ({r['n_features']})" for _,r in d.iloc[::-1].iterrows()],fontsize=9)
        ax.set_xlabel("Total |SHAP| (t/ha)"); ax.set_title("Total Group SHAP\n(larger groups naturally higher)")

        # Right: normalised SHAP per feature
        ax = axes[1]
        d = gdf_norm
        ax.barh(range(len(d)), d["norm_shap_tha_per_feat"].values[::-1], color=cl[::-1])
        ax.set_yticks(range(len(d))); ax.set_yticklabels([f"{r['group']} ({r['n_features']})" for _,r in d.iloc[::-1].iterrows()],fontsize=9)
        ax.set_xlabel("Mean |SHAP| per Feature (t/ha)"); ax.set_title("Normalised Group SHAP\n(fair comparison across group sizes)")

        fig.suptitle(f"{"Multiscale"} Features: Group Importance", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR/"shap_group_comparison.png",dpi=200,bbox_inches="tight"); plt.close(fig)

        print(f"\n  Group SHAP (normalised by size):")
        for _,r in gdf_norm.iterrows():
            print(f"    {r['group']:25s}  norm={r['norm_shap_tha_per_feat']:.4f}  "
                  f"total={r['total_shap_tha']:.4f}  ({r['n_features']} feats)")

print(f"\n{'='*60}\n  DONE -> {OUT_DIR}\n{'='*60}")

# import shap, pandas as pd
# sv = pd.read_parquet("shap_per_fold/shap_values_train_1989-2016_test_2017-2021_2017.parquet")
# cols = pd.read_csv("feature_column_order.csv")
# shap.summary_plot(sv.values, feature_names=cols["feature_name"].tolist())