"""
extract_subset.py — Extract a small subset of multiscale features for testing
==============================================================================
Selects N random grid cells (or specific lat/lon), extracts all years,
subsets to chosen features, merges into a single parquet + CSV file.

Useful for:
  - Quick local testing of ml_multiscale.py without full 27K-cell data
  - Exploring feature distributions
  - Sharing a manageable sample with collaborators

Usage:
  export DATA_ROOT=/scratch3/tri083/AADI/data
  python extract_subset.py

  # Or override defaults:
  export N_CELLS=500
  export SUBSET_OUT=/scratch3/tri083/AADI/data/subset
  python extract_subset.py
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import time

# ══════════════════════════════════════════════════════════════════════
# CONFIG — edit these or set via environment variables
# ══════════════════════════════════════════════════════════════════════

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data"))
MS_DIR    = DATA_ROOT / "processed_multiscale"
OUT_DIR   = Path(os.environ.get("SUBSET_OUT", str(DATA_ROOT / "subset")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = int(os.environ.get("START_YEAR", 1989))
END_YEAR   = int(os.environ.get("END_YEAR", 2021))
SEED       = 42

# Number of random grid cells to sample (set 0 to use ALL cells in the region)
N_CELLS = int(os.environ.get("N_CELLS", 0))  # 0 = all WA cells

# ══════════════════════════════════════════════════════════════════════
# REGION FILTER — Western Australia (lon <= 129)
# ══════════════════════════════════════════════════════════════════════
# Set to None to include all Australia
REGION_NAME = "WA"
REGION_LON_MAX = 129.0   # WA is lon <= 129
REGION_LON_MIN = None    # no lower bound
REGION_LAT_MAX = None    # no bound
REGION_LAT_MIN = None    # no bound

# ══════════════════════════════════════════════════════════════════════
# FEATURE SELECTION — which features to keep in the subset
# ══════════════════════════════════════════════════════════════════════

# Always keep these columns
KEEP_ID = ["lat", "lon", "year", "wheat_yield", "sowing_doy", "no_sow"]

# Static soil features
KEEP_SOIL = ["pawc_0_30_mm", "pawc_0_60_mm", "ph_0_30", "minN_0_30", "profile_depth_cm"]

# Crop-seasonal features (excluding n_days and fw_photo)
CROP_WINDOWS = ["W1_estab", "W2_veg", "W3_preAnth", "W4_grainFill", "W5_matur", "crop"]
CROP_VARS = ["rain_sum", "rain_days", "rad_mean", "rad_sum", "tmean", "tmax_max",
             "tmin_min", "diurnal", "tt_sum", "fT_photo", "heat_days", "frost_days",
             "vpd_mean", "fasw_mean"]

# Monthly rainfall features (all 12 slots)
MONTHLY_SLOTS = [f"mpre{i}" for i in range(6, 0, -1)] + [f"m{i}" for i in range(9)]
MONTHLY_RAIN_VARS = ["rain_sum", "rain_days"]

# Crop-seasonal extras
CROP_EXTRAS = [
    "fw_expan_mean_W2_veg", "fw_expan_mean_W3_preAnth",
    "fw_photo_min_W2_veg", "fw_photo_min_W3_preAnth", "fw_photo_min_W4_grainFill",
    "drought_severe_W3_preAnth", "drought_severe_W4_grainFill",
    "drought_moderate_W3_preAnth", "drought_moderate_W4_grainFill",
    "cum_water_stress_crop", "cum_water_stress_W3_preAnth", "cum_water_stress_W4_grainFill",
    "FASW_at_sowing", "FASW_W4_drying_rate",
    "vern_sum_W2_veg", "fD_mean_W2_veg",
    "hg_mean_W4", "heat_sen_W4", "VPD_mean_crop", "TE_mean_crop", "TT_cum_crop",
]

# ══════════════════════════════════════════════════════════════════════
# BUILD THE TARGET COLUMN LIST
# ══════════════════════════════════════════════════════════════════════

def build_target_columns():
    """Build the list of columns to extract from the parquet files."""
    cols = list(KEEP_ID) + list(KEEP_SOIL)

    # Crop-seasonal: all vars × all windows
    for win in CROP_WINDOWS:
        for v in CROP_VARS:
            cols.append(f"{v}_{win}")

    # Monthly rainfall
    for slot in MONTHLY_SLOTS:
        for v in MONTHLY_RAIN_VARS:
            cols.append(f"{v}_{slot}")

    # Crop extras
    cols.extend(CROP_EXTRAS)

    # Remove duplicates, preserve order
    seen = set()
    unique = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    target_cols = build_target_columns()
    print(f"[CONFIG] Years: {START_YEAR}-{END_YEAR}")
    print(f"[CONFIG] Target columns: {len(target_cols)}")
    print(f"[CONFIG] N_CELLS: {N_CELLS if N_CELLS > 0 else 'ALL'}")
    print(f"[CONFIG] Output: {OUT_DIR}")

    # ── Step 1: Load first year to get grid cell coordinates ──
    first_file = MS_DIR / f"wheat_multiscale_features_{START_YEAR}.parquet"
    if not first_file.exists():
        raise FileNotFoundError(f"Missing: {first_file}")

    df0 = pd.read_parquet(first_file, columns=["lat", "lon"])
    all_cells = df0[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    print(f"  Total grid cells (all Australia): {len(all_cells):,}")

    # ── Apply region filter ──
    mask = np.ones(len(all_cells), dtype=bool)
    if REGION_LON_MAX is not None: mask &= (all_cells["lon"] <= REGION_LON_MAX)
    if REGION_LON_MIN is not None: mask &= (all_cells["lon"] >= REGION_LON_MIN)
    if REGION_LAT_MAX is not None: mask &= (all_cells["lat"] <= REGION_LAT_MAX)
    if REGION_LAT_MIN is not None: mask &= (all_cells["lat"] >= REGION_LAT_MIN)
    region_cells = all_cells[mask].reset_index(drop=True)
    print(f"  Region '{REGION_NAME}': {len(region_cells):,} cells")

    if len(region_cells) == 0:
        raise ValueError(f"No cells found in region {REGION_NAME}! Check filter bounds.")

    # ── Step 2: Sample cells (within region) ──
    if N_CELLS > 0 and N_CELLS < len(region_cells):
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(region_cells), N_CELLS, replace=False)
        sample_cells = region_cells.iloc[idx].reset_index(drop=True)
        print(f"  Sampled: {len(sample_cells)} cells from {REGION_NAME}")
    else:
        sample_cells = region_cells
        print(f"  Using ALL {len(sample_cells)} {REGION_NAME} cells")

    # Round for matching
    sample_cells["_lr"] = np.round(sample_cells["lat"].astype(float), 3)
    sample_cells["_lo"] = np.round(sample_cells["lon"].astype(float), 3)
    cell_set = set(zip(sample_cells["_lr"], sample_cells["_lo"]))

    # ── Step 3: Load each year, filter cells, select columns ──
    frames = []
    for yr in range(START_YEAR, END_YEAR + 1):
        p = MS_DIR / f"wheat_multiscale_features_{yr}.parquet"
        if not p.exists():
            print(f"  [WARN] Missing {p.name}")
            continue

        # Read only the columns we need (faster I/O)
        available = pd.read_parquet(p, columns=["lat"]).columns.tolist()
        # Re-read with actual available columns
        pq_cols_available = pd.read_parquet(p).columns.tolist()
        use_cols = [c for c in target_cols if c in pq_cols_available]
        df = pd.read_parquet(p, columns=use_cols)

        # Filter to sampled cells
        df["_lr"] = np.round(df["lat"].astype(float), 3)
        df["_lo"] = np.round(df["lon"].astype(float), 3)
        mask = [(_lr, _lo) in cell_set for _lr, _lo in zip(df["_lr"], df["_lo"])]
        df = df[mask].drop(columns=["_lr", "_lo"]).reset_index(drop=True)

        frames.append(df)
        print(f"  {yr}: {len(df):,} rows x {len(df.columns)} cols")

    # ── Step 4: Merge all years ──
    merged = pd.concat(frames, ignore_index=True)

    # Report missing columns
    requested = set(target_cols)
    got = set(merged.columns)
    missing = requested - got
    if missing:
        print(f"\n  [NOTE] {len(missing)} requested columns not in data:")
        for m in sorted(missing)[:20]:
            print(f"    - {m}")
        if len(missing) > 20:
            print(f"    ... and {len(missing)-20} more")

    # ── Step 5: Summary stats ──
    n_cells = merged.groupby(["lat", "lon"]).ngroups
    n_years = merged["year"].nunique()
    has_yield = merged["wheat_yield"].notna() & (merged["wheat_yield"] > 0)
    print(f"\n  Merged: {len(merged):,} rows x {len(merged.columns)} cols")
    print(f"  Grid cells: {n_cells}  Years: {n_years}")
    print(f"  Valid yield: {has_yield.sum():,} ({has_yield.mean()*100:.1f}%)")
    if has_yield.any():
        yv = merged.loc[has_yield, "wheat_yield"]
        print(f"  Yield: {yv.mean():.2f} +/- {yv.std():.2f} t/ha "
              f"(range {yv.min():.2f} - {yv.max():.2f})")

    # ── Step 6: Save ──
    n_tag = f"{N_CELLS}cells" if N_CELLS > 0 else "allcells"
    out_pq = OUT_DIR / f"subset_{REGION_NAME}_{n_tag}_{START_YEAR}-{END_YEAR}.parquet"
    out_csv = OUT_DIR / f"subset_{REGION_NAME}_{n_tag}_{START_YEAR}-{END_YEAR}.csv"

    merged.to_parquet(out_pq, index=False)
    merged.to_csv(out_csv, index=False)

    # Save metadata
    import json
    meta = {
        "region": REGION_NAME,
        "region_lon_max": REGION_LON_MAX,
        "region_lon_min": REGION_LON_MIN,
        "region_lat_max": REGION_LAT_MAX,
        "region_lat_min": REGION_LAT_MIN,
        "n_cells": n_cells,
        "n_years": n_years,
        "n_rows": len(merged),
        "n_columns": len(merged.columns),
        "columns": merged.columns.tolist(),
        "years": sorted(merged["year"].unique().tolist()),
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "seed": SEED,
        "source_dir": str(MS_DIR),
    }
    json.dump(meta, open(OUT_DIR / "subset_metadata.json", "w"), indent=2)

    size_mb = out_pq.stat().st_size / 1e6
    print(f"\n  [WRITE] {out_pq.name} ({size_mb:.1f} MB)")
    print(f"  [WRITE] {out_csv.name}")
    print(f"  [WRITE] subset_metadata.json")
    print(f"\n[DONE] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
