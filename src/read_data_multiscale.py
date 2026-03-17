"""
read_data_multiscale.py — Multi-scale Feature Extraction
=========================================================
Extracts features at THREE temporal resolutions in a single pass:
  1. WEEKLY:  24 pre-sow + 36 post-sow = 60 slots x 16 vars = 960 features
  2. MONTHLY: 6 pre-sow + 9 post-sow = 15 slots x 16 vars = 240 features
  3. CROP-SEASONAL: 5 APSIM windows + crop total + extras = 120 features

Output: wheat_multiscale_features_{year}.parquet

Prerequisite: soil_static_features.parquet must exist.
  Generate once: export DATA_ROOT=/scratch3/tri083/AADI/data
  python -c "from common_apsim import *; build_soil_static(load_wheat_mask())"

SLURM:
  export START_YEAR=${SLURM_ARRAY_TASK_ID}; export END_YEAR=${SLURM_ARRAY_TASK_ID}
  python read_data_multiscale.py
"""
import time
import numpy as np
import pandas as pd
from config import (OUT_DIR, YEARS, WINDOWS, CUM_MATURE,
    N_PRE_SOW_MONTHS, N_POST_SOW_MONTHS, WEEKLY_PRE_SOW, WEEKLY_POST_SOW)

from common_apsim import (load_wheat_mask, load_climate_year, load_sm,
    load_yield, compute_sowing, compute_daily_vars,
    aggregate_slot, build_soil_static, lookup_soil)

OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_year(year, wheat_mask, soil_df):
    out_path = OUT_DIR / f"wheat_multiscale_features_{year}.parquet"
    if out_path.exists(): print(f"[SKIP] {year}"); return
    yt0 = time.time()
    print(f"\n{'='*60}\n  YEAR {year}\n{'='*60}")

    # Climate
    ds, lat_vals, lon_vals = load_climate_year(year, wheat_mask)
    n_pts = len(lat_vals); n_time = ds.dims["time"]
    t0=time.time()
    tmin=ds["tmin"].values.astype(np.float32); tmax=ds["tmax"].values.astype(np.float32)
    rain=ds["rain"].values.astype(np.float32); radn=ds["radn"].values.astype(np.float32)
    print(f"  numpy: {time.time()-t0:.1f}s  pts={n_pts:,} days={n_time}")

    # Soil moisture
    sm = load_sm(year, ds["time"], lat_vals, lon_vals)

    # Static soil
    soil = lookup_soil(lat_vals, lon_vals, soil_df); pawc30 = soil["pawc30"]

    #  Sowing
    doys = np.arange(1, n_time+1, dtype=np.int32)
    sow_idx = compute_sowing(rain, sm, pawc30, lat_vals, lon_vals, year, doys)
    sow_doy = np.where(sow_idx>=0, doys[np.clip(sow_idx,0,n_time-1)], np.nan).astype(np.float32)
    no_sow = (sow_idx<0).astype(np.int8)
    print(f"  Sown: {(no_sow==0).sum()}/{n_pts}")

    # Daily APSIM vars
    t0=time.time()
    dv = compute_daily_vars(tmin, tmax, rain, radn, sm, pawc30, lat_vals, n_time)
    dtt=dv["dtt"]; ft=dv["ft"]; vpd=dv["vpd"]; fasw=dv["fasw"]; fw=dv["fw"]
    print(f"  Daily vars: {time.time()-t0:.1f}s")

    # Cumulative TT from sowing
    t0=time.time()
    after_sow = np.zeros((n_time, n_pts), dtype=bool)
    for d in range(n_time): after_sow[d,:] = (d>=sow_idx) & (sow_idx>=0)
    dtt_crop = np.where(after_sow, dtt, 0)
    cum_tt = np.cumsum(dtt_crop, axis=0); cum_tt = np.where(after_sow, cum_tt, 0)
    crop_mask = after_sow & (cum_tt < CUM_MATURE)
    time_months = ds["time"].dt.month.values
    time_dates = pd.DatetimeIndex(ds["time"].values)
    sow_month = np.full(n_pts, 5, dtype=np.int32)
    vs = (sow_idx>=0) & (sow_idx<n_time)
    sow_month[vs] = np.array([time_dates[s].month for s in sow_idx[vs]])
    day_idx = np.arange(n_time)[:, None]; sow_2d = sow_idx[None, :]
    print(f"  Masks: {time.time()-t0:.1f}s")

    F = {}
    # Static
    F["lat"]=lat_vals; F["lon"]=lon_vals; F["sowing_doy"]=sow_doy; F["no_sow"]=no_sow
    F["pawc_0_30_mm"]=pawc30; F["pawc_0_60_mm"]=soil["pawc60"]
    F["ph_0_30"]=soil["ph30"]; F["minN_0_30"]=soil["minN30"]; F["profile_depth_cm"]=soil["prof_d"]

    # WEEKLY
    t0=time.time()
    for slot in range(-WEEKLY_PRE_SOW, WEEKLY_POST_SOW):
        ws = sow_idx + slot*7; we = sow_idx + (slot+1)*7
        mask = (day_idx>=ws[None,:]) & (day_idx<we[None,:]) & (sow_idx[None,:]>=0) & (day_idx>=0) & (day_idx<n_time)
        agg = aggregate_slot(mask, tmin, tmax, rain, radn, dtt, ft, vpd, fasw, fw)
        sn = f"wpre{abs(slot)}" if slot<0 else f"w{slot}"
        for v,a in agg.items(): F[f"{v}_{sn}"] = a
    print(f"  Weekly: {time.time()-t0:.1f}s")

    # MONTHLY
    t0=time.time()
    for slot in range(-N_PRE_SOW_MONTHS, N_POST_SOW_MONTHS):
        tm = ((sow_month+slot-1)%12)+1
        mm = (time_months[:,None]==tm[None,:])
        if slot>=0: mm = mm & (day_idx>=sow_2d) & (sow_2d>=0)
        else: mm = mm & ((day_idx<sow_2d) | (sow_2d<0))
        agg = aggregate_slot(mm, tmin, tmax, rain, radn, dtt, ft, vpd, fasw, fw)
        sn = f"mpre{abs(slot)}" if slot<0 else f"m{slot}"
        for v,a in agg.items(): F[f"{v}_{sn}"] = a
    print(f"  Monthly: {time.time()-t0:.1f}s")

    # CROP-SEASONAL 
    t0=time.time()
    # C1. APSIM windows
    for wn,(lo,hi) in WINDOWS.items():
        wm = (cum_tt>=lo) & (cum_tt<hi) & after_sow
        agg = aggregate_slot(wm, tmin, tmax, rain, radn, dtt, ft, vpd, fasw, fw)
        for v,a in agg.items(): F[f"{v}_{wn}"] = a

    #  Whole crop
    agg_c = aggregate_slot(crop_mask, tmin, tmax, rain, radn, dtt, ft, vpd, fasw, fw)
    for v,a in agg_c.items(): F[f"{v}_crop"] = a

    #  Extra crop-seasonal features
    fw_e = dv["fw_expan"]
    for wn in ["W2_veg","W3_preAnth"]:
        lo,hi=WINDOWS[wn]; wm=(cum_tt>=lo)&(cum_tt<hi)&after_sow
        nd=np.maximum(wm.sum(axis=0).astype(np.float32),1)
        F[f"fw_expan_mean_{wn}"] = np.where(wm,fw_e,0).sum(axis=0)/nd
        fe_min = np.where(wm,fw_e,9999).min(axis=0)
        F[f"fw_expan_min_{wn}"] = np.where(wm.sum(axis=0)>0, fe_min, np.nan)

    for wn in ["W2_veg","W3_preAnth","W4_grainFill"]:
        lo,hi=WINDOWS[wn]; wm=(cum_tt>=lo)&(cum_tt<hi)&after_sow
        fwm=np.where(wm,fw,9999).min(axis=0)
        F[f"fw_photo_min_{wn}"] = np.where(wm.sum(axis=0)>0, fwm, np.nan)

    for wn in ["W2_veg","W3_preAnth","W4_grainFill"]:
        lo,hi=WINDOWS[wn]; wm=(cum_tt>=lo)&(cum_tt<hi)&after_sow
        F[f"drought_severe_{wn}"] = np.where(wm&(fw<0.5),1,0).sum(axis=0).astype(np.float32)
        F[f"drought_moderate_{wn}"] = np.where(wm&(fw<0.8)&(fw>=0.5),1,0).sum(axis=0).astype(np.float32)

    F["cum_water_stress_crop"] = np.where(crop_mask,1-fw,0).sum(axis=0)
    for wn in ["W3_preAnth","W4_grainFill"]:
        lo,hi=WINDOWS[wn]; wm=(cum_tt>=lo)&(cum_tt<hi)&after_sow
        F[f"cum_water_stress_{wn}"] = np.where(wm,1-fw,0).sum(axis=0)

    sow_win = (day_idx>=sow_2d) & (day_idx<=sow_2d+5) & (sow_2d>=0)
    F["FASW_at_sowing"] = np.where(sow_win,fasw,0).sum(axis=0)/np.maximum(sow_win.sum(axis=0).astype(np.float32),1)

    gfl,gfh = WINDOWS["W4_grainFill"]; ttm=(gfl+gfh)/2
    w4=(cum_tt>=gfl)&(cum_tt<gfh)&after_sow
    w41=w4&(cum_tt<ttm); w42=w4&(cum_tt>=ttm)
    F["FASW_W4_drying_rate"] = (np.where(w41,fasw,0).sum(axis=0)/np.maximum(w41.sum(axis=0).astype(np.float32),1) -
                                 np.where(w42,fasw,0).sum(axis=0)/np.maximum(w42.sum(axis=0).astype(np.float32),1))

    vl,vh=WINDOWS["W2_veg"]; wv=(cum_tt>=vl)&(cum_tt<vh)&after_sow
    nv=np.maximum(wv.sum(axis=0).astype(np.float32),1)
    F["vern_sum_W2_veg"]=np.where(wv,dv["dvern"],0).sum(axis=0)
    F["fD_mean_W2_veg"]=np.where(wv,dv["fD"],0).sum(axis=0)/nv

    wgf=(cum_tt>=gfl)&(cum_tt<gfh)&after_sow; ngf=np.maximum(wgf.sum(axis=0).astype(np.float32),1)
    F["hg_mean_W4"]=np.where(wgf,dv["hg"],0).sum(axis=0)/ngf
    F["heat_sen_W4"]=np.where(wgf,dv["kh"],0).sum(axis=0)/ngf

    nc=np.maximum(crop_mask.sum(axis=0).astype(np.float32),1)
    F["VPD_mean_crop"]=np.where(crop_mask,vpd,0).sum(axis=0)/nc
    te=np.where(vpd>0.01,0.006/vpd,0)
    F["TE_mean_crop"]=np.where(crop_mask,te,0).sum(axis=0)/nc
    F["TT_cum_crop"]=np.where(crop_mask,dtt,0).sum(axis=0)

    print(f"  Crop-seasonal: {time.time()-t0:.1f}s")

    #  Yield
    t0=time.time()
    F["wheat_yield"] = load_yield(year, lat_vals, lon_vals)
    F["year"] = np.full(n_pts, year, dtype=np.int32)
    nv = np.sum(~np.isnan(F["wheat_yield"])&(F["wheat_yield"]>0)&(no_sow==0))
    print(f"  Yield: {time.time()-t0:.1f}s valid={nv}/{n_pts}")

    # Save
    df = pd.DataFrame(F); df.to_parquet(out_path, index=False)
    nf = len([c for c in df.columns if c not in ["lat","lon","year","wheat_yield","no_sow"]])
    print(f"  [WRITE] {out_path.name}: {len(df):,} x {len(df.columns)} ({nf} features) [{time.time()-yt0:.0f}s]")

def main():
    t0=time.time(); wm=load_wheat_mask(); sd=build_soil_static(wm)
    for y in YEARS: extract_year(y, wm, sd)
    print(f"\n[DONE] {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()