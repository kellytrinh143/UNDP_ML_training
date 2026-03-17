from __future__ import annotations

import time
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import xarray as xr
from config import (CLIMATE_ROOT,MASK_PATH,YIELD_PATH,SOIL_PATH,SOIL_APSIM,SM_DIR,
    SOIL_STATIC_OUT,TIME_CHUNK,MAX_POINTS,VARS,NORTH_LAT,NORTH_LON,WEST_LON,
    SOW_START_WEST,SOW_START_EAST,SOW_END,SOW_RAIN3,SOW_PAW,SOW_WET,
    FW_PHOTO_THRESH,FW_EXPAN_THRESH,CUM_MATURE,WINDOWS,CONVERT_KG_TO_THA)


#-----------------------------------------------------------------------
# APSIM features
#-----------------------------------------------------------------------------
def crown_temp(tmax, tmin, snow_depth=0.0):
    """Crown temperature (Eq 1-3).  Snow default = 0 → simple threshold."""
    f = 0.4 + 0.0018 * (snow_depth - 15.0) ** 2
    tc_max = np.where(tmax < 0, 2.0 + tmax * f, tmax)
    tc_min = np.where(tmin < 0, 2.0 + tmin * f, tmin)
    return (tc_max + tc_min) * 0.5


def thermal_time(tc):
    """Daily thermal time with cardinals 0/26/34 °C (Eq 4)."""
    return np.where(tc <= 0, 0.0, np.where(tc <= 26, tc, np.where(tc <= 34, 3.25 * (34.0 - tc), 0.0)))


def ft_photo(tmax, tmin):
    """Temperature stress factor for photosynthesis (Eq 20, Fig 10).
    Breakpoints: (0,0) (10,1) (25,1) (35,0)."""
    tm = (tmax + tmin) * 0.5
    return np.clip(
        np.where(tm <= 0, 0.0, np.where(tm <= 10, tm * 0.1, np.where(tm <= 25, 1.0, np.where(tm <= 35, (35.0 - tm) * 0.1, 0.0)))),
        0,
        1,
    )
def grain_fill_fac(tmax,tmin):
    """Temperature factor for grain filling rate (Eq 49, Fig 17)."""
    tm = (tmax + tmin) / 2.0
    return xr.where(tm <= 0, 0.0, xr.where(tm <= 18, tm / 18.0, xr.where(tm <= 25, 1.0, xr.where(tm <= 35, (35.0 - tm) / 10.0, 0.0)))).clip(0, 1)


def heat_sen(tmax): return np.clip(np.where(tmax<=34,0,np.where(tmax<=44,(tmax-34)*0.04,0.4)),0,0.5)

def vern_daily(tc_mean, tmax, tmin):
    """Daily vernalisation increment (Eq 9).
    Conditions: Tmax < 30 AND Tmin < 15."""
    t1 = 1.4 - 0.0778 * tc_mean
    t2 = 0.5 + 13.44 * tc_mean / (tmax - tmin + 3.0) ** 2
    dv = xr.where((tmax < 30) & (tmin < 15), np.minimum(t1, t2), 0.0)
    return dv.clip(min=0)

def vpd_daily(tmax,tmin):
    """Vapour pressure deficit (Eq 88, Tanner & Endo 1983)."""
    es_max = 6.1078 * np.exp(17.269 * tmax / (237.3 + tmax))
    es_min = 6.1078 * np.exp(17.269 * tmin / (237.3 + tmin))
    return np.clip(0.75 * (es_max - es_min), 0.01, None)

def photoperiod_fac(doy,lat,Rp=3.0):
    """Day length (h) with civil twilight = -6° (Section 3.4, Eq 28-29)."""
    """(Eq 8).  fD = 1 - 0.002·Rp·(20-Lp)²."""
    lr=np.deg2rad(lat); d=np.deg2rad(23.45*np.sin(2*np.pi/365.25*(doy-82.25)))
    ch=(np.sin(np.deg2rad(-6))-np.sin(lr)*np.sin(d))/(np.cos(lr)*np.cos(d))
    dl=2*np.rad2deg(np.arccos(np.clip(ch,-1,1)))/15; return np.clip(1-0.002*Rp*(20-dl)**2,0,1)



def compute_daily_vars(tmin,tmax,rain,radn,sm,pawc30,lat_vals,n_time):
    """Compute all daily APSIM intermediates. Returns dict of (n_time,n_pts) arrays."""
    tc=crown_temp(tmax,tmin); dtt=thermal_time(tc); ft=ft_photo(tmax,tmin)
    hg=grain_fill_fac(tmax,tmin); kh=heat_sen(tmax); dv=vern_daily(tc,tmax,tmin); vpd=vpd_daily(tmax,tmin)
    if sm is not None: fasw=np.clip(sm/pawc30[None,:],0,2); fw=np.clip(fasw/FW_PHOTO_THRESH,0,1); fe=np.clip(fasw/FW_EXPAN_THRESH,0,1)
    else: fasw=np.ones_like(tmin); fw=np.ones_like(tmin); fe=np.ones_like(tmin)
    doys=np.arange(1,n_time+1,dtype=np.int32); n=len(lat_vals)
    fD=photoperiod_fac(np.broadcast_to(doys[:n_time,None],(n_time,n)),np.broadcast_to(lat_vals[None,:],(n_time,n)))
    return dict(dtt=dtt,ft=ft,hg=hg,kh=kh,dvern=dv,vpd=vpd,fasw=fasw,fw=fw,fw_expan=fe,fD=fD,doys=doys)


# ── Temporal aggregation ──
def aggregate_slot(mask,tmin,tmax,rain,radn,dtt,ft,vpd,fasw,fw):
    """Aggregate daily vars over boolean mask (n_time,n_pts). Returns dict of (n_pts,) arrays."""
    nd=mask.sum(axis=0).astype(np.float32); ns=np.maximum(nd,1)
    ms=lambda x:np.where(mask,x,0).sum(axis=0); mm=lambda x:ms(x)/ns
    mx=lambda x:np.where(mask,x,-9999).max(axis=0); mn=lambda x:np.where(mask,x,9999).min(axis=0)
    mc=lambda c:np.where(mask&c,1,0).sum(axis=0).astype(np.float32); tm=(tmax+tmin)*0.5
    return {"rain_sum":ms(rain),"rain_days":mc(rain>=1),"rad_mean":mm(radn),"rad_sum":ms(radn),
            "tmean":mm(tm),"tmax_max":np.where(nd>0,mx(tmax),np.nan),"tmin_min":np.where(nd>0,mn(tmin),np.nan),
            "diurnal":mm(tmax-tmin),"tt_sum":ms(dtt),"fT_photo":mm(ft),"heat_days":mc(tmax>34),
            "frost_days":mc(tmin<0),"vpd_mean":mm(vpd),"fasw_mean":mm(fasw),"fw_photo":mm(fw),"n_days":nd}

# ── Data loading ──
def load_wheat_mask(): return xr.open_dataset(MASK_PATH)["wheat_mask"]

def load_variable(v,year,wm):
    nc=CLIMATE_ROOT/v/f"silo_{v}_{year}.nc"
    if not nc.exists(): raise FileNotFoundError(f"Missing: {nc}")
    ds=xr.open_dataset(nc,chunks={"time":TIME_CHUNK},decode_times=True)
    if "latitude" in ds.dims and "longitude" in ds.dims and "lat" in ds and "lon" in ds:
        ds=ds.set_coords(["lat","lon"]).swap_dims({"latitude":"lat","longitude":"lon"})
    da=ds[v] if v in ds.data_vars else ds[list(ds.data_vars)[0]]
    m=wm
    if m.sizes["lat"]!=ds.sizes["lat"] or m.sizes["lon"]!=ds.sizes["lon"] or not np.array_equal(m["lat"].values,ds["lat"].values) or not np.array_equal(m["lon"].values,ds["lon"].values):
        m=m.interp(lat=ds["lat"],lon=ds["lon"],method="nearest").astype(bool)
    if MAX_POINTS is not None:
        idx=np.argwhere(m.values)
        if idx.shape[0]>MAX_POINTS: idx=idx[:MAX_POINTS]; c=np.zeros_like(m.values,dtype=bool); c[idx[:,0],idx[:,1]]=True; m=xr.DataArray(c,coords=m.coords,dims=m.dims)
    return da.transpose("time","lat","lon").where(m).stack(point=("lat","lon")).dropna("point",how="all")

def load_climate_year(year,wm):
    st={}
    for v in VARS: t0=time.time(); st[v]=load_variable(v,year,wm); print(f"    {v}: {time.time()-t0:.1f}s")
    mi=st[VARS[0]]["point"].to_index()
    return xr.Dataset(st).reset_index("point"), mi.get_level_values(0).values.astype(np.float32), mi.get_level_values(1).values.astype(np.float32)

def load_sm(year,ti,lv,lo):
    f=SM_DIR/f"s0_{year}.nc"
    if not f.exists(): return None
    ds=xr.open_dataset(f,chunks={"time":TIME_CHUNK},decode_times=True)
    if "s0" not in ds.data_vars: return None
    sm=ds["s0"]; li=np.abs(sm["lat"].values[:,None]-lv[None,:]).argmin(axis=0)
    oi=np.abs(sm["lon"].values[:,None]-lo[None,:]).argmin(axis=0)
    sp=sm.isel(lat=xr.DataArray(li,dims="point"),lon=xr.DataArray(oi,dims="point"))
    return sp.reindex(time=ti,method="nearest",tolerance=np.timedelta64(1,"D")).transpose("time","point").values.astype(np.float32)

def load_yield(year,lv,lo):
    yds=xr.open_dataset(YIELD_PATH); yda=yds["wheat_yield"] if "wheat_yield" in yds else yds[list(yds.data_vars)[0]]
    yt=yda.sel(time=yda["time"].dt.year==year); yv=np.full(len(lv),np.nan,dtype=np.float32)
    if yt.sizes.get("time",0)>0:
        try: yv=yt.squeeze("time",drop=True).sel(lat=xr.DataArray(lv,dims="p"),lon=xr.DataArray(lo,dims="p"),method="nearest").values.astype(np.float32)
        except: pass
    if CONVERT_KG_TO_THA: yv/=1000
    return yv

# ── Vectorised sowing ──
def compute_sowing(rain,sm,pawc30,lat,lon,year,doys):
    nd,np_=rain.shape; north=(lat>NORTH_LAT)&(lon>NORTH_LON); west=(lon<=WEST_LON)
    sw=pd.Timestamp(year=year,month=SOW_START_WEST[0],day=SOW_START_WEST[1]).dayofyear
    se=pd.Timestamp(year=year,month=SOW_START_EAST[0],day=SOW_START_EAST[1]).dayofyear
    ea=pd.Timestamp(year=year,month=SOW_END[0],day=SOW_END[1]).dayofyear
    sd=np.where(west,sw,se); iw=(doys[:,None]>=sd[None,:])&(doys[:,None]<=ea)
    r3=np.zeros_like(rain); r3[2:]=rain[2:]+rain[1:-1]+rain[:-2]; ro=r3>=SOW_RAIN3
    po=np.where(north[None,:],sm>=SOW_PAW,True) if sm is not None else np.ones_like(rain,dtype=bool)
    wo=(sm<SOW_WET*pawc30[None,:]) if sm is not None else np.ones_like(rain,dtype=bool)
    c=iw&ro&po&wo; ac=c.any(axis=0); fi=c.argmax(axis=0)
    ei=min(np.searchsorted(doys,ea),nd-1)
    we=(sm[ei,:]<SOW_WET*pawc30) if sm is not None else np.ones(np_,dtype=bool)
    si=np.full(np_,-1,dtype=np.int32); si[ac]=fi[ac]; si[(~ac)&we]=ei; return si

# ── Soil static lookup ──
def lookup_soil(lv,lo,sdf):
    n=len(lv); r={"pawc30":np.full(n,50.,dtype=np.float32),"pawc60":np.full(n,np.nan,dtype=np.float32),
                  "ph30":np.full(n,np.nan,dtype=np.float32),"minN30":np.full(n,np.nan,dtype=np.float32),"prof_d":np.full(n,np.nan,dtype=np.float32)}
    if sdf is None: return r
    pts=pd.DataFrame({"_lr":np.round(lv.astype(float),3),"_lo":np.round(lo.astype(float),3),"_i":np.arange(n)})
    cs=["_lr","_lo"]+[c for c in ["pawc_0_30_mm","pawc_0_60_mm","ph_0_30","minN_0_30","profile_depth_cm"] if c in sdf.columns]
    mg=pts.merge(sdf[cs].drop_duplicates(["_lr","_lo"]),on=["_lr","_lo"],how="left").sort_values("_i")
    mp={"pawc_0_30_mm":"pawc30","pawc_0_60_mm":"pawc60","ph_0_30":"ph30","minN_0_30":"minN30","profile_depth_cm":"prof_d"}
    for s,d in mp.items():
        if s in mg.columns:
            v=mg[s].values.astype(np.float32)
            if d=="pawc30": m=np.nanmedian(v); v=np.where(np.isnan(v),m if not np.isnan(m) else 50,v)
            r[d]=v
    return r

# ── Build soil static parquet ──
def _l(t): return t.split("}",1)[-1]
def _fc(p,w):
    if p is None: return None
    for c in list(p):
        if _l(c.tag).lower()==w.lower(): return c
    return None
def _rnl(n):
    if n is None: return None
    vs=[]
    for c in list(n):
        if _l(c.tag).lower() in ("double","float","int","integer") and c.text and c.text.strip(): vs.append(float(c.text.strip()))
    if vs: return vs
    t="".join(n.itertext()).strip()
    if not t: return None
    o=[]
    for tk in t.replace(","," ").split():
        try: o.append(float(tk))
        except: pass
    return o if o else None
def _lb(th): th=np.array(th,dtype=float); return np.concatenate([[0],np.cumsum(th)[:-1]]),np.cumsum(th)
def _ov(lt,lb,z1,z2): return max(0,min(lb,z2)-max(lt,z1))
def _dwm(v,th,z1,z2):
    tp,bt=_lb(th); z2e=min(z2,float(bt[-1]) if len(bt) else 0)
    if z2e<=z1: return np.nan,0
    ws=vs=0
    for i in range(len(th)):
        o=_ov(tp[i],bt[i],z1,z2e)
        if o<=0: continue
        val=v[i] if i<len(v) else np.nan
        if np.isnan(val): continue
        vs+=val*o; ws+=o
    return (vs/ws if ws>0 else np.nan,z2e-z1)
def _dss(v,th,z1,z2):
    tp,bt=_lb(th); z2e=min(z2,float(bt[-1]) if len(bt) else 0)
    if z2e<=z1: return np.nan,0
    t=0; u=False
    for i in range(len(th)):
        o=_ov(tp[i],bt[i],z1,z2e)
        if o<=0: continue
        val=v[i] if i<len(v) else np.nan
        if np.isnan(val): continue
        t+=val*o/(bt[i]-tp[i]); u=True
    return (t if u else np.nan,z2e-z1)
def _pawc(th,dul,ll,z1,z2):
    tp,bt=_lb(th); z2e=min(z2,float(bt[-1]) if len(bt) else 0)
    if z2e<=z1: return np.nan,0
    t=0; u=False
    for i in range(len(th)):
        o=_ov(tp[i],bt[i],z1,z2e)
        if o<=0: continue
        d=dul[i] if i<len(dul) else np.nan; l=ll[i] if i<len(ll) else np.nan
        if np.isnan(d) or np.isnan(l): continue
        t+=(d-l)*o; u=True
    return (float(t) if u else np.nan,z2e-z1)

def build_soil_static(wheat_mask):
    if SOIL_STATIC_OUT.exists():
        print(f"[SOIL] Loading: {SOIL_STATIC_OUT}"); df=pd.read_parquet(SOIL_STATIC_OUT)
        df["_lr"]=np.round(df["lat"].astype(float),3); df["_lo"]=np.round(df["lon"].astype(float),3); return df
    print("[SOIL] Building ...")
    ds=xr.open_dataset(SOIL_PATH); sp,wm=xr.align(ds["soil_proportion"],wheat_mask,join="inner")
    spw=sp.where(wm>0); di=spw.fillna(-1).argmax("soil_id"); dom=ds["soil_id"].isel(soil_id=di)
    dom=dom.where((wm>0)&(spw.max("soil_id")>0))
    dd=dom.rename("dominant_soil_id").to_dataframe().reset_index()
    dd=dd[["lat","lon","dominant_soil_id"]].dropna(subset=["dominant_soil_id"])
    dd["dominant_soil_id"]=dd["dominant_soil_id"].astype(int)
    dd["lat"]=dd["lat"].astype(np.float32); dd["lon"]=dd["lon"].astype(np.float32)
    root=ET.parse(SOIL_APSIM).getroot(); tgt=set(int(x) for x in dd["dominant_soil_id"].unique().tolist()); rows=[]
    for nd in root.iter():
        if not _l(nd.tag).lower().endswith("soil"): continue
        nm=nd.attrib.get("name") or nd.attrib.get("Name") or ""
        if not nm: continue
        try: sid=int(re.findall(r"(\d+)$",nm)[0])
        except: continue
        if sid not in tgt: continue
        ok=any(_l(c.tag).lower()=="soilcrop" and (c.attrib.get("name") or c.attrib.get("Name") or "").lower()=="wheat" for c in nd.iter())
        if not ok: continue
        w=_fc(nd,"Water"); an=_fc(nd,"Analysis"); sa=_fc(nd,"Sample")
        th=_rnl(_fc(w,"Thickness")); dul=_rnl(_fc(w,"DUL")); ll=_rnl(_fc(w,"LL15")) or _rnl(_fc(w,"LL"))
        ph=_rnl(_fc(an,"PH")) or _rnl(_fc(an,"pH")) or _rnl(_fc(an,"PHWater"))
        no3=_rnl(_fc(sa,"NO3")) or _rnl(_fc(sa,"NO3N")); nh4=_rnl(_fc(sa,"NH4")) or _rnl(_fc(sa,"NH4N"))
        if not th: continue
        n=len(th); pad=lambda a:np.concatenate([np.array(a if a else [],dtype=float),np.full(max(0,n-len(a if a else [])),np.nan)])[:n]
        th=np.array(th,dtype=float); dul=pad(dul); ll=pad(ll); ph=pad(ph); no3=pad(no3); nh4=pad(nh4); mN=no3+nh4
        p030,_=_dwm(ph,th,0,300); p3060,_=_dwm(ph,th,300,600)
        n030,_=_dss(mN,th,0,300); n3060,_=_dss(mN,th,300,600)
        pw030,_=_pawc(th,dul,ll,0,300); pw060,_=_pawc(th,dul,ll,0,600)
        rows.append({"soil_id":sid,"pawc_0_30_mm":pw030,"pawc_0_60_mm":pw060,"profile_depth_cm":float(np.sum(th)/10),
                      "ph_0_30":p030,"ph_30_60":p3060,"minN_0_30":n030,"minN_30_60":n3060})
    met=pd.DataFrame(rows).drop_duplicates(subset=["soil_id"]).reset_index(drop=True)
    st=dd.merge(met,left_on="dominant_soil_id",right_on="soil_id",how="left").drop(columns=["soil_id"],errors="ignore")
    SOIL_STATIC_OUT.parent.mkdir(parents=True,exist_ok=True)
    st.to_parquet(SOIL_STATIC_OUT,index=False); print(f"[SOIL] Wrote: {SOIL_STATIC_OUT} ({len(st):,})")
    st["_lr"]=np.round(st["lat"].astype(float),3); st["_lo"]=np.round(st["lon"].astype(float),3); return st