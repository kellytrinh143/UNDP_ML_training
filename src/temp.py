from pathlib import Path
import xarray as xr
import os
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

# Data root is already set in compose.yaml

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data"))
CLIMATE_ROOT = DATA_ROOT / "climate_data"
MASK_PATH   = DATA_ROOT / "make_wheat_mask" / "wheat_mask.nc"
YIELD_PATH = DATA_ROOT / "make_wheat_mask" / "DEWS_historical_1989-2024_wheat_yield_2025-02-05.nc"
SOIL_PATH   = DATA_ROOT / "soil" / "soils.nc"
DATA_ML_PATH = DATA_ROOT/"processed_multiscale"
OUT_DIR = DATA_ROOT / "processed_wheat"
OUT_DIR.mkdir(parents=True, exist_ok=True)

#zarr_path = OUT_DIR / "wheat_multiscale_features_2001.zarr"
#ds = xr.open_zarr(zarr_path)
#print(ds)
ml_path    = DATA_ML_PATH / f"wheat_multiscale_features_{2019}.parquet"



df= pd.read_parquet(ml_path)
print(df)
print(df.columns.tolist())



# Create yield classes
df["yield_class"] = pd.cut(
    df["wheat_yield"],
    bins=[-float("inf"), 1, 3.5, float("inf")],
    labels=["Low (<1 t/ha)", "Medium (1–3.5 t/ha)", "High (>3.5 t/ha)"]
)

# Check counts
print(df["yield_class"].value_counts())

# Plot
plt.figure(figsize=(10, 8))

colors = {
    "Low (<1 t/ha)": "red",
    "Medium (1–3.5 t/ha)": "orange",
    "High (>3.5 t/ha)": "green"
}

for cls, color in colors.items():
    subset = df[df["yield_class"] == cls]
    plt.scatter(
        subset["lon"],
        subset["lat"],
        s=8,
        c=color,
        label=cls,
        alpha=0.7
    )

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Australian Wheat Yield Classes")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("wheat_yield_map.png", dpi=300, bbox_inches="tight")
print("Saved plot to /workspace/src/wheat_yield_map.png")
# ds_yield = xr.open_dataset(YIELD_PATH)
# print(ds_yield)
# ds_soil = xr.open_dataset(SOIL_PATH)
# print(ds_soil)
# soil = ds_soil["soil_id"].values
# print(soil)
# unique_soil = np.unique(soil[~np.isnan(soil)])

# print(unique_soil)
# print("Count:", len(unique_soil))

# ds = xr.open_dataset(MASK_PATH)
# mask = ds["wheat_mask"]
# # Count grid points where wheat exists
# wheat_points = (mask == 1).sum().item()

# print("Number of wheat grid points:", wheat_points)