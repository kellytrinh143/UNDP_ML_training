
from pathlib import Path
import os

# ── Data paths ──
DATA_ROOT    = Path(os.environ.get("DATA_ROOT", "/data"))
CLIMATE_ROOT = DATA_ROOT / "climate_data"
MASK_PATH    = DATA_ROOT / "make_wheat_mask" / "wheat_mask.nc"
YIELD_PATH   = (DATA_ROOT / "make_wheat_mask" /
                "DEWS_historical_1989-2024_wheat_yield_2025-02-05.nc")
SOIL_PATH    = DATA_ROOT / "soil" / "soils.nc"
SOIL_APSIM   = DATA_ROOT / "soil" / "DroughtSoils.apsim"
SM_DIR       = DATA_ROOT / "awo_aadi_grid"

# ── Output ──
OUT_DIR         = DATA_ROOT / "processed_multiscale"
SOIL_STATIC_OUT = OUT_DIR / "soil_static_features.parquet"

# ── Year range ──
START_YEAR = int(os.environ.get("START_YEAR", 1989))
END_YEAR   = int(os.environ.get("END_YEAR", 2021))
YEARS      = list(range(START_YEAR, END_YEAR + 1))

# ── Data loading ──
TIME_CHUNK = 365
MAX_POINTS = None   # None = all; int for debug
VARS       = ["tmin", "tmax", "rain", "radn"]


# ── Multi-scale temporal settings ──
N_PRE_SOW_MONTHS  = 6
N_POST_SOW_MONTHS = 9
WEEKLY_PRE_SOW    = 24       # weeks before sowing
WEEKLY_POST_SOW   = 36      # weeks from sowing onward

CONVERT_KG_TO_THA = True

N_TOTAL_MONTHS = N_PRE_SOW_MONTHS + N_POST_SOW_MONTHS

# -----------------------------------------------------------------------------
# APSIM-style thermal-time windows (page 4)
# -----------------------------------------------------------------------------
TT_SOW_TO_EMERG = 100
TT_EMERG_TO_ENDJUV = 400
TT_ENDJUV_TO_FI = 555
TT_FI_TO_FLOWER = 120
TT_FLOWER_TO_SGF = 650
TT_SGF_TO_EGF = 35
TT_EGF_TO_MATURITY = 200

CUM_EMERG = TT_SOW_TO_EMERG
CUM_ENDJUV = CUM_EMERG + TT_EMERG_TO_ENDJUV
CUM_FI = CUM_ENDJUV + TT_ENDJUV_TO_FI
CUM_FLOWER = CUM_FI + TT_FI_TO_FLOWER
CUM_SGF = CUM_FLOWER + TT_FLOWER_TO_SGF
CUM_EGF = CUM_SGF + TT_SGF_TO_EGF
CUM_MATURE = CUM_EGF + TT_EGF_TO_MATURITY

WINDOWS = {
    "W1_estab": (0, CUM_EMERG),
    "W2_veg": (CUM_EMERG, CUM_FI),
    "W3_preAnth": (CUM_FI, CUM_FLOWER),
    "W4_grainFill": (CUM_FLOWER, CUM_EGF),
    "W5_matur": (CUM_EGF, CUM_MATURE),
}


# ── Water stress thresholds ──
FW_PHOTO_THRESH = 0.5
FW_EXPAN_THRESH = 0.8

# ── DEWS sowing rules ──
NORTH_LAT = -32.24; NORTH_LON = 140.99; WEST_LON = 129.0
SOW_START_WEST = (4, 1); SOW_START_EAST = (4, 26); SOW_END = (7, 31)
SOW_RAIN3 = 15.0; SOW_PAW = 30.0; SOW_WET = 0.95

# ── HPC threading ──
N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK",
             os.environ.get("OMP_NUM_THREADS", "4")))
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[_v] = str(N_CPUS)