"""
config.py — Single source of truth for the street-environment
safety & comfort evaluation tool.

All API keys are read from environment variables / a local .env file
(see .env.example); none are stored here.
"""
import os
import numpy as np

# ===========================================================================
# Paths
# ===========================================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
OUTPUT_DIR   = os.path.join(BASE_DIR, "output")
OUTPUT_STATION_DIR = os.path.join(OUTPUT_DIR, "station")   # station-radius pipeline
CACHE_DIR    = os.path.join(BASE_DIR, "cache")
MODELS_DIR   = os.path.join(BASE_DIR, "models")

for _d in (OUTPUT_DIR, OUTPUT_STATION_DIR, CACHE_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------- DeepLabV3+ source repo & Cityscapes weights ----------
# The DeepLabV3+ implementation is vendored under ./DeepLabV3Plus (MIT; see its LICENSE).
# The pretrained weight file is NOT distributed here — see models/README.md.
REPO_DIR    = os.path.join(BASE_DIR, "DeepLabV3Plus")
WEIGHT_PATH = os.path.join(MODELS_DIR, "best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar")

# ---------- Input datasets (place under ./data — see data/README.md) ----------
DATA_CCTV    = os.path.join(DATA_DIR, "seoul_cctv.csv")
DATA_BELL    = os.path.join(DATA_DIR, "seoul_emergency_bell.csv")
DATA_LIGHT   = os.path.join(DATA_DIR, "seoul_security_light.csv")
DATA_GEOJSON = os.path.join(DATA_DIR, "seoul_dong.geojson")

# ===========================================================================
# Coordinate systems
# ===========================================================================
CRS_WGS = "EPSG:4326"   # geographic (lat/lng)
CRS_KTM = "EPSG:5179"   # Korea TM — metric distance/area math

# ===========================================================================
# Index weights
# ===========================================================================
# Physical Safety Index — weighted facility density (surveillance / response)
W_CCTV  = 2.0   # mechanical monitoring + recording
W_BELL  = 1.5   # emergency reporting / response
W_LIGHT = 1.0   # nighttime visibility (indirect surveillance)

# Visual Comfort Index — view-opening vs. view-blocking (negativity-weighted)
W_POS = 0.8     # openness elements (sky, vegetation, terrain, sidewalk)
W_NEG = 1.2     # enclosing elements (building, wall, fence)

# ===========================================================================
# Sampling
# ===========================================================================
SAMPLE_SPACING_M = 20     # street-view sampling interval along the road network (m)
BIDIRECTIONAL    = False  # sample both road directions per point
GRID_SIZE_M      = 200    # grid cell size (m) for optional grid-based aggregation

def get_station_out_dir():
    """Output folder for the station-radius pipeline (output/station/)."""
    os.makedirs(OUTPUT_STATION_DIR, exist_ok=True)
    return OUTPUT_STATION_DIR

# ===========================================================================
# Cityscapes 19-class taxonomy → view-opening / view-blocking / excluded
# ===========================================================================
CLASS_POS = {1: 'Sidewalk', 8: 'Vegetation', 9: 'Terrain', 10: 'Sky'}
CLASS_NEG = {2: 'Building', 3: 'Wall', 4: 'Fence', 5: 'Pole'}
CLASS_EXC = {
    0: 'Road', 6: 'TrafficLight', 7: 'TrafficSign',
    11: 'Person', 12: 'Rider', 13: 'Car', 14: 'Truck',
    15: 'Bus', 16: 'Train', 17: 'Motorcycle', 18: 'Bicycle',
}

# Overlay palette (Cityscapes colours)
CITYSCAPES_COLORS = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [220,  20,  60], [  0,   0, 142], [  0,   0, 142], [  0,   0, 142],
    [  0,   0, 142], [  0,   0, 142], [  0,   0, 142],
])

# ===========================================================================
# Safety–comfort matrix quadrants (2040 Seoul Plan-aligned evaluation typology)
# ===========================================================================
QUADRANTS = {
    "Q1": ("Stable",               "I_phy >= 0, I_per >= 0"),
    "Q2": ("Facility-Deficient",   "I_phy <  0, I_per >= 0"),
    "Q3": ("Compound-Vulnerable",  "I_phy <  0, I_per <  0"),
    "Q4": ("Enclosure-Dominant",   "I_phy >= 0, I_per <  0"),
}
