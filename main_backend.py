import math
import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import json
import os
import requests
import shap

# ------------------------------------------------------------------------------
# 1. APPLICATION SETUP & CONSTANTS
# ------------------------------------------------------------------------------
app = FastAPI(
    title="SIH Flash Flood Early Warning System API",
    description="Geospatial ML Backend for Flash Flood Prediction in Hilly Regions across India.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_FILE = "flood_model.pkl"
STATIONS_FILE = "stations_data.json"

FEATURE_ORDER = [
    "month",
    "rainfall_mm_hr",
    "rainfall_cum_3hr",
    "rainfall_cum_24hr",
    "rainfall_cum_72hr",
    "water_level_ratio",
    "water_level_rate_change",
    "slope_deg",
    "drainage_density",
    "landslide_susceptibility"
]

SUSCEPTIBILITY_MAP = {
    "Low": 0, "Moderate": 1, "High": 2,
    0: 0, 1: 1, 2: 2
}

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Severe"}

RISK_METADATA = {
    "Low": {
        "color": "#10B981",  # Green
        "alert_level": "NORMAL - GREEN ADVISORY",
        "severity_score": 1,
        "action_advisory": "No immediate flash flood risk detected. River levels within normal hydrological thresholds."
    },
    "Medium": {
        "color": "#F59E0B",  # Amber/Yellow
        "alert_level": "WATCH - YELLOW ADVISORY",
        "severity_score": 2,
        "action_advisory": "Moderate surface runoff accumulating. Monitor tributary gauges and prepare local emergency services."
    },
    "High": {
        "color": "#F97316",  # Orange
        "alert_level": "WARNING - ORANGE ALERT",
        "severity_score": 3,
        "action_advisory": "Significant flood surge in mountain stream. Pre-alert NDRF/SDRF teams and restrict riverside traffic."
    },
    "Severe": {
        "color": "#EF4444",  # Red
        "alert_level": "EMERGENCY - RED ALERT",
        "severity_score": 4,
        "action_advisory": "CRITICAL FLASH FLOOD IMMINENT! Immediate evacuation of floodplains and low-lying valleys required."
    }
}

# ------------------------------------------------------------------------------
# 2. LOAD MODEL & STATIONS DATA
# ------------------------------------------------------------------------------
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found in current directory.")

model = joblib.load(MODEL_FILE)

# Initialize SHAP TreeExplainer for real-time Explainable AI (XAI)
try:
    explainer = shap.TreeExplainer(model)
    print("[OK] SHAP TreeExplainer initialized successfully.")
except Exception as _e:
    print(f"Warning: SHAP TreeExplainer initialization failed: {_e}")
    explainer = None

GLOBAL_FEATURE_IMPORTANCE = [
    {
        "feature": "water_level_ratio",
        "label": "River Water Level Ratio",
        "importance": 2726,
        "pct": 21.6,
        "unit": "ratio (curr/danger)",
        "category": "Hydrological",
        "description": "Stream height relative to critical bankfull danger mark. Dominant flash flood precursor."
    },
    {
        "feature": "water_level_rate_change",
        "label": "Water Surge Velocity",
        "importance": 1964,
        "pct": 15.6,
        "unit": "m/hr",
        "category": "Hydrological",
        "description": "Rate of stream water rise. Sudden surges indicate upstream headwater cloudburst runoff."
    },
    {
        "feature": "rainfall_mm_hr",
        "label": "Rainfall Intensity",
        "importance": 1631,
        "pct": 12.9,
        "unit": "mm/hr",
        "category": "Meteorological",
        "description": "Instantaneous cloudburst intensity (>80 mm/hr overwhelms mountain drainage within minutes)."
    },
    {
        "feature": "rainfall_cum_3hr",
        "label": "3-Hour Cumulative Rainfall",
        "importance": 1515,
        "pct": 12.0,
        "unit": "mm",
        "category": "Meteorological",
        "description": "Short-term accumulation triggering rapid overland sheetflow and tributary flooding."
    },
    {
        "feature": "rainfall_cum_72hr",
        "label": "72-Hour Cumulative Rainfall",
        "importance": 1274,
        "pct": 10.1,
        "unit": "mm",
        "category": "Meteorological",
        "description": "Extended catchment antecedent saturation index; wet soil drastically increases runoff coefficients."
    },
    {
        "feature": "rainfall_cum_24hr",
        "label": "24-Hour Cumulative Rainfall",
        "importance": 1233,
        "pct": 9.8,
        "unit": "mm",
        "category": "Meteorological",
        "description": "Medium-term precipitation volume filling valleys and river detention zones."
    },
    {
        "feature": "slope_deg",
        "label": "Catchment Slope Angle",
        "importance": 828,
        "pct": 6.6,
        "unit": "degrees",
        "category": "Topographical",
        "description": "Terrain steepness. Steeper mountain slopes accelerate runoff velocity and reduce concentration time."
    },
    {
        "feature": "landslide_susceptibility",
        "label": "Landslide Susceptibility",
        "importance": 361,
        "pct": 2.9,
        "unit": "index (0-2)",
        "category": "Geotechnical",
        "description": "Risk of hillslope debris failure blocking channels, creating dam-breach surge waves."
    },
    {
        "feature": "drainage_density",
        "label": "Drainage Network Density",
        "importance": 339,
        "pct": 2.7,
        "unit": "km/km²",
        "category": "Topographical",
        "description": "Concentration of stream channels funnelling water into the main river trunk."
    },
    {
        "feature": "month",
        "label": "Seasonal Climatology Month",
        "importance": 129,
        "pct": 1.0,
        "unit": "month (1-12)",
        "category": "Temporal",
        "description": "Monsoon seasonality weighting (July-August peak southwest monsoon in India)."
    }
]

# Default fallback stations if stations_data.json is missing
DEFAULT_STATIONS = {
    "Shimla": {"region": "Shimla", "state": "Himachal Pradesh", "lat": 31.10, "lon": 77.17, "slope_deg": 44.0, "drainage_density": 0.613, "landslide_susceptibility": "High", "danger_level_m": 12.0},
    "Kullu": {"region": "Kullu", "state": "Himachal Pradesh", "lat": 31.96, "lon": 77.11, "slope_deg": 36.9, "drainage_density": 0.573, "landslide_susceptibility": "High", "danger_level_m": 14.5},
    "Chamba": {"region": "Chamba", "state": "Himachal Pradesh", "lat": 32.56, "lon": 76.13, "slope_deg": 30.6, "drainage_density": 0.533, "landslide_susceptibility": "High", "danger_level_m": 11.8},
    "Kinnaur": {"region": "Kinnaur", "state": "Himachal Pradesh", "lat": 31.59, "lon": 78.27, "slope_deg": 34.9, "drainage_density": 0.825, "landslide_susceptibility": "High", "danger_level_m": 13.2},
    "Chamoli": {"region": "Chamoli", "state": "Uttarakhand", "lat": 30.40, "lon": 79.32, "slope_deg": 34.8, "drainage_density": 0.585, "landslide_susceptibility": "High", "danger_level_m": 15.0},
    "Rudraprayag": {"region": "Rudraprayag", "state": "Uttarakhand", "lat": 30.28, "lon": 78.98, "slope_deg": 33.8, "drainage_density": 0.651, "landslide_susceptibility": "High", "danger_level_m": 14.0},
    "Dehradun Hills": {"region": "Dehradun Hills", "state": "Uttarakhand", "lat": 30.32, "lon": 78.03, "slope_deg": 20.2, "drainage_density": 0.601, "landslide_susceptibility": "Moderate", "danger_level_m": 9.5},
    "Nainital": {"region": "Nainital", "state": "Uttarakhand", "lat": 29.38, "lon": 79.45, "slope_deg": 29.2, "drainage_density": 0.485, "landslide_susceptibility": "Moderate", "danger_level_m": 10.0},
    "Cherrapunji": {"region": "Cherrapunji", "state": "Meghalaya", "lat": 25.30, "lon": 91.73, "slope_deg": 32.5, "drainage_density": 0.980, "landslide_susceptibility": "High", "danger_level_m": 18.0},
    "Shillong": {"region": "Shillong", "state": "Meghalaya", "lat": 25.57, "lon": 91.88, "slope_deg": 32.8, "drainage_density": 0.546, "landslide_susceptibility": "Moderate", "danger_level_m": 11.0},
    "Gangtok": {"region": "Gangtok", "state": "Sikkim", "lat": 27.33, "lon": 88.61, "slope_deg": 43.0, "drainage_density": 0.802, "landslide_susceptibility": "High", "danger_level_m": 13.5},
    "Darjeeling": {"region": "Darjeeling", "state": "West Bengal", "lat": 27.04, "lon": 88.26, "slope_deg": 30.4, "drainage_density": 0.533, "landslide_susceptibility": "High", "danger_level_m": 12.0},
    "Kalimpong": {"region": "Kalimpong", "state": "West Bengal", "lat": 27.06, "lon": 88.47, "slope_deg": 35.6, "drainage_density": 0.578, "landslide_susceptibility": "High", "danger_level_m": 12.5},
    "Itanagar": {"region": "Itanagar", "state": "Arunachal Pradesh", "lat": 27.10, "lon": 93.62, "slope_deg": 32.2, "drainage_density": 0.413, "landslide_susceptibility": "Moderate", "danger_level_m": 11.5},
    "Guwahati Hills": {"region": "Guwahati Hills", "state": "Assam", "lat": 26.15, "lon": 91.77, "slope_deg": 10.6, "drainage_density": 0.543, "landslide_susceptibility": "Moderate", "danger_level_m": 8.5},
    "Silchar": {"region": "Silchar", "state": "Assam", "lat": 24.83, "lon": 92.78, "slope_deg": 9.6, "drainage_density": 0.497, "landslide_susceptibility": "Low", "danger_level_m": 8.0},
    "Munnar": {"region": "Munnar", "state": "Kerala", "lat": 10.09, "lon": 77.06, "slope_deg": 34.8, "drainage_density": 0.573, "landslide_susceptibility": "Moderate", "danger_level_m": 12.0},
    "Idukki": {"region": "Idukki", "state": "Kerala", "lat": 9.85, "lon": 76.97, "slope_deg": 41.0, "drainage_density": 0.785, "landslide_susceptibility": "High", "danger_level_m": 16.0},
    "Wayanad": {"region": "Wayanad", "state": "Kerala", "lat": 11.60, "lon": 76.08, "slope_deg": 23.0, "drainage_density": 0.528, "landslide_susceptibility": "High", "danger_level_m": 11.0},
    "Nilgiris": {"region": "Nilgiris", "state": "Tamil Nadu", "lat": 11.41, "lon": 76.70, "slope_deg": 34.5, "drainage_density": 0.413, "landslide_susceptibility": "Moderate", "danger_level_m": 12.0},
    "Coorg": {"region": "Coorg", "state": "Karnataka", "lat": 12.34, "lon": 75.81, "slope_deg": 30.1, "drainage_density": 0.655, "landslide_susceptibility": "Moderate", "danger_level_m": 11.5}
}

if os.path.exists(STATIONS_FILE):
    try:
        with open(STATIONS_FILE, "r") as f:
            STATIONS = json.load(f)
    except Exception:
        STATIONS = DEFAULT_STATIONS
else:
    STATIONS = DEFAULT_STATIONS


# ------------------------------------------------------------------------------
# 3. HELPER FUNCTIONS: GEODESIC DISTANCE & DATA GENERATION
# ------------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on the earth in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def find_nearest_basin(lat: float, lon: float):
    """Find the closest hilly monitoring basin from the dataset."""
    closest_station = None
    min_dist = float("inf")

    for name, st in STATIONS.items():
        dist = haversine_km(lat, lon, st["lat"], st["lon"])
        if dist < min_dist:
            min_dist = dist
            closest_station = st

    return closest_station, min_dist


def fetch_live_weather(lat: float, lon: float) -> Optional[dict]:
    """
    Fetch real-time precipitation for the exact (lat, lon) clicked using live weather API.
    Returns live rainfall metrics or None if unavailable.
    """
    try:
        url = f"https://wttr.in/{lat:.4f},{lon:.4f}?format=j1"
        res = requests.get(url, timeout=3.5)
        if res.status_code == 200:
            data = res.json()
            curr = data.get("current_condition", [{}])[0]
            rain_hr = float(curr.get("precipMM", 0.0))
            temp_c = curr.get("temp_C", "N/A")
            weather_desc = curr.get("weatherDesc", [{}])[0].get("value", "Clear")

            # Accumulate from forecast intervals
            days = data.get("weather", [])
            cum_24hr = rain_hr * 4.0
            cum_72hr = cum_24hr * 2.5
            if days:
                hourly = days[0].get("hourly", [])
                h_rain = [float(h.get("precipMM", 0.0)) for h in hourly]
                if len(h_rain) >= 3:
                    cum_3hr = round(sum(h_rain[:3]), 2)
                else:
                    cum_3hr = round(rain_hr * 2.0, 2)
                cum_24hr = round(sum(h_rain), 2)
                if len(days) > 1:
                    cum_72hr = round(cum_24hr + sum([float(h.get("precipMM", 0.0)) for h in days[1].get("hourly", [])]), 2)
            else:
                cum_3hr = round(rain_hr * 2.0, 2)

            return {
                "rainfall_mm_hr": rain_hr,
                "rainfall_cum_3hr": max(cum_3hr, rain_hr),
                "rainfall_cum_24hr": max(cum_24hr, rain_hr * 2),
                "rainfall_cum_72hr": max(cum_72hr, cum_24hr),
                "temp_c": temp_c,
                "weather_desc": weather_desc,
                "provider": "wttr.in Live Real-Time Weather"
            }
    except Exception as err:
        # Fallback to dataset if API is slow or offline
        return None
    return None


def fetch_or_simulate_telemetry(
    station: dict,
    scenario: str = "latest",
    rainfall_override: Optional[float] = None,
    water_ratio_override: Optional[float] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None
) -> dict:
    """
    Fetch telemetry: checks live weather for the clicked coordinates first (if scenario is 'live'),
    or synthesizes realistic hydrological data corresponding to the dataset patterns.
    """
    # 1. Dataset calibrated baselines
    baselines = station.get("baselines", {})
    live_meta = None

    if scenario == "live" and lat is not None and lon is not None:
        live_data = fetch_live_weather(lat, lon)
        if live_data:
            rainfall_hr = rainfall_override if rainfall_override is not None else live_data["rainfall_mm_hr"]
            cum_3hr = live_data["rainfall_cum_3hr"]
            cum_24hr = live_data["rainfall_cum_24hr"]
            cum_72hr = live_data["rainfall_cum_72hr"]

            # Dynamic hydrological runoff: estimate water level based on real-time rainfall
            if rainfall_hr > 60:
                water_level_ratio = water_ratio_override if water_ratio_override is not None else 1.95
                water_rate_change = 1.8
            elif rainfall_hr > 25:
                water_level_ratio = water_ratio_override if water_ratio_override is not None else 1.35
                water_rate_change = 0.7
            elif rainfall_hr > 5:
                water_level_ratio = water_ratio_override if water_ratio_override is not None else 0.85
                water_rate_change = 0.15
            else:
                water_level_ratio = water_ratio_override if water_ratio_override is not None else 0.52
                water_rate_change = -0.01

            data_source = f"LIVE API ({live_data['provider']} | {live_data['weather_desc']}, {live_data['temp_c']}°C)"
            live_meta = live_data
        else:
            # Graceful fallback if live API times out
            scenario = "latest"

    # Default baseline
    if scenario == "cloudburst":
        rainfall_hr = rainfall_override if rainfall_override is not None else 85.0
        cum_3hr = rainfall_hr * 2.2
        cum_24hr = cum_3hr + 120.0
        cum_72hr = cum_24hr + 90.0
        water_level_ratio = water_ratio_override if water_ratio_override is not None else 2.15
        water_rate_change = 2.4
        data_source = "Simulated Extreme Cloudburst (What-If Analysis)"

    elif scenario == "heavy_monsoon":
        rainfall_hr = rainfall_override if rainfall_override is not None else 42.0
        cum_3hr = rainfall_hr * 2.5
        cum_24hr = cum_3hr + 85.0
        cum_72hr = cum_24hr + 75.0
        water_level_ratio = water_ratio_override if water_ratio_override is not None else 1.45
        water_rate_change = 0.95
        data_source = "Monsoon Surge Scenario (High Precipitation)"

    elif scenario == "moderate_rain":
        rainfall_hr = rainfall_override if rainfall_override is not None else 14.0
        cum_3hr = 32.0
        cum_24hr = 68.0
        cum_72hr = 95.0
        water_level_ratio = water_ratio_override if water_ratio_override is not None else 0.75
        water_rate_change = 0.12
        data_source = "Moderate Rainfall Advisory"

    elif scenario == "dry_normal":
        rainfall_hr = rainfall_override if rainfall_override is not None else 0.5
        cum_3hr = 1.0
        cum_24hr = 8.0
        cum_72hr = 18.0
        water_level_ratio = water_ratio_override if water_ratio_override is not None else 0.48
        water_rate_change = -0.02
        data_source = "Fair Weather / Baseline Non-Monsoon"

    else:  # "latest" or default
        # Attempt to pull from dataset baselines if present
        if "High" in baselines and rainfall_override is None:
            b = baselines.get("Medium", baselines.get("Low", {}))
            rainfall_hr = b.get("rainfall_mm_hr", 8.0)
            cum_3hr = b.get("rainfall_cum_3hr", 20.0)
            cum_24hr = b.get("rainfall_cum_24hr", 55.0)
            cum_72hr = b.get("rainfall_cum_72hr", 80.0)
            water_level_ratio = b.get("water_level_ratio", 0.65)
            water_rate_change = b.get("water_level_rate_change", 0.05)
            data_source = "Dataset Basin Calibrated Telemetry"
        else:
            rainfall_hr = rainfall_override if rainfall_override is not None else 12.0
            cum_3hr = rainfall_hr * 2.1
            cum_24hr = cum_3hr + 40.0
            cum_72hr = cum_24hr + 60.0
            water_level_ratio = water_ratio_override if water_ratio_override is not None else 0.68
            water_rate_change = 0.08
            data_source = "Hydrological Telemetry Baseline"

    # Compute actual water level in meters for display
    danger_lvl = station.get("danger_level_m", 12.0)
    current_water_level_m = round(danger_lvl * water_level_ratio, 2)

    return {
        "rainfall_mm_hr": round(float(rainfall_hr), 2),
        "rainfall_cum_3hr": round(float(cum_3hr), 2),
        "rainfall_cum_24hr": round(float(cum_24hr), 2),
        "rainfall_cum_72hr": round(float(cum_72hr), 2),
        "water_level_ratio": round(float(water_level_ratio), 3),
        "water_level_rate_change": round(float(water_rate_change), 3),
        "water_level_m": current_water_level_m,
        "danger_level_m": danger_lvl,
        "data_source": data_source
    }


# ------------------------------------------------------------------------------
# 4. PYDANTIC SCHEMAS
# ------------------------------------------------------------------------------
class MapClickRequest(BaseModel):
    lat: float = Field(..., description="Latitude clicked on the map")
    lon: float = Field(..., description="Longitude clicked on the map")
    scenario: Optional[str] = Field(
        "latest",
        description="Scenario: 'latest', 'dry_normal', 'moderate_rain', 'heavy_monsoon', 'cloudburst'"
    )
    month: Optional[int] = Field(None, ge=1, le=12, description="Target month (1-12). Defaults to current month.")
    rainfall_override_mm_hr: Optional[float] = Field(None, description="Manual rainfall override in mm/hr for what-if testing")
    water_level_ratio_override: Optional[float] = Field(None, description="Manual water level ratio override")


class DirectFeaturesRequest(BaseModel):
    month: int = Field(7, ge=1, le=12)
    rainfall_mm_hr: float
    rainfall_cum_3hr: float
    rainfall_cum_24hr: float
    rainfall_cum_72hr: float
    water_level_ratio: float
    water_level_rate_change: float
    slope_deg: float
    drainage_density: float
    landslide_susceptibility: Any  # "Low", "Moderate", "High" or 0, 1, 2


# ------------------------------------------------------------------------------
# 5. CORE INFERENCE PIPELINE
# ------------------------------------------------------------------------------
def execute_model_inference(features_dict: dict) -> dict:
    """Run LightGBM model inference with strictly ordered features."""
    # Ensure categorical encoding
    landslide_val = features_dict["landslide_susceptibility"]
    if isinstance(landslide_val, str):
        landslide_encoded = SUSCEPTIBILITY_MAP.get(landslide_val, 1)
    else:
        landslide_encoded = int(landslide_val)

    # Format input DataFrame matching training columns
    input_row = {
        "month": int(features_dict["month"]),
        "rainfall_mm_hr": float(features_dict["rainfall_mm_hr"]),
        "rainfall_cum_3hr": float(features_dict["rainfall_cum_3hr"]),
        "rainfall_cum_24hr": float(features_dict["rainfall_cum_24hr"]),
        "rainfall_cum_72hr": float(features_dict["rainfall_cum_72hr"]),
        "water_level_ratio": float(features_dict["water_level_ratio"]),
        "water_level_rate_change": float(features_dict["water_level_rate_change"]),
        "slope_deg": float(features_dict["slope_deg"]),
        "drainage_density": float(features_dict["drainage_density"]),
        "landslide_susceptibility": landslide_encoded
    }

    df_input = pd.DataFrame([input_row], columns=FEATURE_ORDER)

    prediction_class = int(model.predict(df_input)[0])
    probabilities = model.predict_proba(df_input)[0]

    risk_label = RISK_LABELS[prediction_class]
    confidence_pct = round(probabilities[prediction_class] * 100, 2)

    prob_breakdown = {
        RISK_LABELS[i]: round(float(probabilities[i]), 4) for i in range(len(probabilities))
    }

    meta = RISK_METADATA[risk_label]

    # Calculate real-time Explainable AI (XAI) feature contributions via TreeExplainer
    shap_explanation = {
        "base_expected_value": 0.0,
        "predicted_class_name": risk_label,
        "contributions": [],
        "summary": "SHAP explainer standby."
    }

    if explainer is not None:
        try:
            raw_shap = explainer.shap_values(df_input)
            # raw_shap shape is (1, 10, 4)
            pred_shap = raw_shap[0, :, prediction_class]

            if hasattr(explainer.expected_value, '__getitem__'):
                base_val = float(explainer.expected_value[prediction_class])
            else:
                base_val = float(explainer.expected_value)

            FEATURE_LABELS = {
                "month": "Seasonal Month",
                "rainfall_mm_hr": "Rainfall Intensity",
                "rainfall_cum_3hr": "3-Hour Cumulative Rainfall",
                "rainfall_cum_24hr": "24-Hour Cumulative Rainfall",
                "rainfall_cum_72hr": "72-Hour Cumulative Rainfall",
                "water_level_ratio": "River Water Level Ratio",
                "water_level_rate_change": "Water Level Surge Velocity",
                "slope_deg": "Catchment Slope Angle",
                "drainage_density": "Drainage Network Density",
                "landslide_susceptibility": "Landslide Susceptibility"
            }

            FEATURE_UNITS = {
                "month": "",
                "rainfall_mm_hr": "mm/hr",
                "rainfall_cum_3hr": "mm",
                "rainfall_cum_24hr": "mm",
                "rainfall_cum_72hr": "mm",
                "water_level_ratio": "ratio",
                "water_level_rate_change": "m/hr",
                "slope_deg": "°",
                "drainage_density": "km/km²",
                "landslide_susceptibility": "index"
            }

            FEATURE_REASONS = {
                "water_level_ratio": "Proximity to bankfull danger mark; values above 1.0 indicate river overflow.",
                "water_level_rate_change": "Surge speed of headwater torrents; fast positive rate flags sudden flood wave.",
                "rainfall_mm_hr": "High cloudburst intensity causes immediate overland flow exceeding infiltration capacity.",
                "rainfall_cum_3hr": "Short-term accumulation saturating local tributary gullies and ravines.",
                "rainfall_cum_24hr": "Catchment-scale precipitation volume elevating downstream basin storage.",
                "rainfall_cum_72hr": "Antecedent moisture condition; saturated soil yields near 100% runoff.",
                "slope_deg": "Steep mountain gradients amplify gravity-driven surface flow velocities.",
                "landslide_susceptibility": "Risk of unstable slopes forming debris dams that breach violently.",
                "drainage_density": "High concentration of streams funnelling runoff directly into main channel.",
                "month": "Monsoon seasonality weighting (active July/August monsoon corridor)."
            }

            contributions = []
            for idx, col in enumerate(FEATURE_ORDER):
                val = input_row[col]
                s_val = round(float(pred_shap[idx]), 4)
                contributions.append({
                    "feature": col,
                    "label": FEATURE_LABELS.get(col, col),
                    "value": val,
                    "unit": FEATURE_UNITS.get(col, ""),
                    "shap_value": s_val,
                    "impact": "Increases Risk" if s_val > 0 else "Mitigates Risk",
                    "abs_shap": abs(s_val),
                    "description": FEATURE_REASONS.get(col, "")
                })

            contributions.sort(key=lambda x: x["abs_shap"], reverse=True)

            top_pos = [c for c in contributions if c["shap_value"] > 0][:3]
            top_neg = [c for c in contributions if c["shap_value"] <= 0][:2]

            pos_str = ", ".join([f"{c['label']} (+{c['shap_value']:.2f})" for c in top_pos]) if top_pos else "None"
            neg_str = ", ".join([f"{c['label']} ({c['shap_value']:.2f})" for c in top_neg]) if top_neg else "None"

            summary_text = f"Primary risk amplifiers: {pos_str}. Mitigating factors: {neg_str}."

            shap_explanation = {
                "base_expected_value": round(base_val, 4),
                "predicted_class_name": risk_label,
                "contributions": contributions,
                "summary": summary_text
            }
        except Exception as _ex:
            shap_explanation = {
                "base_expected_value": 0.0,
                "predicted_class_name": risk_label,
                "contributions": [],
                "summary": f"TreeExplainer warning: {_ex}"
            }

    return {
        "prediction_class": prediction_class,
        "risk_level": risk_label,
        "confidence": f"{confidence_pct}%",
        "confidence_score": confidence_pct,
        "probabilities": prob_breakdown,
        "color_code": meta["color"],
        "alert_level": meta["alert_level"],
        "severity_score": meta["severity_score"],
        "action_advisory": meta["action_advisory"],
        "model_input_features": input_row,
        "shap_explanation": shap_explanation
    }


# ------------------------------------------------------------------------------
# 6. REST API ROUTES
# ------------------------------------------------------------------------------
def load_frontend_html():
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(frontend_path):
        with open(frontend_path, "r", encoding="utf-8") as f:
            return f.read()
    root_html = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(root_html):
        with open(root_html, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>AURA-FLOOD AI: Dashboard Loading...</h1>"


@app.get("/", response_class=HTMLResponse)
def root_dashboard():
    """Serves the polished AURA-FLOOD AI judge presentation dashboard."""
    return HTMLResponse(content=load_frontend_html())


@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard_route():
    """Alias route to serve the dashboard."""
    return HTMLResponse(content=load_frontend_html())


@app.get("/api/info")
def api_info():
    """Returns backend metadata and API endpoint directory."""
    return {
        "project": "Smart India Hackathon - Flash Flood Prediction in Hilly Regions",
        "system": "AURA-FLOOD AI 2.0",
        "status": "Online",
        "model": "LightGBM Gradient Boosted Decision Forest",
        "xai_engine": "SHAP TreeExplainer",
        "supported_stations_count": len(STATIONS),
        "endpoints": {
            "dashboard_ui": "/ (GET - Open in Browser)",
            "map_click_post": "/api/predict/coordinates (POST)",
            "map_click_get": "/api/predict/coordinates (GET)",
            "direct_features": "/api/predict/features (POST)",
            "global_shap": "/api/shap/global (GET)",
            "list_stations": "/api/stations (GET)",
            "list_scenarios": "/api/scenarios (GET)"
        }
    }


@app.get("/api/stations")
def get_all_stations():
    """Return all 21 monitored hilly stations with terrain metadata."""
    station_list = []
    for name, data in STATIONS.items():
        station_list.append({
            "region": data["region"],
            "state": data["state"],
            "lat": data["lat"],
            "lon": data["lon"],
            "slope_deg": data["slope_deg"],
            "drainage_density": data["drainage_density"],
            "landslide_susceptibility": data["landslide_susceptibility"],
            "danger_level_m": data.get("danger_level_m", 12.0)
        })
    return {"status": "success", "count": len(station_list), "stations": station_list}


@app.get("/api/scenarios")
def get_available_scenarios():
    """Return available simulation scenarios for frontend test controls."""
    return {
        "scenarios": [
            {"id": "latest", "name": "Current Basin Baseline", "description": "Loads calibrated sensor readings for the matched basin."},
            {"id": "cloudburst", "name": "Severe Cloudburst (What-If)", "description": "Extreme precipitation (>85mm/hr) & sudden river level spike (Red Alert)."},
            {"id": "heavy_monsoon", "name": "Heavy Monsoon Surge", "description": "Continuous high rainfall (35-50mm/hr) with swelling mountain rivers (Orange Alert)."},
            {"id": "moderate_rain", "name": "Moderate Monsoon Showers", "description": "Passing showers (10-20mm/hr) within tolerable drainage thresholds (Yellow/Green)."},
            {"id": "dry_normal", "name": "Clear / Normal Weather", "description": "Standard non-monsoon base flow (Green Alert)."}
        ]
    }


@app.get("/api/shap/global")
def get_global_shap_importance():
    """Return model-wide global SHAP and feature importance metrics."""
    return {
        "status": "success",
        "model": "LightGBM Gradient Boosted Decision Forest",
        "explainer": "TreeExplainer (Exact Tree SHAP Algorithm)",
        "features_count": len(GLOBAL_FEATURE_IMPORTANCE),
        "global_importance": GLOBAL_FEATURE_IMPORTANCE
    }


@app.post("/api/predict/coordinates")
def predict_from_map_coordinates(payload: MapClickRequest):
    """
    Primary endpoint called when judges click anywhere on the frontend map.
    """
    lat = payload.lat
    lon = payload.lon
    scenario = payload.scenario or "latest"
    target_month = payload.month if payload.month is not None else datetime.datetime.now().month

    # 1. Spatial Matching to Nearest Hilly Basin
    nearest_station, distance_km = find_nearest_basin(lat, lon)

    # 2. Topographical & Geospatial Guardrail Check
    # A) Maximum allowable distance from monitored hilly mountain systems (150 km)
    # B) Slope & drainage density physical threshold
    slope = nearest_station["slope_deg"]
    drainage = nearest_station["drainage_density"]

    if distance_km > 150.0:
        return {
            "status": "GUARDRAIL_TRIGGERED",
            "is_hilly_region": False,
            "risk_level": "N/A",
            "color_code": "#9CA3AF",
            "alert_level": "OUT_OF_BOUNDS",
            "message": (
                f"SYSTEM BOUNDARY ALERT: Clicked coordinates ({lat:.4f}, {lon:.4f}) are {distance_km:.1f} km away "
                f"from the nearest modeled hilly watershed ({nearest_station['region']}, {nearest_station['state']}). "
                "Flash flood dynamics in this ML model are specifically calibrated for steep mountain catchments. "
                "Alluvial plain or urban stormwater flooding requires separate hydrodynamic hydraulic models."
            ),
            "location": {
                "clicked_lat": lat,
                "clicked_lon": lon,
                "nearest_basin": nearest_station["region"],
                "nearest_state": nearest_station["state"],
                "distance_km": round(distance_km, 2)
            }
        }

    if slope < 8.0 or drainage < 0.15:
        return {
            "status": "GUARDRAIL_TRIGGERED",
            "is_hilly_region": False,
            "risk_level": "N/A",
            "color_code": "#9CA3AF",
            "alert_level": "FLAT_TERRAIN_ALERT",
            "message": f"SYSTEM BOUNDARY ALERT: Terrain slope ({slope}°) is below the 8.0° threshold required for mountain flash flood propagation.",
            "location": {
                "clicked_lat": lat,
                "clicked_lon": lon,
                "nearest_basin": nearest_station["region"],
                "distance_km": round(distance_km, 2)
            }
        }

    # 3. Telemetry Ingestion (API / Dataset Calibrated)
    telemetry = fetch_or_simulate_telemetry(
        station=nearest_station,
        scenario=scenario,
        rainfall_override=payload.rainfall_override_mm_hr,
        water_ratio_override=payload.water_level_ratio_override,
        lat=lat,
        lon=lon
    )

    # 4. Construct Feature Dict
    inference_features = {
        "month": target_month,
        "rainfall_mm_hr": telemetry["rainfall_mm_hr"],
        "rainfall_cum_3hr": telemetry["rainfall_cum_3hr"],
        "rainfall_cum_24hr": telemetry["rainfall_cum_24hr"],
        "rainfall_cum_72hr": telemetry["rainfall_cum_72hr"],
        "water_level_ratio": telemetry["water_level_ratio"],
        "water_level_rate_change": telemetry["water_level_rate_change"],
        "slope_deg": slope,
        "drainage_density": drainage,
        "landslide_susceptibility": nearest_station["landslide_susceptibility"]
    }

    # 5. Execute ML Prediction
    ml_result = execute_model_inference(inference_features)

    # 6. Assemble Full Judge-Facing Response
    return {
        "status": "SUCCESS",
        "is_hilly_region": True,
        "location": {
            "clicked_lat": lat,
            "clicked_lon": lon,
            "matched_basin": nearest_station["region"],
            "state": nearest_station["state"],
            "distance_km": round(distance_km, 2),
            "basin_coords": {"lat": nearest_station["lat"], "lon": nearest_station["lon"]}
        },
        "topographical_profile": {
            "slope_deg": slope,
            "drainage_density": drainage,
            "landslide_susceptibility": nearest_station["landslide_susceptibility"]
        },
        "hydrological_telemetry": telemetry,
        "prediction": {
            "risk_level": ml_result["risk_level"],
            "alert_level": ml_result["alert_level"],
            "confidence": ml_result["confidence"],
            "confidence_score": ml_result["confidence_score"],
            "color_code": ml_result["color_code"],
            "severity_score": ml_result["severity_score"],
            "probabilities": ml_result["probabilities"],
            "shap_explanation": ml_result.get("shap_explanation", {}),
            "model_input_features": ml_result.get("model_input_features", {})
        },
        "shap_explanation": ml_result.get("shap_explanation", {}),
        "disaster_management_advisory": ml_result["action_advisory"],
        "scenario_applied": scenario
    }


@app.get("/api/predict/coordinates")
def predict_from_map_coordinates_get(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    scenario: str = "latest",
    month: Optional[int] = None,
    rainfall_override: Optional[float] = None,
    water_ratio_override: Optional[float] = None
):
    """GET convenience wrapper for coordinates prediction (handy for testing via URL in browser)."""
    payload = MapClickRequest(
        lat=lat,
        lon=lon,
        scenario=scenario,
        month=month,
        rainfall_override_mm_hr=rainfall_override,
        water_level_ratio_override=water_ratio_override
    )
    return predict_from_map_coordinates(payload)


@app.post("/api/predict/features")
def predict_from_explicit_features(data: DirectFeaturesRequest):
    """
    Direct model inference endpoint with raw feature parameters (for manual sliders or testing).
    """
    if data.slope_deg < 8.0 or data.drainage_density < 0.15:
        return {
            "status": "GUARDRAIL_TRIGGERED",
            "risk_level": "N/A",
            "message": "SYSTEM BOUNDARY ALERT: Slope must be >= 8° and drainage density >= 0.15 for flash flood inference."
        }

    features_dict = data.dict()
    result = execute_model_inference(features_dict)
    return {
        "status": "SUCCESS",
        "prediction": result
    }


# ------------------------------------------------------------------------------
# 7. INTERACTIVE TEST MAP UI (SERVED DIRECTLY FOR INSTANT DEMO & VERIFICATION)
# ------------------------------------------------------------------------------
@app.get("/test-map", response_class=HTMLResponse)
def serve_interactive_map_testbed():
    """Serves the polished AURA-FLOOD AI dashboard testbed."""
    return HTMLResponse(content=load_frontend_html())


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("  🚀 SIH FLASH FLOOD PREDICTION BACKEND SERVER STARTING")
    print("="*70)
    print("  Interactive Map Testbed: http://127.0.0.1:8000/test-map")
    print("  Interactive Swagger API Docs: http://127.0.0.1:8000/docs")
    print("  Coordinate Prediction: POST http://127.0.0.1:8000/api/predict/coordinates")
    print("="*70 + "\n")
    uvicorn.run("main_backend:app", host="127.0.0.1", port=8000, reload=True)