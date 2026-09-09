import math
import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import json
import os
import requests

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
        "model_input_features": input_row
    }


# ------------------------------------------------------------------------------
# 6. REST API ROUTES
# ------------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "project": "Smart India Hackathon - Flash Flood Prediction in Hilly Regions",
        "status": "Online",
        "model": "LightGBM Gradient Boosted Decision Forest",
        "supported_stations_count": len(STATIONS),
        "endpoints": {
            "map_click_post": "/api/predict/coordinates (POST)",
            "map_click_get": "/api/predict/coordinates (GET)",
            "direct_features": "/api/predict/features (POST)",
            "list_stations": "/api/stations (GET)",
            "list_scenarios": "/api/scenarios (GET)",
            "interactive_test_map": "/test-map (GET - Open in Browser)"
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
            "probabilities": ml_result["probabilities"]
        },
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
    """
    Serves a full-featured interactive web testbed with an India Terrain Map.
    Judges and developers can click anywhere on the map, trigger predictions,
    switch scenarios, and inspect live JSON and visual badges.
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIH - Flash Flood Prediction Map Testbed</title>
    <!-- Leaflet CSS & JS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { display: flex; height: 100vh; overflow: hidden; background: #0f172a; color: #f8fafc; }
        #map { flex: 1; height: 100%; z-index: 1; }
        #sidebar {
            width: 440px;
            background: #1e293b;
            border-left: 1px solid #334155;
            display: flex;
            flex-direction: column;
            padding: 20px;
            gap: 16px;
            overflow-y: auto;
            z-index: 10;
            box-shadow: -4px 0 20px rgba(0,0,0,0.5);
        }
        h1 { font-size: 1.15rem; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }
        .subtitle { font-size: 0.8rem; color: #94a3b8; margin-top: -10px; }
        .card { background: #0f172a; border-radius: 8px; border: 1px solid #334155; padding: 14px; }
        .card-title { font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 700; margin-bottom: 8px; letter-spacing: 0.5px; }
        .badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.9rem;
            color: #fff;
            text-align: center;
        }
        .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.82rem; }
        .metric { background: #1e293b; padding: 8px; border-radius: 6px; }
        .metric-label { color: #94a3b8; font-size: 0.72rem; }
        .metric-value { font-weight: 600; color: #f1f5f9; margin-top: 2px; }
        .btn-group { display: flex; flex-wrap: wrap; gap: 6px; }
        button {
            background: #334155; color: #f8fafc; border: none; padding: 8px 12px;
            border-radius: 6px; font-size: 0.78rem; font-weight: 600; cursor: pointer;
            transition: all 0.2s ease;
        }
        button:hover { background: #475569; }
        button.active { background: #0284c7; color: white; box-shadow: 0 0 10px rgba(2, 132, 199, 0.5); }
        .alert-box { border-left: 4px solid #ef4444; background: rgba(239, 68, 68, 0.1); padding: 10px; border-radius: 4px; font-size: 0.82rem; }
        .prob-bar { height: 6px; border-radius: 3px; background: #334155; margin-top: 4px; overflow: hidden; }
        .prob-fill { height: 100%; transition: width 0.3s; }
        pre { background: #090d16; padding: 10px; border-radius: 6px; font-size: 0.7rem; color: #a5b4fc; overflow-x: auto; max-height: 180px; }
        .hint { font-size: 0.75rem; color: #38bdf8; background: rgba(56, 189, 248, 0.1); padding: 8px; border-radius: 6px; }
    </style>
</head>
<body>
    <div id="map"></div>
    <div id="sidebar">
        <div>
            <h1>⚡ SIH Flash Flood Predictor</h1>
            <p class="subtitle">Hilly Regions Early Warning AI Backend</p>
        </div>

        <div class="hint">
            📍 <b>Instructions:</b> Click anywhere on the map! Try clicking in <b>Himachal, Uttarakhand, Sikkim, Meghalaya, or Kerala</b> (hilly catchments) vs <b>Delhi or Rajasthan</b> (boundary guardrail alert).
        </div>

        <div class="card">
            <div class="card-title">Test Simulation Scenario</div>
            <div class="btn-group" id="scenario-buttons">
                <button class="active" onclick="setScenario('live', this)">🌐 Live Weather API</button>
                <button onclick="setScenario('latest', this)">Baseline Sensor</button>
                <button onclick="setScenario('dry_normal', this)">Fair Weather</button>
                <button onclick="setScenario('moderate_rain', this)">Moderate Rain</button>
                <button onclick="setScenario('heavy_monsoon', this)">Heavy Monsoon</button>
                <button onclick="setScenario('cloudburst', this)">🚨 Cloudburst</button>
            </div>
        </div>

        <div id="result-container" style="display: none; display: flex; flex-direction: column; gap: 14px;">
            <div class="card" id="status-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div class="card-title" style="margin: 0;">Predicted Flash Flood Risk</div>
                    <span id="risk-badge" class="badge">Low</span>
                </div>
                <div id="alert-title" style="font-weight: 700; font-size: 0.88rem; margin-bottom: 4px;">NORMAL</div>
                <div id="advisory-text" style="font-size: 0.8rem; color: #cbd5e1;"></div>
            </div>

            <div class="card">
                <div class="card-title">Geographic & Topographic Profile</div>
                <div class="metric-grid">
                    <div class="metric">
                        <div class="metric-label">Matched Basin</div>
                        <div class="metric-value" id="res-basin">-</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">State</div>
                        <div class="metric-value" id="res-state">-</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Terrain Slope</div>
                        <div class="metric-value" id="res-slope">-</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Drainage Density</div>
                        <div class="metric-value" id="res-drainage">-</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Landslide Vulnerability</div>
                        <div class="metric-value" id="res-landslide">-</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Distance from Click</div>
                        <div class="metric-value" id="res-distance">-</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Hydrological & Rainfall Telemetry</div>
                <div class="metric-grid">
                    <div class="metric">
                        <div class="metric-label">Rainfall Rate</div>
                        <div class="metric-value" id="res-rain-hr">-</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">24h Cum. Rainfall</div>
                        <div class="metric-value" id="res-rain-24h">-</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Water Level / Danger</div>
                        <div class="metric-value" id="res-water-level">-</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Level Rate of Change</div>
                        <div class="metric-value" id="res-rate-change">-</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Confidence & Probabilities</div>
                <div style="font-size: 0.82rem; margin-bottom: 8px;">
                    Model Confidence: <b id="res-confidence" style="color: #38bdf8;">0%</b>
                </div>
                <div id="prob-container" style="display: flex; flex-direction: column; gap: 6px; font-size: 0.75rem;"></div>
            </div>

            <div class="card">
                <div class="card-title">API Response Payload (JSON)</div>
                <pre id="json-preview"></pre>
            </div>
        </div>

        <div id="placeholder-prompt" class="card" style="text-align: center; color: #64748b; padding: 40px 10px;">
            🗺️ Click anywhere on the map of India to query the ML model.
        </div>
    </div>

    <script>
        // Initialize Map centered on Northern India
        const map = L.map('map').setView([28.5, 82.0], 5);

        // ESRI World Topo Map for realistic mountain terrain visualization
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
            maxZoom: 18
        }).addTo(map);

        let currentScenario = 'live';
        let currentMarker = null;
        let lastLat = null;
        let lastLon = null;

        function setScenario(sc, btn) {
            currentScenario = sc;
            document.querySelectorAll('#scenario-buttons button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (lastLat && lastLon) {
                queryPrediction(lastLat, lastLon);
            }
        }

        // Fetch stations to plot markers
        fetch('/api/stations')
            .then(res => res.json())
            .then(data => {
                if (data.stations) {
                    data.stations.forEach(st => {
                        const circle = L.circleMarker([st.lat, st.lon], {
                            radius: 7,
                            fillColor: "#0284c7",
                            color: "#ffffff",
                            weight: 2,
                            opacity: 1,
                            fillOpacity: 0.85
                        }).addTo(map);

                        circle.bindTooltip(`<b>${st.region}, ${st.state}</b><br>Slope: ${st.slope_deg}° | Landslide: ${st.landslide_susceptibility}`, {
                            direction: 'top'
                        });

                        circle.on('click', (e) => {
                            L.DomEvent.stopPropagation(e);
                            queryPrediction(st.lat, st.lon);
                        });
                    });
                }
            });

        // Map Click Listener
        map.on('click', (e) => {
            const lat = e.latlng.lat;
            const lon = e.latlng.lng;
            queryPrediction(lat, lon);
        });

        async function queryPrediction(lat, lon) {
            lastLat = lat;
            lastLon = lon;

            if (currentMarker) {
                map.removeLayer(currentMarker);
            }
            currentMarker = L.marker([lat, lon]).addTo(map);

            document.getElementById('placeholder-prompt').style.display = 'none';
            document.getElementById('result-container').style.display = 'flex';

            try {
                const response = await fetch('/api/predict/coordinates', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        lat: lat,
                        lon: lon,
                        scenario: currentScenario
                    })
                });

                const data = await response.json();
                document.getElementById('json-preview').innerText = JSON.stringify(data, null, 2);

                if (data.status === 'GUARDRAIL_TRIGGERED') {
                    document.getElementById('risk-badge').style.background = '#64748b';
                    document.getElementById('risk-badge').innerText = 'N/A';
                    document.getElementById('alert-title').innerText = 'BOUNDARY GUARDRAIL ALERT';
                    document.getElementById('alert-title').style.color = '#ef4444';
                    document.getElementById('advisory-text').innerText = data.message;

                    document.getElementById('res-basin').innerText = data.location.nearest_basin || 'None';
                    document.getElementById('res-state').innerText = data.location.nearest_state || 'Out of Catchment';
                    document.getElementById('res-distance').innerText = (data.location.distance_km || 0) + ' km';
                    document.getElementById('res-slope').innerText = 'N/A';
                    document.getElementById('res-drainage').innerText = 'N/A';
                    document.getElementById('res-landslide').innerText = 'N/A';

                    document.getElementById('res-rain-hr').innerText = 'N/A';
                    document.getElementById('res-rain-24h').innerText = 'N/A';
                    document.getElementById('res-water-level').innerText = 'N/A';
                    document.getElementById('res-rate-change').innerText = 'N/A';
                    document.getElementById('res-confidence').innerText = 'N/A';
                    document.getElementById('prob-container').innerHTML = '';
                    return;
                }

                // Render Success Prediction
                const pred = data.prediction;
                const badge = document.getElementById('risk-badge');
                badge.innerText = pred.risk_level;
                badge.style.background = pred.color_code;

                document.getElementById('alert-title').innerText = pred.alert_level;
                document.getElementById('alert-title').style.color = pred.color_code;
                document.getElementById('advisory-text').innerText = data.disaster_management_advisory;

                document.getElementById('res-basin').innerText = data.location.matched_basin;
                document.getElementById('res-state').innerText = data.location.state;
                document.getElementById('res-distance').innerText = data.location.distance_km + ' km';
                document.getElementById('res-slope').innerText = data.topographical_profile.slope_deg + '°';
                document.getElementById('res-drainage').innerText = data.topographical_profile.drainage_density;
                document.getElementById('res-landslide').innerText = data.topographical_profile.landslide_susceptibility;

                const hydro = data.hydrological_telemetry;
                document.getElementById('res-rain-hr').innerText = hydro.rainfall_mm_hr + ' mm/hr';
                document.getElementById('res-rain-24h').innerText = hydro.rainfall_cum_24hr + ' mm';
                document.getElementById('res-water-level').innerText = hydro.water_level_m + 'm / ' + hydro.danger_level_m + 'm (' + hydro.water_level_ratio + 'x)';
                document.getElementById('res-rate-change').innerText = (hydro.water_level_rate_change > 0 ? '+' : '') + hydro.water_level_rate_change + ' m/hr';

                document.getElementById('res-confidence').innerText = pred.confidence;
                document.getElementById('res-confidence').style.color = pred.color_code;

                // Probability breakdown
                let probHtml = '';
                const colors = { 'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#F97316', 'Severe': '#EF4444' };
                for (const [lvl, prob] of Object.entries(pred.probabilities)) {
                    const pct = Math.round(prob * 100);
                    probHtml += `
                        <div>
                            <div style="display:flex; justify-content:space-between;">
                                <span>${lvl}</span>
                                <span>${pct}%</span>
                            </div>
                            <div class="prob-bar">
                                <div class="prob-fill" style="width: ${pct}%; background: ${colors[lvl] || '#0284c7'};"></div>
                            </div>
                        </div>
                    `;
                }
                document.getElementById('prob-container').innerHTML = probHtml;

            } catch (err) {
                console.error("API error:", err);
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


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