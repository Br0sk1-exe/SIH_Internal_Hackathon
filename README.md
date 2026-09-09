# 🌊 SIH - Flash Flood Prediction in Hilly Regions

AI-driven geospatial early warning system for flash floods across India's mountainous watersheds (Himalayas & Western Ghats), built for the **Smart India Hackathon (SIH)**.

---

## 📌 Problem Statement
> **"Flash Flood Prediction in Hilly Regions"**  
Flash floods in steep terrains are characterized by rapid onset, intense overland runoff, and short lag times between cloudburst precipitation and river surges. This project combines LightGBM gradient-boosted decision trees, topographical catchments analysis, and real-time weather & hydrological telemetry to forecast flash flood hazards.

---

## 🏗️ Architecture & Features

```
  Judges / Frontend Map Click (lat, lon)
                     │
                     ▼
  ┌────────────────────────────────────────────────────────┐
  │         FastAPI Backend (main_backend.py)             │
  │                                                        │
  │  1. Spatial Matching (Haversine nearest-basin lookup)  │
  │  2. Topographical Guardrail Check                      │
  │     - Slope >= 8.0° & Drainage Density >= 0.15         │
  │     - Distance cutoff (150 km max to hilly catchment)  │
  │  3. Telemetry Ingestion (Live Weather API / Calibrated)│
  │  4. LightGBM ML Inference (flood_model.pkl)            │
  └────────────────────────────────────────────────────────┘
                     │
                     ▼
  ┌────────────────────────────────────────────────────────┐
  │                 Judge-Ready Response                   │
  │  • Risk Level: Low / Medium / High / Severe           │
  │  • Confidence % & Class Probabilities                  │
  │  • Topographic Profile (Slope, Drainage, Landslide)    │
  │  • Hydrological Gauges (Rainfall, Water Level Ratio)   │
  │  • NDRF / Disaster Management Actionable Advisory      │
  └────────────────────────────────────────────────────────┘
```

- **Live Weather & Hydrology Ingestion**: Supports real-time precipitation for clicked coordinates via weather APIs, with seamless fallback to dataset-calibrated telemetry so the demo never fails.
- **Topographical Guardrails**: Rejects non-hilly/plain terrain (e.g. New Delhi, Rajasthan) with informative domain boundary warnings explaining the physical differences between flash floods and riverine floods.
- **"What-If" Scenario Simulation**: Includes interactive presets (`cloudburst`, `heavy_monsoon`, `moderate_rain`, `dry_normal`) to demonstrate model reactions during presentations.
- **Built-in Map Testbed**: Serves an interactive Leaflet testbed directly at `http://127.0.0.1:8000/test-map`.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install fastapi uvicorn pydantic joblib lightgbm pandas requests
```

### 2. Run the Backend Server
```bash
python main_backend.py
```
*(Server starts at `http://127.0.0.1:8000`)*

### 3. Open the Interactive Visual Testbed
Visit:
> **[http://127.0.0.1:8000/test-map](http://127.0.0.1:8000/test-map)**

Click anywhere on the map of India to trigger instant predictions!

### 4. Interactive Swagger API Docs
Visit:
> **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

## 📡 API Endpoints

### 1. Map Coordinates Click (Primary Endpoint)
- **POST** `/api/predict/coordinates`
- **GET** `/api/predict/coordinates?lat=30.40&lon=79.32&scenario=cloudburst`

**Request Payload:**
```json
{
  "lat": 30.4000,
  "lon": 79.3200,
  "scenario": "live"
}
```
*Scenarios available:* `"live"`, `"latest"`, `"dry_normal"`, `"moderate_rain"`, `"heavy_monsoon"`, `"cloudburst"`.

**Sample Response:**
```json
{
  "status": "SUCCESS",
  "is_hilly_region": true,
  "location": {
    "clicked_lat": 30.4,
    "clicked_lon": 79.32,
    "matched_basin": "Chamoli",
    "state": "Uttarakhand",
    "distance_km": 0.01
  },
  "topographical_profile": {
    "slope_deg": 34.8,
    "drainage_density": 0.585,
    "landslide_susceptibility": "High"
  },
  "hydrological_telemetry": {
    "rainfall_mm_hr": 85.0,
    "rainfall_cum_24hr": 307.0,
    "water_level_m": 8.17,
    "danger_level_m": 3.8
  },
  "prediction": {
    "risk_level": "Severe",
    "alert_level": "EMERGENCY - RED ALERT",
    "confidence": "99.8%",
    "color_code": "#EF4444",
    "probabilities": {
      "Low": 0.0001,
      "Medium": 0.0002,
      "High": 0.0013,
      "Severe": 0.9984
    }
  },
  "disaster_management_advisory": "CRITICAL FLASH FLOOD IMMINENT! Immediate evacuation of floodplains and low-lying valleys required."
}
```

### 2. List Monitored Hilly Basins
- **GET** `/api/stations`
- Returns all 21 monitored mountain watershed stations across Himachal Pradesh, Uttarakhand, Sikkim, Meghalaya, Assam, West Bengal, Kerala, Tamil Nadu, and Karnataka.

### 3. Direct Feature Inference
- **POST** `/api/predict/features`
- Allows manual overrides and custom sliders.

---

## 🧪 Automated Testing
Run the automated test suite to verify all endpoints, guardrails, and ML predictions:
```bash
python test_backend.py
```