# 🌊 AURA-FLOOD AI — SIH Flash Flood Early Warning & Explainable AI (XAI) System

AI-driven geospatial early warning system for flash floods across India's mountainous watersheds (Himalayas & Western Ghats), built for the **Smart India Hackathon (SIH)**.

---

## 📌 Problem Statement
> **"Flash Flood Prediction in Hilly Regions"**  
Flash floods in steep terrains are characterized by rapid onset, intense overland runoff, and short lag times between cloudburst precipitation and river surges. This project combines LightGBM gradient-boosted decision trees, topographical catchments analysis, real-time weather & hydrological telemetry, and **TreeSHAP Explainable AI** to forecast flash flood hazards.

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
  │  5. Real-Time SHAP Attribution (TreeExplainer)         │
  └────────────────────────────────────────────────────────┘
                     │
                     ▼
  ┌────────────────────────────────────────────────────────┐
  │                 Judge-Ready Response                   │
  │  • Risk Level: Low / Medium / High / Severe           │
  │  • Confidence % & Class Probabilities                  │
  │  • SHAP Waterfall & Diverging Feature Weights          │
  │  • Topographic Profile (Slope, Drainage, Landslide)    │
  │  • Hydrological Gauges (Rainfall, Water Level Ratio)   │
  │  • NDRF / Disaster Management Actionable Advisory      │
  └────────────────────────────────────────────────────────┘
```

- **Polished Judge-Ready UI (`frontend/index.html`)**: Dark command cockpit with CartoDB Leaflet map, pulsing hazard beacons, dynamic river liquid level column, What-If slider playground, and government incident bulletin printer.
- **Explainable AI (TreeSHAP)**: Explains *why* the model made each prediction with exact feature contributions ($\phi_i$), natural language executive summaries, and global feature importance rankings.
- **Live Weather & Hydrology Ingestion**: Real-time precipitation for clicked coordinates via weather APIs with dataset baseline fallback.
- **Topographical Guardrails**: Rejects non-hilly/plain terrain (e.g., New Delhi, Rajasthan) with domain boundary warnings explaining physical differences between flash floods and riverine floods.
- **"What-If" Scenario Simulation**: Includes interactive presets (`cloudburst`, `heavy_monsoon`, `moderate_rain`, `dry_normal`) and interactive sliders for real-time judge testing.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install fastapi uvicorn pydantic joblib lightgbm pandas requests shap
```

### 2. Run the Full Stack Application
```bash
python main_backend.py
```
*(Server starts at `http://127.0.0.1:8000`)*

### 3. Open the Frontend Dashboard
Visit in your web browser:
> **[http://127.0.0.1:8000](http://127.0.0.1:8000)** *(or [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard))*

*Alternatively, double-click [`index.html`](index.html) in the root directory.*

### 4. Interactive Swagger API Docs
Visit:
> **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

## 📡 Key API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive AURA-FLOOD AI dashboard. |
| `POST` | `/api/predict/coordinates` | Primary map-click inference endpoint returning hazard level + SHAP explanations. |
| `GET` | `/api/predict/coordinates` | GET convenience wrapper for coordinate lookup via URL. |
| `POST` | `/api/predict/features` | Direct raw feature inference for custom slider simulations. |
| `GET` | `/api/shap/global` | Model-wide global SHAP feature importance rankings. |
| `GET` | `/api/stations` | List of all 21 monitored mountain watersheds with terrain metadata. |
| `GET` | `/api/scenarios` | Simulation scenario presets. |

---

## 🧪 Verification & Automated Testing

Run the direct pipeline test suite:
```bash
python test_backend.py
```
*Executes all 10 unit and integration tests verifying coordinate lookups, cloudburst triggers, live weather queries, flatland guardrails, SHAP explanations, and frontend template serving.*