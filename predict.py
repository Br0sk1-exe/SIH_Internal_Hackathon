import joblib
import pandas as pd

final_model = joblib.load("flood_model.pkl")

FEATURE_ORDER = [
    "month", "rainfall_mm_hr", "rainfall_cum_3hr", "rainfall_cum_24hr",
    "rainfall_cum_72hr", "water_level_ratio", "water_level_rate_change",
    "slope_deg", "drainage_density", "landslide_susceptibility"
]

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Severe"}


def predict_flood_risk(features_dict):
    slope = features_dict["slope_deg"]
    drainage_density = features_dict["drainage_density"]

    # Guardrail: scoped to hilly/flash-flood-prone terrain only
    if slope < 8 or drainage_density < 0.15:
        return {
            "risk": "Not Applicable",
            "reason": "This system is designed for flash flood risk in hilly terrain. "
                      "The selected location's terrain profile falls outside hilly/flash-flood-prone "
                      "conditions — general river or urban flooding involves different mechanisms "
                      "not modeled here."
        }

    # Build as a DataFrame with matching column names (fixes the sklearn warning,
    # since the model was originally trained on a DataFrame, not a plain list)
    row = pd.DataFrame([features_dict], columns=FEATURE_ORDER)

    prediction = final_model.predict(row)[0]
    return {"risk": RISK_LABELS[int(prediction)]}


# ---- Sanity tests ----
if __name__ == "__main__":
    test_input = {
    
    "rainfall_mm_hr": 65.0,
    "rainfall_cum_3hr": 150.0,
    "rainfall_cum_24hr": 280.0,
    "rainfall_cum_72hr": 400.0,
    "water_level_ratio": 1.6,
    "water_level_rate_change": 1.1,
    "slope_deg": 42,
    "drainage_density": 0.9,
    "landslide_susceptibility": 2,
    "month": 7,
    }
    print("Low-risk-ish scenario result:", predict_flood_risk(test_input))

    test_input_2 = test_input.copy()
    test_input_2["water_level_ratio"] = 1.4
    test_input_2["water_level_rate_change"] = 0.9
    test_input_2["rainfall_mm_hr"] = 80
    print("High-risk-ish scenario result:", predict_flood_risk(test_input_2))

    test_input_3 = test_input.copy()
    test_input_3["slope_deg"] = 2
    test_input_3["drainage_density"] = 0.1
    print("Flat terrain (should be Not Applicable):", predict_flood_risk(test_input_3))