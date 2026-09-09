from main_backend import (
    get_all_stations,
    predict_from_map_coordinates,
    predict_from_map_coordinates_get,
    predict_from_explicit_features,
    MapClickRequest,
    DirectFeaturesRequest
)
import json

def run_tests():
    print("=" * 75)
    print("RUNNING DIRECT PIPELINE VERIFICATION FOR SIH FLASH FLOOD BACKEND")
    print("=" * 75)

    # 1. Get Stations
    stations_resp = get_all_stations()
    assert stations_resp["status"] == "success"
    print(f"[PASS] 1. GET /api/stations - OK ({stations_resp['count']} hilly basins loaded)")

    # 2. Test Map Click in Hilly Catchment (Shimla, Himachal Pradesh)
    # Shimla coordinates: 31.1048, 77.1734
    req1 = MapClickRequest(lat=31.1048, lon=77.1734, scenario="latest")
    data1 = predict_from_map_coordinates(req1)
    assert data1["status"] == "SUCCESS"
    assert data1["location"]["matched_basin"] == "Shimla"
    print(f"\n[PASS] 2. POST /api/predict/coordinates [Shimla, Himachal Pradesh]:")
    print(f"     Matched Basin: {data1['location']['matched_basin']} ({data1['location']['state']})")
    print(f"     Slope: {data1['topographical_profile']['slope_deg']} deg | Drainage: {data1['topographical_profile']['drainage_density']}")
    # 2b. Test Live Weather Scenario for Shimla
    req_live = MapClickRequest(lat=31.1048, lon=77.1734, scenario="live")
    data_live = predict_from_map_coordinates(req_live)
    assert data_live["status"] == "SUCCESS"
    print(f"\n[PASS] 2b. POST /api/predict/coordinates [Live Weather API Query]:")
    print(f"     Source: {data_live['hydrological_telemetry']['data_source']}")
    print(f"     Precipitation: {data_live['hydrological_telemetry']['rainfall_mm_hr']} mm/hr")
    print(f"     Predicted Risk: {data_live['prediction']['risk_level']} (Confidence: {data_live['prediction']['confidence']})")

    # 3. Test Cloudburst Scenario in Chamoli (Uttarakhand)
    # Chamoli coordinates: 30.40, 79.32
    req2 = MapClickRequest(lat=30.40, lon=79.32, scenario="cloudburst")
    data2 = predict_from_map_coordinates(req2)
    assert data2["status"] == "SUCCESS"
    print(f"\n[PASS] 3. POST /api/predict/coordinates [Chamoli - Extreme Cloudburst Scenario]:")
    print(f"     Rainfall Intensity: {data2['hydrological_telemetry']['rainfall_mm_hr']} mm/hr")
    print(f"     24hr Cum. Rainfall: {data2['hydrological_telemetry']['rainfall_cum_24hr']} mm")
    print(f"     Water Level: {data2['hydrological_telemetry']['water_level_m']}m / Danger: {data2['hydrological_telemetry']['danger_level_m']}m")
    print(f"     Predicted Risk: {data2['prediction']['risk_level']} ({data2['prediction']['alert_level']})")
    print(f"     Color Code: {data2['prediction']['color_code']}")
    print(f"     Advisory: {data2['disaster_management_advisory']}")
    assert data2["prediction"]["risk_level"] in ["High", "Severe"]

    # 4. Test Fair Weather in Munnar (Kerala - Western Ghats)
    # Munnar coordinates: 10.09, 77.06
    req3 = MapClickRequest(lat=10.09, lon=77.06, scenario="dry_normal")
    data3 = predict_from_map_coordinates(req3)
    assert data3["status"] == "SUCCESS"
    print(f"\n[PASS] 4. POST /api/predict/coordinates [Munnar, Kerala - Fair Weather]:")
    print(f"     Matched Basin: {data3['location']['matched_basin']} ({data3['location']['state']})")
    print(f"     Predicted Risk: {data3['prediction']['risk_level']} ({data3['prediction']['confidence']})")
    print(f"     Alert: {data3['prediction']['alert_level']}")
    assert data3["prediction"]["risk_level"] == "Low"

    # 5. Test Topographical Guardrail: Click in Flat Plains (New Delhi: 28.6139, 77.2090)
    req4 = MapClickRequest(lat=28.6139, lon=77.2090)
    data4 = predict_from_map_coordinates(req4)
    print(f"\n[PASS] 5. POST /api/predict/coordinates [New Delhi - Flat Plains Guardrail Test]:")
    print(f"     Status: {data4['status']}")
    print(f"     Is Hilly Region: {data4['is_hilly_region']}")
    print(f"     Risk Level: {data4['risk_level']}")
    print(f"     Distance to Nearest Hilly Basin: {data4['location']['distance_km']} km")
    print(f"     Guardrail Notice: {data4['message']}")
    assert data4["status"] == "GUARDRAIL_TRIGGERED"
    assert data4["risk_level"] == "N/A"

    # 6. Test GET Endpoint Wrapper (Cherrapunji, Meghalaya)
    data5 = predict_from_map_coordinates_get(lat=25.30, lon=91.73, scenario="heavy_monsoon")
    assert data5["status"] == "SUCCESS"
    print(f"\n[PASS] 6. GET /api/predict/coordinates [Cherrapunji, Meghalaya - Monsoon Surge]:")
    print(f"     Matched Basin: {data5['location']['matched_basin']} ({data5['location']['state']})")
    print(f"     Predicted Risk: {data5['prediction']['risk_level']} ({data5['prediction']['confidence']})")

    # 7. Test Direct Features Prediction
    req_feat = DirectFeaturesRequest(
        month=7,
        rainfall_mm_hr=90.0,
        rainfall_cum_3hr=190.0,
        rainfall_cum_24hr=350.0,
        rainfall_cum_72hr=480.0,
        water_level_ratio=2.2,
        water_level_rate_change=2.5,
        slope_deg=40.0,
        drainage_density=0.8,
        landslide_susceptibility="High"
    )
    data6 = predict_from_explicit_features(req_feat)
    assert data6["status"] == "SUCCESS"
    print(f"\n[PASS] 7. POST /api/predict/features [Direct Feature Vector]:")
    print(f"     Risk: {data6['prediction']['risk_level']} (Confidence: {data6['prediction']['confidence']})")

    print("\n" + "=" * 75)
    print("ALL TESTS PASSED! BACKEND LOGIC & INFERENCE PIPELINE VERIFIED!")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()
