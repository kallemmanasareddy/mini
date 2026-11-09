
import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "project_name": "Highway Construction",
    "land_area_acres": 120,
    "number_of_parcels": 50,
    "affected_households": 100,
    "ownership_dispute_percentage": 20,
    "litigation_cases": 5,
    "land_records_verified_percentage": 80,
    "compensation_readiness_percentage": 60,
    "environmental_clearance_pending": 1,
    "forest_land_percentage": 10,
    "public_objections_count": 15,
    "utility_shifting_required": 1,
    "planned_duration_days": 365,
    "project_type": "Highway",
    "district": "Mysuru"
}

response = requests.post(url, json=data)

print("STATUS CODE:", response.status_code)
print("API RESPONSE:", response.json())
