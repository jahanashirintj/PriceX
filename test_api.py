import requests
import json

payload = {
    "listing_type": "SALE",
    "area_sqft": 1500.0,
    "bhk": 3,
    "location_type": "URBAN",
    "locality": "Koramangala",
    "age_of_property": 5
}

try:
    response = requests.post("http://localhost:8000/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
