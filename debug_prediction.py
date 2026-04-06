import requests
import json

def test_predict(area, bhk, locality):
    payload = {
        "listing_type": "SALE",
        "area_sqft": float(area),
        "bhk": int(bhk),
        "location_type": "URBAN",
        "locality": locality,
        "age_of_property": 5
    }
    response = requests.post("http://localhost:8000/predict", json=payload)
    return response.json()["predicted_price"]

areas = [800, 1500, 2500]
for a in areas:
    price = test_predict(a, 2, "Koramangala")
    print(f"Area: {a} -> Price: {price}")
