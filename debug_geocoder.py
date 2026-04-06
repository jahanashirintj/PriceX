from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

def test_geocode(locality, city):
    try:
        geolocator = Nominatim(user_agent="hb_test_agent")
        location = geolocator.geocode(f"{locality}, {city}")
        if location:
            print(f"Success: {location.latitude}, {location.longitude}")
        else:
            print("Failed: No location found")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_geocode("Koramangala", "Bangalore")
    test_geocode("Bandra", "Mumbai")
