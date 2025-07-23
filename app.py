from flask import Flask, request, jsonify
from flask_cors import CORS
import googlemaps
from dotenv import load_dotenv
import os
import traceback
from bs4 import BeautifulSoup

app = Flask(__name__)

CORS(app, origins=["http://localhost:3000"])

# Load environment variables
load_dotenv()
GMAPS_API_KEY = os.getenv("GMAPS_API_KEY")
gmaps = googlemaps.Client(key=GMAPS_API_KEY)


def clean_html_instruction(html):
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


def get_location_details(lat, lng):
    # Reverse geocode to get state and country
    try:
        result = gmaps.reverese_geocode((lat, lng))
        if not result:
            raise ValueError("Could not deduce geocode location")

        state = None
        country = None
        for component in result[0]['address_components']:
            if 'administrative_area_level_1' in component['types']:
                state = component['long_name']
            if 'country' in component['types']:
                country = component['long_name']
        return state, country
    except Exception as e:
        raise ValueError(f"Reverse geocoding failed: {str(e)}")


def validate_destination(destination, state=None, country=None):
    # Ensure destination is in the same state or country.
    try:
        geocode_result = gmaps.geocode(destination)
        if not geocode_result:
            raise ValueError(f"Invalid destination: '{
                             destination}' not found.")

        # Check if any result matches the state or country
        for result in geocode_result:
            address_components = result['address_components']
            dest_state = None
            dest_country = None
            for component in address_components:
                if 'administrative_area_level_1' in component['types']:
                    dest_state = component['long_name']
                if 'country' in component['types']:
                    dest_country = component['long_name']

            # Prioritize state if provided, else country
            if state and dest_state == state:
                return result['formatted_address']
            if country and dest_country == country:
                return result['formatted_address']

        # If no match found, raise error
        raise ValueError(f"No destination found in {state or country}")
    except Exception as e:
        raise ValueError(f"Destination validation failed: {str(e)}")


def fetch_route_data(origin, destination, mode='driving', departure_time='now'):
    print(f"[DEBUG] Fetching route from '{origin}' to '{
          destination}' with mode '{mode}'")

    # Geocode check
    origin_geocode = gmaps.geocode(origin)
    destination_geocode = gmaps.geocode(destination)

    if not origin_geocode:
        raise ValueError(f"Invalid origin: '{origin}' not found.")
    if not destination_geocode:
        raise ValueError(f"Invalid destination: '{destination}' not found.")

    # Proceed to get directions
    directions = gmaps.directions(
        origin, destination, mode=mode, departure_time=departure_time, alternatives=True)

    distance_matrix = gmaps.distance_matrix(
        origins=[origin],
        destinations=[destination],
        mode=mode,
        departure_time=departure_time
    )

    result = {
        "routes": [],
        "distance": None,
        "duration": None,
        "duration_in_traffic": None,
        "traffic_severity": None
    }

    for route in directions:
        route_info = {
            "summary": route['summary'],
            "distance": route['legs'][0]['distance']['text'],
            "duration": route['legs'][0]['duration']['text'],
            "steps": [clean_html_instruction(step['html_instructions']) for step in route['legs'][0]['steps']]
        }
        result["routes"].append(route_info)

    element = distance_matrix['rows'][0]['elements'][0]
    result["distance"] = element['distance']['text']
    result["duration"] = element['duration']['text']
    if 'duration_in_traffic' in element:
        result["duration_in_traffic"] = element['duration_in_traffic']['text']
        normal_duration = element['duration']['value']
        traffic_duration = element['duration_in_traffic']['value']
        severity = "Low"
        if traffic_duration > normal_duration * 1.5:
            severity = "High"
        elif traffic_duration > normal_duration * 1.2:
            severity = "Medium"
        result["traffic_severity"] = severity

    return result


@app.route('/')
def hello():
    return "Welcome to TrafficChatter"


@app.route('/chat', methods=['POST'])
def handle_chat():
    try:
        data = request.get_json()
        print("Raw request data:", data)

        user_input = data.get('user_input') if data else None
        if not user_input:
            return jsonify({"error": "No user_input provided"}), 400

        lines = [line.strip()
                 for line in user_input.splitlines() if line.strip()]
        keys = ['origin', 'destination', 'mode']
        parsed_data = {keys[i]: lines[i]
                       for i in range(min(len(lines), len(keys)))}

        print("Parsed data:", parsed_data)

        origin = parsed_data.get('origin')
        destination = parsed_data.get('destination')
        mode = parsed_data.get('mode', 'driving')

        if not origin or not destination:
            return jsonify({"error": "Missing origin or destination"}), 400

        print(f"[DEBUG] Inputs — origin: {
              origin}, destination: {destination}, mode: {mode}")
        route_data = fetch_route_data(origin, destination, mode)
        return jsonify(route_data)

    except Exception as e:
        print("Error occurred:", str(e))
        traceback.print_exc()  # ← This prints the full stack trace
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
