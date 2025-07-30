from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_cors import CORS
import googlemaps
from dotenv import load_dotenv
import os
import traceback
from bs4 import BeautifulSoup
import json
import g4f
from g4f.Provider import bing, You
import uuid


app = Flask(__name__)
app.secret_key = os.urandom(24)  # needed for session if used

CORS(app, origins=["http://localhost:3000", "https://traffichat.vercel.app"])
# Load environment variables
load_dotenv()
GMAPS_API_KEY = os.getenv("GMAPS_API_KEY")
gmaps = googlemaps.Client(key=GMAPS_API_KEY)
gpt_client = g4f.Client(
    Provider=[bing, You],
    headless=True
)

# In-memory session store
user_sessions = {}

# System prompt
BASE_SYSTEM_PROMPT = """You are a helpful assistant chatbot trained by OpenAI.
You answer questions, help users with travel/directions/traffic, or have friendly small talk.
You respond in a human-like, warm tone and you're always excited to help.
If a user requests routing, return JSON like:
{
  "origin": "...",
  "destination": "...",
  "mode": "driving"
}
Otherwise, just respond naturally in English.
"""


def clean_html_instruction(html):
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


def fetch_route_data(origin, destination, mode='driving', departure_time='now'):
    origin_geocode = gmaps.geocode(origin)
    destination_geocode = gmaps.geocode(destination)
    if not origin_geocode:
        raise ValueError(f"Invalid origin: '{origin}' not found.")
    if not destination_geocode:
        raise ValueError(f"Invalid destination: '{destination}' not found.")

    directions = gmaps.directions(
        origin, destination, mode=mode, departure_time=departure_time, alternatives=True)
    distance_matrix = gmaps.distance_matrix(
        origins=[origin], destinations=[
            destination], mode=mode, departure_time=departure_time
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
    return "Welcome to TrafficChatter!"


@app.route('/chat', methods=['POST'])
def handle_chat():
    global user_sessions
    try:
        data = request.get_json()
        user_input = data.get('user_input')
        session_id = data.get('session_id') or str(
            uuid.uuid4())  # frontend should send this
        lat = data.get('lat')
        lng = data.get('lng')

        if not user_input:
            return jsonify({"error": "No user_input provided"}), 400

        # 🔒 Per-session conversation history
        if session_id not in user_sessions:
            user_sessions[session_id] = [
                {"role": "system", "content": BASE_SYSTEM_PROMPT}]

        user_sessions[session_id].append(
            {"role": "user", "content": user_input})

        # Reverse geocode user location if provided
        user_location = None
        user_state = None
        user_country = None
        if lat is not None and lng is not None:
            try:
                geocode_result = gmaps.reverse_geocode((lat, lng))
                if geocode_result:
                    user_location = geocode_result[0]['formatted_address']
                    for comp in geocode_result[0]['address_components']:
                        if 'administrative_area_level_1' in comp['types']:
                            user_state = comp['long_name']
                        if 'country' in comp['types']:
                            user_country = comp['long_name']
            except Exception as geo_err:
                print("[WARN] Reverse geocode failed:", geo_err)

        # GPT Response (streamed chunks aggregated)
        stream = gpt_client.chat.completions.create(
            model="gpt-4o",
            messages=user_sessions[session_id],
            temperature=0.8,
            stream=True
        )

        gpt_output = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            gpt_output += delta

        user_sessions[session_id].append(
            {"role": "assistant", "content": gpt_output})

        # Try to extract JSON from response
        extracted_data = {}
        try:
            json_start = gpt_output.find('{')
            json_end = gpt_output.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_snippet = gpt_output[json_start:json_end]
                extracted_data = json.loads(json_snippet)
        except Exception:
            print("[WARN] No valid JSON in this GPT response.")

        # Fallback: use user location if origin missing
        origin = extracted_data.get('origin') or user_location
        destination = extracted_data.get('destination')
        mode = extracted_data.get('mode', 'driving')

        route_info = None
        if origin and destination:
            print(f"[ROUTE] From: {origin} To: {destination} Mode: {mode}")
            route_info = fetch_route_data(origin, destination, mode)

        return jsonify({
            "session_id": session_id,
            "response": gpt_output,
            "route": route_info,
            "raw_data": extracted_data,
            "used_origin": origin,
            "user_location": user_location,
            "state": user_state,
            "country": user_country
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
