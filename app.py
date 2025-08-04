from flask import Flask, request, jsonify
from flask_cors import CORS
import googlemaps
from dotenv import load_dotenv
import os
import traceback
from bs4 import BeautifulSoup
import json
from openai import OpenAI
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)

CORS(app, origins=["http://localhost:3000", "https://traffichat.vercel.app"])

# Load environment variables
load_dotenv()
GMAPS_API_KEY = os.getenv("GMAPS_API_KEY")
OR_API_KEY = os.getenv("OR_API_KEY")

# Initialize clients
gmaps = googlemaps.Client(key=GMAPS_API_KEY)
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OR_API_KEY
)

# In-memory session store
user_sessions = {}

# System prompt
BASE_SYSTEM_PROMPT = """You are TrafficChatter, a helpful assistant chatbot specializing in travel, directions, and traffic information.
You respond in a human-like, warm tone and are always excited to help.

When a user requests routing information, first extract the origin, destination, and mode (driving, walking, bicycling, or transit; default to 'driving' if unspecified) from the input.
If route data is provided (e.g., from Google Maps API), incorporate it into your response by summarizing the key details (distance, duration, traffic severity, and main route steps) in a natural, friendly manner.
**Route Details:**
{route_data}

For non-route requests, provide travel tips, traffic advice, or friendly conversation in English. Be concise but helpful.
"""

def clean_html_instruction(html):
    #Clean HTML instructions from Google Maps API response#
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)

def fetch_route_data(origin, destination, mode='driving', departure_time='now'):
    #Fetch route data from Google Maps API#
    try:
        # Validate inputs Macmillan
        origin_geocode = gmaps.geocode(origin)
        destination_geocode = gmaps.geocode(destination)
        
        if not origin_geocode:
            raise ValueError(f"Invalid origin: '{origin}' not found.")
        if not destination_geocode:
            raise ValueError(f"Invalid destination: '{destination}' not found.")

        # Get directions and distance matrix
        directions = gmaps.directions(
            origin, destination, mode=mode, departure_time=departure_time, alternatives=True
        )
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

        # Process route alternatives
        for route in directions:
            route_info = {
                "summary": route['summary'],
                "distance": route['legs'][0]['distance']['text'],
                "duration": route['legs'][0]['duration']['text'],
                "steps": [clean_html_instruction(step['html_instructions']) for step in route['legs'][0]['steps']]
            }
            result["routes"].append(route_info)

        # Process main route data
        element = distance_matrix['rows'][0]['elements'][0]
        result["distance"] = element['distance']['text']
        result["duration"] = element['duration']['text']
        
        # Calculate traffic severity if available
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
    
    except Exception as e:
        print(f"Route fetch error: {str(e)}")
        raise

def get_openrouter_response(messages, temperature=0.8, model="openai/gpt-4o-mini"):
    #Get response from OpenRouter API with conversation history#
    try:
        chat_completion = openrouter_client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=1000
        )
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        print(f"OpenRouter API error: {str(e)}")
        return "I'm having trouble connecting to my AI service right now. Please try again in a moment."

def extract_route_json(text):
    #Extract JSON route data from AI response#
    try:
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_snippet = text[json_start:json_end]
            extracted_data = json.loads(json_snippet)
            if any(key in extracted_data for key in ['origin', 'destination']):
                return extracted_data
        return {}
    
    except Exception as e:
        print(f"JSON extraction error: {str(e)}")
        return {}

def get_user_location_info(lat, lng):
    #Get user location information from coordinates#
    try:
        geocode_result = gmaps.reverse_geocode((lat, lng))
        if not geocode_result:
            return None, None, None, ""
        
        user_location = geocode_result[0]['formatted_address']
        user_state = None
        user_country = None
        
        for comp in geocode_result[0]['address_components']:
            if 'administrative_area_level_1' in comp['types']:
                user_state = comp['long_name']
            if 'country' in comp['types']:
                user_country = comp['long_name']
        
        location_context = f"\n\nUser's current location: {user_location}"
        return user_location, user_state, user_country, location_context
    
    except Exception as geo_err:
        print(f"[WARN] Reverse geocode failed: {geo_err}")
        return None, None, None, ""

# Define a list of possible place types
PLACE_TYPES = ["church", "hospital", "restaurant", "school", "mosque", "pharmacy", "bank", "hotel", "gym", "mall", "bar", "cafe", "park", "gas station"]

def extract_place_type(user_input):
    user_input = user_input.lower()
    for place in PLACE_TYPES:
        if place in user_input:
            return place
    return None  # fallback if nothing is found

def fetch_nearby_places(lat, lng, place_type):
    # Fetch nearby places from Google Maps Places API
    try:
        places_result = gmaps.places_nearby(location=(lat, lng), radius=1500, type=place_type)
        
        if not places_result.get('results'):
            return f"No {place_type}s found nearby."

        places_info = []
        for place in places_result['results']:
            name = place.get('name')
            address = place.get('vicinity')
            places_info.append(f"{name} - {address}")

        return "Here are some nearby places:\n" + "\n".join(places_info)

    except Exception as e:
        print(f"Nearby places fetch error: {str(e)}")
        return "I'm having trouble finding nearby places right now. Please try again later."


@app.route('/')
def hello():
    return "Welcome to TrafficChatter!"

@app.route('/chat', methods=['POST'])
def handle_chat():
    global user_sessions
    try:
        data = request.get_json()
        user_input = data.get('user_input')
        session_id = data.get('session_id') or str(uuid.uuid4())
        lat = data.get('lat')
        lng = data.get('lng')
        model = data.get('model', 'openai/gpt-4o-mini')

        if not user_input:
            return jsonify({"error": "No user_input provided"}), 400

        # Initialize session conversation history
        if session_id not in user_sessions:
            user_sessions[session_id] = [
                {"role": "system", "content": BASE_SYSTEM_PROMPT}
            ]

        # Get user location information
        user_location, user_state, user_country, location_context = get_user_location_info(lat, lng)

        # Add user message with location context
        user_message = user_input + location_context
        user_sessions[session_id].append({"role": "user", "content": user_message})

        # Check if the user is asking for nearby places
        if "nearest" in user_input.lower() or "find" in user_input.lower():
            place_type = extract_place_type(user_input) or "church"
            nearby_places_response = fetch_nearby_places(lat, lng, place_type)
            user_sessions[session_id].append({"role": "assistant", "content": nearby_places_response})
            return jsonify({
                "session_id": session_id,
                "response": nearby_places_response,
                "user_location": user_location,
                "state": user_state,
                "country": user_country,
                "model_used": model
            })

        # Get initial AI response to parse route data
        ai_response = get_openrouter_response(user_sessions[session_id], model=model)
        user_sessions[session_id].append({"role": "assistant", "content": ai_response})

        # Extract route information
        extracted_data = extract_route_json(ai_response)
        origin = extracted_data.get('origin') or user_location
        destination = extracted_data.get('destination')
        mode = extracted_data.get('mode', 'driving')

        # Prepare response data
        response_data = {
            "session_id": session_id,
            "response": ai_response,
            "route": None,
            "raw_data": extracted_data,
            "used_origin": origin,
            "user_location": user_location,
            "state": user_state,
            "country": user_country,
            "model_used": model
        }

        # Fetch route information if we have origin and destination
        if origin and destination:
            try:
                print(f"[ROUTE] From: {origin} To: {destination} Mode: {mode}")
                route_info = fetch_route_data(origin, destination, mode)
                
                # Create a text summary of the route information
                route_summary = f"🚗 *Route found from {origin} to {destination}:*\n"
                route_summary += f"📏 Distance: {route_info.get('distance', 'N/A')}\n"
                route_summary += f"🕒 Duration: {route_info.get('duration', 'N/A')}\n"
                if route_info.get('duration_in_traffic'):
                    route_summary += f"🚦 In traffic: {route_info.get('duration_in_traffic', 'N/A')}\n"
                if route_info.get('traffic_severity'):
                    route_summary += f"📊 Severity: {route_info.get('traffic_severity', 'N/A')}\n"
                for idx, route in enumerate(route_info.get('routes', [])):
                    route_summary += f"\n📍 Route {idx + 1}: {route['summary']}\n"
                    route_summary += f"- Distance: {route['distance']}\n"
                    route_summary += f"- Duration: {route['duration']}\n"
                    route_summary += "- 🧭 Steps:\n"
                    for stepIdx, step in enumerate(route['steps']):
                        route_summary += f"  {stepIdx + 1}. {step}\n"
                # Add the route summary to the response data
                response_data["response"] = route_summary
                response_data["route"] = {
                    "distance": route_info["distance"],
                    "duration": route_info["duration"],
                    "duration_in_traffic": route_info.get("duration_in_traffic"),
                    "traffic_severity": route_info.get("traffic_severity"),
                    "routes": route_info["routes"]
                }
                # Add route summary to conversation history
                user_sessions[session_id].append({
                    "role": "system",
                    "content": route_summary
                })
                
            except Exception as route_err:
                print(f"[ROUTE ERROR] {str(route_err)}")
                response_data["route"] = {"error": str(route_err)}
                response_data["response"] = f"{ai_response}\n\n**Route Error:** {str(route_err)}"

        # Limit conversation history to prevent token overflow
        if len(user_sessions[session_id]) > 20:
            user_sessions[session_id] = [user_sessions[session_id][0]] + user_sessions[session_id][-18:]

        return jsonify(response_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/clear_session', methods=['POST'])
def clear_session():
    #Clear conversation history for a session#
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id and session_id in user_sessions:
            del user_sessions[session_id]
            return jsonify({"message": "Session cleared successfully"})
        
        return jsonify({"message": "Session not found or already clear"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/models', methods=['GET'])
def get_available_models():
    #Get list of available OpenRouter models#
    popular_models = [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-haiku",
        "google/gemini-pro-1.5",
        "meta-llama/llama-3.1-70b-instruct",
        "mistralai/mixtral-8x7b-instruct"
    ]
    
    return jsonify({
        "popular_models": popular_models,
        "default": "openai/gpt-4o-mini"
    })

@app.route('/health', methods=['GET'])
def health_check():
    #Health check endpoint#
    try:
        test_response = openrouter_client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="openai/gpt-4o-mini",
            max_tokens=5
        )
        
        return jsonify({
            "status": "healthy",
            "openrouter_connected": True,
            "gmaps_key_present": bool(GMAPS_API_KEY),
            "active_sessions": len(user_sessions),
            "test_response": test_response.choices[0].message.content if test_response.choices else "No response"
        })
    
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "openrouter_connected": False,
            "gmaps_key_present": bool(GMAPS_API_KEY),
            "active_sessions": len(user_sessions)
        }), 500

@app.route('/usage', methods=['GET'])
def get_usage_info():
    #Get current usage statistics#
    return jsonify({
        "active_sessions": len(user_sessions),
        "total_conversations": sum(len(session) for session in user_sessions.values()),
        "sessions": {sid: len(messages) for sid, messages in user_sessions.items()}
    })

if __name__ == '__main__':
    app.run(debug=True)