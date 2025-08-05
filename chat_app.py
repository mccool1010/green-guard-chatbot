from flask import Flask, request, jsonify, render_template
import os
import google.generativeai as genai
import requests
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "http://127.0.0.1:5500", "methods": ["POST", "OPTIONS"]}})

# Configure Gemini API
genai.configure(api_key="AIzaSyBAgXgwsXv-EKFaHvJInXu3X35Qif-skzo")

generation_config = {
    "temperature": 0.65,
    "top_p": 0.85,
    "top_k": 30,
    "max_output_tokens": 1024,
}

try:
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    print("Gemini model initialized successfully")
except Exception as e:
    print(f"Failed to initialize Gemini model: {e}")
    print(traceback.format_exc())

IMAGE_CLASSIFICATION_URL = "http://127.0.0.1:5000/predict"

DISEASE_RESOURCES = {
    "Alternaria Leaf Spot": "https://example.com/alternaria-guide",
    "Cercospora Leaf Spot": "https://example.com/cercospora-guide",
    "Downy Mildew": "https://example.com/mildew-guide",
    "Leaf Curl Virus": "https://example.com/leaf-curl-handbook",
    "Bhendi Yellow Vein Mosaic Disease": "https://example.com/mosaic-guide",
    "Phyllosticta Leaf Spot": "https://example.com/phyllosticta-guide"
}

CLASS_MAPPING = {
    0: "Alternaria Leaf Spot",
    1: "Cercospora Leaf Spot",
    2: "Downy Mildew",
    3: "Healthy",
    4: "Leaf Curl Virus",
    5: "Phyllosticta Leaf Spot",
    6: "Bhendi Yellow Vein Mosaic Disease"
}

conversation_context = {
    'current_disease': None,
    'conversation_history': [],
    'last_plant_focus': None
}

def classify_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            response = requests.post(
                IMAGE_CLASSIFICATION_URL,
                files={"image": image_file},
                timeout=10
            )
            
        if response.status_code == 200:
            result = response.json()
            print("Raw classification response:", result)
            
            if 'predicted_disease' in result:
                return result['predicted_disease']
            elif 'class_id' in result:
                return CLASS_MAPPING.get(result['class_id'], "Unknown Disease")
            
            print("Unexpected response format:", result)
            return None
            
        print(f"Classification failed: {response.status_code} - {response.text}")
        return None
        
    except Exception as e:
        print(f"Classification error: {str(e)}")
        print(traceback.format_exc())
        return None

def detect_disease(input_text):
    try:
        disease_keywords = {
            "Alternaria Leaf Spot": ["alternaria", "dark", "leaf spot", "yellow circle", "yellow halo"],
            "Cercospora Leaf Spot": ["cercospora", "circular", "brown", "spots"],
            "Downy Mildew": ["downy", "mildew", "white", "powdery", "powder"],
            "Leaf Curl Virus": ["leaf curl", "curly"],
            "Bhendi Yellow Vein Mosaic Disease": ["yellow vein", "mosaic"],
            "Phyllosticta Leaf Spot": ["phyllosticta", "reddish margins", "brown margins", "holes"]
        }
        
        input_lower = input_text.lower()
        for disease, keywords in disease_keywords.items():
            if any(kw in input_lower for kw in keywords):
                return disease
        return None
    except Exception as e:
        print(f"Error in detect_disease: {str(e)}")
        print(traceback.format_exc())
        return None

def is_plant_related(input_text):
    try:
        plant_keywords = ["plant", "okra", "leaf", "leaves", "disease", "yellow", "spots", "wilt", "mildew", "virus", "mosaic", "health", "grow", "garden", "farm"]
        return any(kw in input_text.lower() for kw in plant_keywords)
    except Exception as e:
        print(f"Error in is_plant_related: {str(e)}")
        print(traceback.format_exc())
        return False

def generate_disease_timeline(disease):
    """Generate disease-specific timeline with unique progression details and expert notes"""
    disease_timelines = {
        "Alternaria Leaf Spot": {
            "early": "3-5 days: Small brown spots with concentric rings (2-5mm diameter)",
            "middle": "1-2 weeks: Spots enlarge (up to 1cm) with yellow halos",
            "late": "2+ weeks: Leaves turn yellow and drop prematurely",
            "critical": "When 30% of leaves are affected",
            "speed_factors": [
                "Wet conditions (leaf wetness >12hrs)",
                "Temperatures 20-30°C",
                "Poor air circulation"
            ],
            "slow_factors": [
                "Dry weather",
                "Morning watering (avoids long leaf wetness)",
                "Proper plant spacing"
            ],
            "special_notes": [
                "🔹 The fungus survives in plant debris - clean fields thoroughly after harvest",
                "🔹 Rotate with non-host crops for at least 2 years",
                "🔹 Chlorothalonil-based fungicides are most effective when applied preventatively"
            ]
        },
        "Cercospora Leaf Spot": {
            "early": "4-7 days: Small circular brown spots with reddish margins",
            "middle": "1-2 weeks: Spots develop gray centers with dark borders",
            "late": "3 weeks: Severe defoliation starting from lower leaves",
            "critical": "When spots merge covering >50% leaf surface",
            "speed_factors": [
                "High humidity (>85%)",
                "Overhead irrigation",
                "Infected plant debris in soil"
            ],
            "slow_factors": [
                "Drip irrigation",
                "Regular fungicide sprays",
                "Resistant varieties"
            ],
            "special_notes": [
                "🔹 The pathogen can survive in seeds - use certified disease-free seeds",
                "🔹 Mancozeb fungicides work well when applied at first sign of spots",
                "🔹 Remove and destroy infected leaves immediately to slow spread"
            ]
        },
        "Downy Mildew": {
            "early": "2-4 days: Pale green/yellow angular spots on upper leaf surfaces",
            "middle": "5-7 days: White fluffy growth appears on leaf undersides",
            "late": "10-14 days: Leaves curl, turn brown and die",
            "critical": "When white spores appear on stems",
            "speed_factors": [
                "Cool nights (15-20°C) with dew",
                "High humidity",
                "Dense plant canopy"
            ],
            "slow_factors": [
                "Copper-based fungicides",
                "Morning sunlight exposure",
                "Good weed control"
            ],
            "special_notes": [
                "🔹 Fungus spreads rapidly during rainy seasons - be extra vigilant",
                "🔹 Apply fungicides to both upper and lower leaf surfaces",
                "🔹 The pathogen doesn't survive in soil but can overwinter in weeds"
            ]
        },
        "Leaf Curl Virus": {
            "early": "5-10 days: Slight upward curling of young leaves",
            "middle": "2 weeks: Severe leaf thickening and distortion",
            "late": "3 weeks: Plant stunting with no fruit production",
            "critical": "10 days after first symptoms appear",
            "speed_factors": [
                "High whitefly populations",
                "Temperatures >30°C",
                "Susceptible okra varieties"
            ],
            "slow_factors": [
                "Whitefly control",
                "Early planting",
                "Barrier crops"
            ],
            "special_notes": [
                "🔹 There is NO CURE for viral infections - focus on prevention",
                "🔹 Use yellow sticky traps to monitor whitefly populations",
                "🔹 Remove and burn infected plants immediately - do not compost",
                "🔹 Plant resistant varieties like 'Pusa Sawani' and 'Arka Anamika'"
            ]
        },
        "Bhendi Yellow Vein Mosaic Disease": {
            "early": "7-10 days: Yellow vein clearing on young leaves",
            "middle": "2 weeks: Complete yellow mosaic pattern develops",
            "late": "3 weeks: Severe stunting with malformed fruits",
            "critical": "When flowering is affected",
            "speed_factors": [
                "Whitefly transmission",
                "Weed hosts nearby",
                "Warm dry weather"
            ],
            "slow_factors": [
                "Virus-free seeds",
                "Yellow sticky traps",
                "Early whitefly control"
            ],
            "special_notes": [
                "🔹 The virus is transmitted in persistent manner by whiteflies",
                "🔹 Remove and destroy infected plants within 3 days of detection",
                "🔹 Spray systemic insecticides like Imidacloprid for whitefly control",
                "🔹 Grow barrier crops like maize around okra fields"
            ]
        },
        "Phyllosticta Leaf Spot": {
            "early": "5-8 days: Small reddish-brown spots with dark margins",
            "middle": "2 weeks: Spots develop light centers and may fall out (shot holes)",
            "late": "3 weeks: Severe leaf drop occurs",
            "critical": "When 40% of leaves are infected",
            "speed_factors": [
                "Rainy weather",
                "Wounds on leaves",
                "High nitrogen fertilization"
            ],
            "slow_factors": [
                "Copper fungicides",
                "Proper sanitation",
                "Balanced fertilization"
            ],
            "special_notes": [
                "🔹 Fungus spreads through splashing water - avoid overhead irrigation",
                "🔹 Prune affected leaves during dry weather to prevent spread",
                "🔹 Apply Bordeaux mixture (1%) at 15 day intervals as preventive measure",
                "🔹 The pathogen can survive on tools - disinfect after use"
            ]
        }
    }

    # Get timeline data or use default if disease not found
    timeline = disease_timelines.get(disease, {
        "early": "3-7 days: Initial symptoms appear",
        "middle": "1-2 weeks: Disease progresses",
        "late": "2+ weeks: Severe damage occurs",
        "critical": "When symptoms become irreversible",
        "speed_factors": ["Environmental stress", "Pest pressure"],
        "slow_factors": ["Proper care", "Early treatment"],
        "special_notes": ["🔹 Consult local agricultural extension for specific advice"]
    })

    # Construct the response
    response = f"""
⏳ **{disease} Progression Timeline**

🌱 Early Stage:
- {timeline['early']}
- First visible signs appear

🔄 Middle Stage:
- {timeline['middle']}
- Disease becomes well-established

⚠️ Late Stage:
- {timeline['late']}
- Plant health severely compromised

🛑 Critical Point:
- {timeline['critical']}
- Beyond this point, recovery is unlikely

⚡ Progression Factors:
{''.join(f'- Accelerated by: {factor}\n' for factor in timeline['speed_factors'])}
{''.join(f'- Slowed by: {factor}\n' for factor in timeline['slow_factors'])}

💡 Expert Notes:
{''.join(f'{note}\n' for note in timeline['special_notes'])}
"""

    return response

def generate_plant_response(input_text, disease_context=None):
    """Generate responses for plant-related queries with timeline support"""
    try:
        # Check for timeline queries
        timeline_triggers = [
            "worsen", "timeline", "progress", "how long", 
            "worsening time" , "duration", "develop",
            "advance", "speed", "fast", "slow", "phase", "period"
        ]
        
        is_timeline_query = any(
            trigger in input_text.lower()
            for trigger in timeline_triggers
        )
        
        if is_timeline_query and conversation_context['current_disease']:
            return generate_disease_timeline(conversation_context['current_disease'])
        
        prompt = f"""You're OkraBot, an expert in okra plant care. Respond to:

{disease_context if disease_context else ''}
User query: "{input_text}"

Guidelines:
1. Be concise but thorough
2. Use markdown formatting
3. For diseases include:
   - Key symptoms
   - Recommended treatments
   - Prevention tips
4. Maintain warm, helpful tone
5. Suggest timeline with "Want timeline?" if relevant

Response:"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Error generating plant response: {e}")
        return "I'm having trouble with plant questions right now. Could you try again?"

def generate_general_response(input_text):
    """Generate responses for non-plant conversations"""
    try:
        prompt = f"""You're OkraBot, a friendly chatbot. The user asked:

User: "{input_text}"

Respond warmly but briefly, gently steering toward plant topics when appropriate.

Response:"""
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Error generating general response: {e}")
        return "I'm having trouble responding. Maybe ask me about okra plants?"

def generate_response(input_text, image_path=None):
    try:
        print(f"Processing input: {input_text}")
        conversation_context['conversation_history'].append({"user": input_text, "bot": ""})
        
        if image_path:
            print("Processing image:", image_path)
            classification_result = classify_image(image_path)
            print("Classification result:", classification_result)
            
            if not classification_result:
                response = "⚠️ Couldn't analyze the image. Please describe the issue."
            elif classification_result.lower() == "healthy":
                response = "✅ Healthy plant detected! No signs of disease found."
                conversation_context['current_disease'] = None
            else:
                conversation_context['current_disease'] = classification_result
                disease_context = f"Context: The user's plant has {classification_result}"
                response = generate_plant_response(
                    f"Explain {classification_result} and suggest treatments",
                    disease_context
                )
                if classification_result in DISEASE_RESOURCES:
                    response += f"\n\n📚 Learn more: {DISEASE_RESOURCES[classification_result]}"
            
            conversation_context['conversation_history'][-1]["bot"] = response
            return response

        # Text conversation handling
        is_plant_query = is_plant_related(input_text)
        detected_disease = detect_disease(input_text)
        
        if is_plant_query:
            if detected_disease:
                conversation_context['current_disease'] = detected_disease
                disease_context = f"Context: Possible {detected_disease}"
                response = generate_plant_response(input_text, disease_context)
                if detected_disease in DISEASE_RESOURCES:
                    response += f"\n\n📚 Learn more: {DISEASE_RESOURCES[detected_disease]}"
            else:
                response = generate_plant_response(input_text)
        else:
            if any(word in input_text.lower() for word in ["hi", "hello", "hey"]):
                response = "👋 Hello! I'm OkraBot, your okra plant assistant. How can I help today?"
            else:
                response = generate_general_response(input_text)
        
        conversation_context['conversation_history'][-1]["bot"] = response
        return response
        
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
        print(traceback.format_exc())
        return "Oops! Something went wrong. Let's try again."

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    print("Received request to /chat")
    print("Form data:", request.form)
    print("Files:", request.files)
    try:
        user_input = request.form.get("message", "").strip()
        image_file = request.files.get("image")
        response = ""
        
        if image_file:
            image_path = os.path.join("uploads", image_file.filename)
            os.makedirs("uploads", exist_ok=True)
            image_file.save(image_path)
            response = generate_response(user_input, image_path)
            os.remove(image_path)
        else:
            response = generate_response(user_input)
        
        return jsonify({"response": response})
    
    except Exception as e:
        print(f"Chat error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"response": "An error occurred. Please try again."})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True, port=5001)