import google.generativeai as genai

# Configure the API
genai.configure(api_key="AIzaSyBAgXgwsXv-EKFaHvJInXu3X35Qif-skzo")

generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 30,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}

try:
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    print("Model initialized successfully")

    # Test a simple prompt
    prompt = "Hello! Can you tell me about okra plants?"
    response = model.generate_content(prompt)
    print("Response:", response.text)

except Exception as e:
    print(f"Error: {e}")