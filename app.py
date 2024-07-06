from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from gtts import gTTS
import os
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Configure Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("No GEMINI_API_KEY set for Flask application")

genai.configure(api_key=gemini_api_key)

# Define a function to get a response from the Gemini model
def get_gemini_response(user_input):
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
    )
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(user_input)
    return response.text

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = get_gemini_response(user_message)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run()

