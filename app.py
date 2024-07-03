from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define a function to get a response from the Gemini model
def get_gemini_response(message):
    try:
        # Create the model
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            generation_config=generation_config,
            # safety_settings = Adjust safety settings
            # See https://ai.google.dev/gemini-api/docs/safety-settings
        )

        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(message)
        return response.text
    except Exception as e:
        logging.error(f"Error: {e}")
        return "Sorry, I couldn't process that request."

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        if not user_message:
            logging.warning("Received empty message.")
            return jsonify({'response': 'Message is required'}), 400

        logging.info(f"Received message: {user_message}")
        response = get_gemini_response(user_message)
        logging.info(f"Response: {response}")

        return jsonify({'response': response})
    except Exception as e:
        logging.error(f"Error handling request: {e}")
        return jsonify({'response': 'An error occurred while processing your request'}), 500

if __name__ == "__main__":
    app.run(debug=False, host = '0.0.0.0')
