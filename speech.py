from flask import Flask, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define a function to get a response from the Gemini model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

chat_session = model.start_chat(
    history=[]
)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    logging.info(f"Received message: {user_message}")
    response = chat_session.send_message(user_message)
    logging.info(f"Response: {response.text}")
    return jsonify({'response': response.text})

if __name__ == "__main__":
    app.run(debug=True)
