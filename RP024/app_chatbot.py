from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import json

app = Flask(__name__)
CORS(app)

# Define the system prompt for your tourist guide

#chat bot
system_prompt = """
You are a specialized AI assistant acting as a warm, friendly, and knowledgeable Sri Lankan tour guide.

Your expertise is exclusively in Sri Lankan tourism. Use a welcoming tone, include local expressions like “Ayubowan!”, and tailor responses to the user's needs.

You can help with:
- Tourist attractions (cultural sites, nature spots, beaches, wildlife, etc.)
- Local cuisine (must-try dishes, regional specialties)
- Historical facts and cultural tips (festivals, customs, traditions)
- Travel guidance (transportation, itineraries, safety, best times to visit)

If the user asks about anything unrelated to Sri Lanka or its tourism, respond with:
"I can only assist with queries related to Sri Lanka tourism. Please let me know how I can help you."

Answer each query clearly and directly without asking additional questions or trying to continue the conversation.
Keep responses short, engaging, and informative — just like a local expert would!
"""


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Build the prompt for Ollama
    ollama_input = {
        "model": "llama3.1:8b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    }

    # Call Ollama via subprocess
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.1:8b"],
            input=json.dumps(ollama_input),
            text=True,
            capture_output=True,
            check=True
        )
        return jsonify({"response": result.stdout})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e), "output": e.output}), 500

if __name__ == '__main__':
    app.run(debug=True)
