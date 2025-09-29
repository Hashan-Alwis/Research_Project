from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import pandas as pd
import joblib
import os
import base64
from ultralytics import YOLO
import cv2
from io import BytesIO
from PIL import Image
import subprocess
import json


import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)
CORS(app)

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



#get tourist category 
try:
    with open('RF_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    loaded_preprocessor = joblib.load('preprocessor.pkl')

except Exception as e:
    print(f"Error loading model/preprocessor: {e}")
    loaded_model = None
    loaded_preprocessor = None


def get_prediction(df):
    x =loaded_preprocessor.transform(df) 
    y_pred = loaded_model.predict(x)
    # print(y_pred[0])
    return y_pred[0]


@app.route('/get_category', methods=['POST'])
def predict_category():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400
    
    keys_list = list(data.keys())
    tourist_data= [data.get(field) for field in keys_list]
    tourist_data = tourist_data[:3] + tourist_data[3] + tourist_data[4:]

    df_column= ['Age', 'Travel Style', 'Monthly Income (USD)', 'Preferred Activities 1',
       'Preferred Activities 2', 'Preferred Activities 3',
       'Preferred Travel Season', 'Preferred Mode of Transportation',
       'Tech Savvy', 'Accommodation Type']

    df = pd.DataFrame([tourist_data], columns=df_column)
    # print(df)
    try:
        category = get_prediction(df)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    return jsonify({"status": "success", "category": category}), 200

##################


#food identification model

best_model = YOLO("food_identification_model.pt")

def numpy_to_base64(image_np, format="PNG"):
    """Convert NumPy image array to base64-encoded string."""
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    image_pil.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/food', methods=['POST'])
def detect_food():
    data = request.get_json()

    if 'image' not in data or 'filename' not in data:
        return jsonify({"error": "Missing image or filename"}), 400

    try:
        image_data = base64.b64decode(data['image'])
        file_path = os.path.join(UPLOAD_FOLDER, data['filename'])


        with open(file_path, "wb") as f:
            f.write(image_data)

        # Perform YOLO prediction
        results = best_model.predict(source=file_path, conf=0.5, show=False, save=False, verbose=False)
        result = results[0]  

        
        pred_image = result.plot()
        base64_str = numpy_to_base64(pred_image)

        
        class_names = []
        if result.boxes is not None and len(result.boxes.cls) > 0:
            class_ids = result.boxes.cls.tolist()  
            class_names = [result.names[int(cls_id)] for cls_id in class_ids]

        class_names = list(set(class_names))

        return jsonify({"image": base64_str, "class_names": class_names})

    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

