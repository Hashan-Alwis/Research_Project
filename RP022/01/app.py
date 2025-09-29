from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Load the YOLO model
best_model = YOLO("mayantha_model.pt")

def numpy_to_base64(image_np, format="PNG"):
    """Convert NumPy image array to base64-encoded string."""
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    image_pil.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app = Flask(__name__)
CORS(app)



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

        return jsonify({"image": base64_str, "class_names": class_names})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
