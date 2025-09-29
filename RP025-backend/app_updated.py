from flask import Flask, request, jsonify
from flask_cors import CORS
from bark_funtion import predict_image_bark 
from mature_funtion import predict_image_maturity
from leaf_function import predict_image_leaf, predict_dying_leaf
from yolo_funtion import get_area_percentage, leaf_bligh_model, leaf_gall_model
import os
import pickle
import pandas as pd


import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'E:\\MR.Mind_Projects\\'

#check bark condition
@app.route('/check_bark', methods=['POST'])
def check_bark():
    data = request.get_json()
    
    if not data or 'image_path' not in data:
        return jsonify({'error': 'No image path provided'}), 400
    
    image_path = os.path.join(UPLOAD_FOLDER, data['image_path'])
    
    predicted_label, confidence = predict_image_bark(image_path)
    
    if predicted_label is None:
        return jsonify({'error': confidence}), 400
    
    return jsonify({
        'predicted_label': predicted_label,
        'confidence': float(confidence)
    })

#check maturity
@app.route('/predict_maturity', methods=['POST'])
def predict_maturity():
    data = request.get_json()
    
    if not data or 'image_path' not in data:
        return jsonify({'error': 'No image path provided'}), 400
    
    image_path = os.path.join(UPLOAD_FOLDER, data['image_path'])
    
    if not image_path or not isinstance(image_path, str):
        return jsonify({'error': 'Invalid image path'}), 400
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'File not found'}), 400
    
    try:
        predicted_class, confidence = predict_image_maturity(image_path)
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    
    return jsonify({
        'predicted_class': predicted_class,
        'confidence': float(confidence)
    })

#leaf disease detection

@app.route('/predict_leaf', methods=['POST'])
def check_disease():
    data = request.get_json()
    
    if not data or 'image_path' not in data:
        return jsonify({'error': 'No image path provided'}), 400
    
    image_path = os.path.join(UPLOAD_FOLDER, data['image_path'])
    
    if not image_path or not isinstance(image_path, str):
        return jsonify({'error': 'Invalid image path'}), 400
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'File not found'}), 400
    
    try:
        predicted_class, confidence = predict_image_leaf(image_path)
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    
    return jsonify({
        'predicted_class': predicted_class,
        'confidence': float(confidence)
    })

#check dying leaf

@app.route('/predict_dying_leaf', methods=['POST'])
def check_dying_reason():
    data = request.get_json()
    
    if not data or 'image_path' not in data:
        return jsonify({'error': 'No image path provided'}), 400
    
    image_path = os.path.join(UPLOAD_FOLDER, data['image_path'])
    
    if not image_path or not isinstance(image_path, str):
        return jsonify({'error': 'Invalid image path'}), 400
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'File not found'}), 400
    
    try:
        predicted_class, confidence = predict_dying_leaf(image_path)
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    
    return jsonify({
        'predicted_class': predicted_class,
        'confidence': float(confidence)
    })

#check leaf disease area percentage
MODEL_DICT = {
    "leaf_bligh_model": leaf_bligh_model,
    "leaf_gall_model": leaf_gall_model
}

@app.route('/predict_leaf_area', methods=['POST'])
def predict_leaf_area():
    data = request.get_json()
    
    if not data or 'image_path' not in data or 'model_type' not in data:
        return jsonify({'error': 'Both image_path and model_type are required'}), 400
    
    image_path = os.path.join(UPLOAD_FOLDER, data['image_path'])
    model_type = data['model_type']
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'File not found'}), 400
    
    if model_type not in MODEL_DICT:
        return jsonify({'error': f'Invalid model_type. Choose one of {list(MODEL_DICT.keys())}'}), 400
    
    model_info = MODEL_DICT[model_type]
    
    try:
        percentage = get_area_percentage(image_path, model_info)
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}',
                        'success': False
                        }), 500
    
    return jsonify({
        # 'image_path': image_path,
        'success': True,
        'affected_area_percentage': percentage
    })

# check the quality of cinamen
with open('cinamen_quality_model.pkl', 'rb') as file:
    model = pickle.load(file)

feature_map = {
    "moisture": "Moisture (%)",
    "ash": "Ash (%)",
    "chromium_mgkg": "Chromium (mg/kg)",
    "coumarin_mgkg": "Coumarin (mg/kg)",
    "ph_level": "pH_Level",
    "eugenol": "Eugenol (%)"
}
@app.route('/predict_cinamen_quality', methods=['POST'])
def predict_cinamen_quality():
    try:
        data = request.get_json(force=True)
        input_data = {feature_map[k]: v for k, v in data.items() if k in feature_map}
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    

if __name__ == '__main__':
    app.run(debug=True)
