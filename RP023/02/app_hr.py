import json
import time
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pickle
import pandas as pd
import joblib

import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)
CORS(app)

try:
    with open('RF_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    loaded_preprocessor = joblib.load('preprocessor.pkl')

except Exception as e:
    print(f"Error loading model/preprocessor: {e}")
    loaded_model = None
    loaded_preprocessor = None


def get_employee_category(df):
    try:
        x = loaded_preprocessor.transform(df)
        y_pred = loaded_model.predict(x)
        return y_pred[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


@app.route('/get_category', methods=['POST'])
def predict_category():
    data = request.get_json()

    if not data or "candidate" not in data:
        return jsonify({"status": "error", "message": "Invalid JSON data"}), 400

    candidate = data.get("candidate", {})
    
    candidate_fields = ["age", "applying_position", "experience", "leadership_experience", 
                        "english_proficiency", "salary_expectation", "gender"]
    
    df_column = ['age', 'applying position', 'experience', 'leadership experience',
                    'english proficiency', 'salary expectation', 'gender']
    
    
    candidate_data = [candidate.get(field) for field in candidate_fields]
    # print(candidate_data)

    
    if None in candidate_data:
        return jsonify({"status": "error", "message": "Missing required candidate fields"}), 400

    df = pd.DataFrame([candidate_data], columns=df_column)

    category = get_employee_category(df)
    
    if category is None:
        return jsonify({"status": "error", "message": "Prediction failed"}), 500

    return jsonify({"status": "success", "category": category}), 200


if __name__ == '__main__':
    app.run(debug=True)
