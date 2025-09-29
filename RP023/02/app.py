import json , time
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
import numpy as np
import pickle
import pandas as pd
import joblib


with open('RF_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

loaded_preprocessor = joblib.load('preprocessor.pkl')


def get_employee_category(df):
    x =loaded_preprocessor.transform(df) 
    y_pred = loaded_model.predict(x)
    print(y_pred[0])
    return y_pred


app = Flask(__name__)

@app.route('/get_category', methods=['POST'])
def capture_json():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Invalid JSON data"}), 400
    
    status = data.get("status")
    candidate = data.get("candidate", {})
    name = candidate.get("name")
    age = candidate.get("age")
    applying_position = candidate.get("applying_position")
    experience = candidate.get("experience")
    leadership_experience = candidate.get("leadership_experience")
    english_proficiency = candidate.get("english_proficiency")
    salary_expectation = candidate.get("salary_expectation")
    gender = candidate.get("gender")

    data =pd.DataFrame([[age, applying_position, experience, leadership_experience, english_proficiency, salary_expectation, gender]], 
                       columns=['age', 'applying position', 'experience', 'leadership experience','english proficiency', 'salary expectation', 'gender'])
    
    category = get_employee_category(data)



    
    return jsonify({"status": "success", "category": category[0]}), 200

if __name__ == '__main__':
    app.run(debug=True)
