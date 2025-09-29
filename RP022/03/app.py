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
    with open('price_predicting_model.pkl', 'rb') as file:
        price_predicting_model = pickle.load(file)

except Exception as e:
    print(f"Error loading model: {e}")
    price_predicting_model = None



@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        data = request.get_json()
        
        df = pd.DataFrame([data])
        
        prediction = price_predicting_model.predict(df)[0]
        return jsonify({'price': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

