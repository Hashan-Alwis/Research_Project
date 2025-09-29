from flask import Flask, request, jsonify
import pandas as pd
from functions import get_next_day_whether
from flask_cors import CORS


import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

@app.route('/predict_whether', methods=['POST'])
def predict_whether():
    global data_store
    try:
        json_data = request.get_json()

        df = pd.DataFrame(json_data)
        
        df['time'] = pd.to_datetime(df['time'])
        
        df.set_index('time', inplace=True)
        # print(df)

        predictions = get_next_day_whether(df)
        # print(predictions)
        
        response = {
            "temperature": float(predictions[0][0]),
            "apparent_temperature": float(predictions[1][0]),
            "wind_speed": float(predictions[2][0]),
            "wind_gust": float(predictions[3][0])
        }

        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
