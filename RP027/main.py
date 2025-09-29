import json
import time
import os
import base64
import io
import tempfile
import logging
import warnings
import numpy as np
import joblib
import librosa
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from pydub import AudioSegment
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


custom_temp_dir = os.path.join(os.getcwd(), "temp_audio_files")
os.makedirs(custom_temp_dir, exist_ok=True)

model = load_model('best_model.h5')
encoder = joblib.load('encoder.pkl')

def extract_audio_features(filename):
    try:
        y, sr = librosa.load(filename, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc.reshape(1, 40, 1)
    except Exception as e:
        logging.error(f"Feature extraction failed: {str(e)}")
        return None

app = Flask(__name__)
CORS(app)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/articulation', methods=['POST'])
def get_articulation():
    request_data = request.get_json()
    
    audio_list = [
        request_data.get('track1', ''),
        request_data.get('track2', ''),
        request_data.get('track3', ''),
        request_data.get('track4', ''),
        request_data.get('track5', '')
    ]

    prediction_list = []
    error_list = []

    for i, audio in enumerate(audio_list, start=1):
        try:
            if not audio:
                error_list.append(f"Track {i}: No audio data provided.")
                continue

            audio_data = base64.b64decode(audio)

            # print(f"First 20 bytes of audio: {audio_data[:20]}")
            if audio_data[:4] == b'\x1aE\xdf\xa3':
                format = "webm"
            elif audio_data[:4] == b'RIFF':
                format = "wav"
            elif audio_data[:3] == b'ID3' or audio_data[:2] == b'\xff\xfb':
                format = "mp3"
            elif audio_data[:4] == b'OggS':
                format = "ogg"
            else:
                error_list.append(f"Track {i}: Unsupported audio format.")
                continue

            # print(format)    

            audio_io = io.BytesIO(audio_data)
            audio_segment = AudioSegment.from_file(audio_io, format=format)

            # play(audio_segment)

            with tempfile.NamedTemporaryFile(suffix=".wav", dir=custom_temp_dir, delete=False) as temp_audio_file:
                temp_filename = temp_audio_file.name
                audio_segment.export(temp_filename, format="wav")

                # y, sr = librosa.load(temp_audio_file.name, sr=None)
                # duration = librosa.get_duration(y=y, sr=sr)
                # print(f"Audio Duration: {duration} seconds")


            audio_features = extract_audio_features(temp_filename)
            os.remove(temp_filename) 

            if audio_features is None:
                error_list.append(f"Track {i}: Feature extraction failed.")
                continue


            prediction = model.predict(audio_features).round()
            y_decoded = encoder.inverse_transform(prediction)
            # print(y_decoded)
            articulation_problem = y_decoded[0][0]
            prediction_list.append(articulation_problem)

            print(f'Audio processed successfully. Audio model output {i}: {articulation_problem}.')

        except Exception as e:
            error_list.append(f"Track {i}: {str(e)}")


    response_data = {
        "problems": prediction_list,
        "errors": error_list
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
