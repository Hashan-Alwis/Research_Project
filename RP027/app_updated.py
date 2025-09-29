import json, time
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
import numpy as np
import pickle
import pandas as pd
import joblib
import io
import librosa
from pydub import AudioSegment
import tempfile
import os
from tensorflow.keras.models import load_model
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

custom_temp_dir = os.path.join(os.getcwd(), "temp_audio_files")
if not os.path.exists(custom_temp_dir):
    os.makedirs(custom_temp_dir)

model = load_model('best_model.h5')
encoder = joblib.load('encoder.pkl')

def extract_audio_features(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    out = np.array(mfcc).reshape(1, 40, 1)
    return out

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/articulation', methods=['POST'])
def get_articulation():
    request_data = request.get_json()

    audio_paths = [
        request_data.get('track1', ''),
        request_data.get('track2', ''),
        request_data.get('track3', ''),
        request_data.get('track4', ''),
        request_data.get('track5', '')
    ]

    prediction_list = []

    for i, audio_path in enumerate(audio_paths, start=1):
        try:
            if audio_path and os.path.exists(audio_path):

                # Convert input file (mp4) to WAV temporarily if needed
                # If the file is already wav, skip this step
                file_ext = os.path.splitext(audio_path)[1].lower()
                if file_ext != '.wav':
                    audio_segment = AudioSegment.from_file(audio_path)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
                        temp_wav_path = temp_wav_file.name
                        audio_segment.export(temp_wav_path, format="wav")
                else:
                    temp_wav_path = audio_path  # Already WAV

                audio_features = extract_audio_features(temp_wav_path)
                p = model.predict(audio_features)
                p = p.round()

                y_decoded = encoder.inverse_transform(p)
                articulation_problem = y_decoded[0][0]

                prediction_list.append(articulation_problem)

                response_message = f'Audio processed successfully. Audio model output {i}: {articulation_problem}.'

                # Cleanup temp wav if created
                if file_ext != '.wav':
                    os.remove(temp_wav_path)

            else:
                response_message = f'File path not provided or does not exist for track {i}.'
                print(response_message)

        except Exception as e:
            response_message = f'An unexpected error occurred during audio processing of track {i}: {str(e)}'
            print(response_message)

    return jsonify({'predictions': prediction_list})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
