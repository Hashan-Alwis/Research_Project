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
import warnings

warnings.filterwarnings('ignore')

from collections import Counter

custom_temp_dir = os.path.join(os.getcwd(), "temp_audio_files")
if not os.path.exists(custom_temp_dir):
    os.makedirs(custom_temp_dir)

model = load_model('final_best_model.h5')
encoder = joblib.load('encoder.pkl')

saved_path = "E:\\MR.Mind_Projects\\RP026\\updated\\DATA_SET"

def extract_audio_features(filename):
    """Extract MFCC features for model prediction"""
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    out = np.array(mfcc).reshape(1, 40, 1)
    return out

def split_audio_to_chunks(audio_path, chunk_length_ms=4000):
    """Split audio into 4 second chunks and return list of temp wav file paths"""
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_length_ms)):
        end = start + chunk_length_ms
        chunk = audio[start:end]
        if len(chunk) < 1000:  # ignore very small chunks (<1s)
            continue
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        chunk.export(temp_file.name, format="wav")
        chunks.append(temp_file.name)
    return chunks

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/emotion', methods=['POST'])
def get_emotion():
    request_data = request.get_json()
    audio_path = request_data.get('track', '')   # only one audio path

    audio_path= saved_path + "/" + audio_path
    # audio_path = os.path.join(saved_path, audio_path)
    print(f"shtisssasdfasdfasdfodfasdf  {audio_path}")

    all_predictions = []
    response_message = ""

    try:
        if audio_path and os.path.exists(audio_path):
            file_ext = os.path.splitext(audio_path)[1].lower()

            # Convert to WAV if needed
            if file_ext != '.wav':
                audio_segment = AudioSegment.from_file(audio_path)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
                    temp_wav_path = temp_wav_file.name
                    audio_segment.export(temp_wav_path, format="wav")
            else:
                temp_wav_path = audio_path  # Already WAV

            # Split audio into 4 sec chunks
            chunk_paths = split_audio_to_chunks(temp_wav_path)

            for chunk_path in chunk_paths:
                try:
                    audio_features = extract_audio_features(chunk_path)

                    probs = model.predict(audio_features)[0]
                    predicted_index = np.argmax(probs)
                    confidence = float(probs[predicted_index])
                    print(confidence)

                    if confidence < 0.5:
                        all_predictions.append("noise")
                    else:
                        predicted_emotion = encoder.inverse_transform(probs.reshape(1, -1))
                        all_predictions.append(predicted_emotion[0][0])
                        
                except Exception as e:
                    all_predictions.append(f"error: {str(e)}")
                finally:
                    os.remove(chunk_path)  

            response_message = 'success'


            if file_ext != '.wav':
                os.remove(temp_wav_path)

        else:
            response_message = "File path not provided or does not exist."

    except Exception as e:
        response_message = f"An unexpected error occurred during audio processing: {str(e)}"

    

    pred_dic = {'anger': 0, 'disgust': 0, 'happy': 0, 'noise': 0, 'sad': 0}

    pred_counts = Counter(all_predictions)
    for key in pred_dic:
        pred_dic[key] += pred_counts.get(key, 0)

    # print(pred_dic)

    return jsonify({
        'predictions': pred_dic,
        'success': response_message
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
