import tensorflow as tf
from time import time
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input
import json
import base64


loaded_model = tf.keras.models.load_model("save_model/")


def predict_attention(img):
    label_mapping = {0: "attention", 1: "not_attention"}
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(img_array)
    predictions = loaded_model.predict(preprocessed_img, verbose=0)
    output = np.argmax(predictions, axis=-1)
    output_categories = [label_mapping[label] for label in output]
    return output_categories


face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')


capture_interval = 0 

def generate_frames():
    cap = cv2.VideoCapture(0)
    start_time = time()

    try:
        while True:
            if time() - start_time >= capture_interval:
                start_time = time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Error: Failed to capture frame.")
                    break

                result = ["no face detected"]
                try:
                    height, width = frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
                    face_net.setInput(blob)
                    detections = face_net.forward()

                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.5: 
                            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                            (x1, y1, x2, y2) = box.astype("int")

                            only_face = frame[y1:y2, x1:x2]

                            resized = cv2.resize(only_face, (224, 224), interpolation=cv2.INTER_AREA)
                            result = predict_attention(resized)

                            label_position = (x1, y1 - 10)
                            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            # cv2.putText(frame, result[0], label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            # print(result[0])

                except Exception as e:
                    print("Error occurred:", e)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame_encoded = base64.b64encode(buffer).decode('utf-8')

                response_data = {"Prediction": result[0], "frame": frame_encoded, "success": "true"}
                yield f"data: {json.dumps(response_data)}\n\n"
    finally:
        cap.release()
