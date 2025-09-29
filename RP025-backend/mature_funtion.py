import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

model = tf.keras.models.load_model('maturity_model.h5')

class_names = ['mature', 'unmatured']

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0 
    return img_array

def predict_image_maturity(img_path):
    img_array = load_and_preprocess_image(img_path)
    prediction = model.predict(img_array)
    confidence = max(prediction[0]) 
    # print(f"Confidence: {confidence:.2f}")
    predicted_class_index = np.argmax(prediction, axis=-1)[0]  
    predicted_class = class_names[predicted_class_index]
    # print(f"Prediction: {predicted_class}")
    
    return predicted_class, confidence

# test_img_path = "E:\\MR.Mind_Projects\\25-3\\dataset\\test\\mature\\IMG_4957.JPG"
# maturity, confidence =predict_image_maturity(test_img_path)
# print(f"Maturity: {maturity}, Confidence: {confidence}")