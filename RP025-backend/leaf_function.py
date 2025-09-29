import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


model_disease = tf.keras.models.load_model('leaf_model___2.h5')
model_dying = tf.keras.models.load_model('model_dying_leaves.h5')

class_labels_disease  ={'LeafBlight': 0, 'LeafGall': 1, 'Magnesium': 2, 'Potassium': 3, 'healthy': 4}

class_labels_disease = {v: k for k, v in class_labels_disease.items()}

class_labels_dying  ={'disease': 0, 'natural': 1}

class_labels_dying = {v: k for k, v in class_labels_dying.items()}





def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array


def predict_image_leaf(img_path):
    img_array = preprocess_image(img_path)
    predictions = model_disease.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1) 
    predicted_class_label = class_labels_disease[predicted_class_idx[0]]
    
    return predicted_class_label, predictions[0].max()


def predict_dying_leaf(img_path):
    img_array = preprocess_image(img_path)
    predictions = model_dying.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1) 
    predicted_class_label = class_labels_dying[predicted_class_idx[0]]
    
    return predicted_class_label, predictions[0].max()




# img_path = 'E:\\MR.Mind_Projects\\25-4\\dying_leaves\\test\\disease\\IMG_0849(1).JPG'

# predicted_class, predictions = predict_dying_leaf(img_path)
# print(f'Predicted Class: {predicted_class}')
# print(f'Prediction Probabilities: {predictions}')