import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# Load the trained model
model = tf.keras.models.load_model("flower_classifier_model.h5")

# Define class names (must match training order)
class_names = ['chamomile', 'dandelion', 'rose', 'sunflower', 'tulip']

# Image properties
img_height, img_width = 180, 180

def predict_flower(img_path):
    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0  # normalize

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    print(f"Image: {img_path}")
    print(f"Predicted: {predicted_class} ({confidence:.2f}%)")

# Example
predict_flower("flowers_dataset/daisy/5547758_eea9edfd54_n.jpg")