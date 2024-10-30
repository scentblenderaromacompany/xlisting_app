import os
import psycopg2
from db_connection import engine, Session
from tensorflow.keras.models import load_model
import pickle
from PIL import Image
import numpy as np
import json
import logging
from utils.logger import logger

# Initialize session
session = Session()

def load_classification_model():
    """
    Load the trained classification model and label encoders.
    """
    try:
        model = load_model('models/jewelry_classifier/jewelry_classifier.h5')
        with open('models/jewelry_classifier/type_encoder.pkl', 'rb') as f:
            type_encoder = pickle.load(f)
        with open('models/jewelry_classifier/material_encoder.pkl', 'rb') as f:
            material_encoder = pickle.load(f)
        logger.info("Classification model and encoders loaded successfully")
        return model, type_encoder, material_encoder
    except Exception as e:
        logger.error(f"Error loading classification model: {e}")
        return None, None, None

def preprocess_new_image(image_path):
    """
    Preprocess the new image for prediction.
    """
    try:
        image = Image.open(image_path)
        image = image.resize((256, 256))
        image = image.convert('RGB')
        img_array = np.array(image) / 255.0  # Normalize
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return None

def predict_attributes(model, encoders, image_array):
    """
    Predict jewelry attributes using the classification model.
    """
    try:
        prediction = model.predict(np.expand_dims(image_array, axis=0))
        type_pred = encoders['type_encoder'].inverse_transform([np.argmax(prediction[0])])[0]
        material_pred = encoders['material_encoder'].inverse_transform([np.argmax(prediction[1])])[0]
        return {'type': type_pred, 'material': material_pred}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {}

def store_prediction(original_image_url, attributes):
    """
    Store the prediction results in PostgreSQL.
    """
    try:
        cursor = session.connection().cursor()
        cursor.execute(\"""
            INSERT INTO predictions (original_image_url, attributes)
            VALUES (%s, %s)
        \""", (
            original_image_url,
            json.dumps(attributes)
        ))
        session.connection().commit()
        cursor.close()
        logger.info(f"Stored prediction for image {original_image_url}")
    except Exception as e:
        logger.error(f"Error storing prediction: {e}")

def run_inference(image_path, original_url):
    """
    Run the inference pipeline on a new image.
    """
    model, type_encoder, material_encoder = load_classification_model()
    if model is None:
        logger.error("Model not loaded. Exiting inference.")
        return {}
    
    image_array = preprocess_new_image(image_path)
    if image_array is None:
        logger.error("Image preprocessing failed. Exiting inference.")
        return {}
    
    attributes = predict_attributes(model, {'type_encoder': type_encoder, 'material_encoder': material_encoder}, image_array)
    if attributes:
        store_prediction(original_url, attributes)
    return attributes

def main():
    """
    Example usage of the inference pipeline.
    """
    sample_image = 'data/images/sample.jpg'  # Ensure this image exists
    original_url = 'http://example.com/sample.jpg'
    attributes = run_inference(sample_image, original_url)
    print(attributes)

if __name__ == "__main__":
    main()
