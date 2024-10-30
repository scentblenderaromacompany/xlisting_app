import os
import io
import json
import numpy as np
import psycopg2
from PIL import Image
from sqlalchemy.orm import sessionmaker
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle
import logging
from utils.logger import logger
from utils.ImageDataGenerator import get_image_data_generator
from utils.mlflow_tracking import setup_mlflow, log_model

# Database connection
from db_connection import engine, Session

# Initialize session
session = Session()

def define_model(num_types, num_materials):
    """
    Define the multi-output classification model using EfficientNetB0.
    """
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Multi-output layers
    type_output = Dense(num_types, activation='softmax', name='type')(x)
    material_output = Dense(num_materials, activation='softmax', name='material')(x)
    
    model = Model(inputs=base_model.input, outputs=[type_output, material_output])
    
    model.compile(
        optimizer='adam',
        loss={
            'type': 'categorical_crossentropy',
            'material': 'categorical_crossentropy'
        },
        metrics=['accuracy']
    )
    return model

def load_data():
    """
    Fetch preprocessed images and their attributes from PostgreSQL.
    """
    try:
        cursor = session.connection().cursor()
        cursor.execute("SELECT processed_image, attributes FROM preprocessed_images")
        data = cursor.fetchall()
        images = []
        types = []
        materials = []
        for row in data:
            image_data = row[0]
            attributes = json.loads(row[1])
            img = Image.open(io.BytesIO(image_data))
            img = img.convert('RGB')
            img = img.resize((256, 256))
            img_array = img_to_array(img)
            images.append(img_array)
            types.append(attributes.get('type', 'unknown'))
            materials.append(attributes.get('material', 'unknown'))
        cursor.close()
        images = np.array(images) / 255.0  # Normalize
        return images, types, materials
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        return None, None, None

def encode_labels(types, materials):
    """
    Encode categorical labels using LabelEncoder and convert to categorical.
    """
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.utils import to_categorical
    
    type_encoder = LabelEncoder()
    material_encoder = LabelEncoder()
    
    types_encoded = type_encoder.fit_transform(types)
    materials_encoded = material_encoder.fit_transform(materials)
    
    num_types = len(type_encoder.classes_)
    num_materials = len(material_encoder.classes_)
    
    types_categorical = to_categorical(types_encoded, num_classes=num_types)
    materials_categorical = to_categorical(materials_encoded, num_classes=num_materials)
    
    # Save encoders
    with open('models/jewelry_classifier/type_encoder.pkl', 'wb') as f:
        pickle.dump(type_encoder, f)
    with open('models/jewelry_classifier/material_encoder.pkl', 'wb') as f:
        pickle.dump(material_encoder, f)
    
    return types_categorical, materials_categorical, num_types, num_materials

def train_model():
    """
    Train the jewelry classification model.
    """
    images, types, materials = load_data()
    if images is None:
        logger.error("No data loaded. Exiting training.")
        return
    
    y_types, y_materials, num_types, num_materials = encode_labels(types, materials)
    
    # Split the data
    X_train, X_val, y_train_type, y_val_type, y_train_material, y_val_material = train_test_split(
        images, y_types, y_materials, test_size=0.2, random_state=42
    )
    
    # Define the model
    model = define_model(num_types, num_materials)
    
    # Data Augmentation
    datagen = get_image_data_generator()
    datagen.fit(X_train)
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('models/jewelry_classifier/jewelry_classifier.h5', monitor='val_loss', save_best_only=True)
    
    # Setup MLflow
    setup_mlflow()
    
    # Train the model
    history = model.fit(
        datagen.flow(X_train, {'type': y_train_type, 'material': y_train_material}, batch_size=32),
        epochs=50,
        validation_data=(X_val, {'type': y_val_type, 'material': y_val_material}),
        callbacks=[early_stop, checkpoint]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 8))
    
    # Type Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history.history['type_accuracy'], label='Train Type Accuracy')
    plt.plot(history.history['val_type_accuracy'], label='Val Type Accuracy')
    plt.legend()
    plt.title('Type Accuracy')
    
    # Material Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history['material_accuracy'], label='Train Material Accuracy')
    plt.plot(history.history['val_material_accuracy'], label='Val Material Accuracy')
    plt.legend()
    plt.title('Material Accuracy')
    
    # Type Loss
    plt.subplot(2, 2, 3)
    plt.plot(history.history['type_loss'], label='Train Type Loss')
    plt.plot(history.history['val_type_loss'], label='Val Type Loss')
    plt.legend()
    plt.title('Type Loss')
    
    # Material Loss
    plt.subplot(2, 2, 4)
    plt.plot(history.history['material_loss'], label='Train Material Loss')
    plt.plot(history.history['val_material_loss'], label='Val Material Loss')
    plt.legend()
    plt.title('Material Loss')
    
    plt.tight_layout()
    plt.savefig('models/jewelry_classifier/training_history.png')
    plt.close()
    logger.info("Training history plot saved.")
    
    # Evaluate the model
    y_pred = model.predict(X_val)
    y_pred_type = np.argmax(y_pred[0], axis=1)
    y_pred_material = np.argmax(y_pred[1], axis=1)
    y_true_type = np.argmax(y_val_type, axis=1)
    y_true_material = np.argmax(y_val_material, axis=1)
    
    # Load encoders
    with open('models/jewelry_classifier/type_encoder.pkl', 'rb') as f:
        type_encoder = pickle.load(f)
    with open('models/jewelry_classifier/material_encoder.pkl', 'rb') as f:
        material_encoder = pickle.load(f)
    
    # Decode labels
    y_true_type_labels = type_encoder.inverse_transform(y_true_type)
    y_pred_type_labels = type_encoder.inverse_transform(y_pred_type)
    y_true_material_labels = material_encoder.inverse_transform(y_true_material)
    y_pred_material_labels = material_encoder.inverse_transform(y_pred_material)
    
    # Classification Report for Type
    type_report = classification_report(y_true_type_labels, y_pred_type_labels, output_dict=True)
    material_report = classification_report(y_true_material_labels, y_pred_material_labels, output_dict=True)
    
    # Save evaluation metrics to PostgreSQL
    try:
        cursor = session.connection().cursor()
        for attribute, metrics in type_report.items():
            if attribute in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            cursor.execute(
                \"""
                INSERT INTO model_evaluation (attribute, accuracy, precision, recall, f1_score)
                VALUES (%s, %s, %s, %s, %s)
                \""",
                (
                    attribute,
                    metrics['precision'],
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1-score']
                )
            )
        for attribute, metrics in material_report.items():
            if attribute in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            cursor.execute(
                \"""
                INSERT INTO model_evaluation (attribute, accuracy, precision, recall, f1_score)
                VALUES (%s, %s, %s, %s, %s)
                \""",
                (
                    attribute,
                    metrics['precision'],
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1-score']
                )
            )
        session.connection().commit()
        cursor.close()
        logger.info("Model evaluation metrics stored in database")
    except Exception as e:
        logger.error(f"Error storing model evaluation metrics in database: {e}")
    
    # Log model to MLflow
    metrics = {
        'type_accuracy': type_report['weighted avg']['f1-score'],
        'material_accuracy': material_report['weighted avg']['f1-score']
    }
    params = {
        'model_type': 'EfficientNetB0',
        'optimizer': 'adam',
        'epochs': 50
    }
    log_model(model, params, metrics, artifacts={'training_history': 'models/jewelry_classifier/training_history.png'})

if __name__ == "__main__":
    train_model()
