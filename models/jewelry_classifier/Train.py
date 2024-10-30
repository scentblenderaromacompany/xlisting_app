import os
import psycopg2
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import json
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection parameters using environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "jewelry_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")  # Secure password handling

# Connect to PostgreSQL
engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


def fetch_data():
    """
    Fetches preprocessed image paths and associated attributes from the database.

    Returns:
        pd.DataFrame: DataFrame containing image paths and attributes.
    """
    query = """
    SELECT images.image_path, attributes.type, attributes.material, attributes.color, 
           attributes.brand, attributes.metal
    FROM images
    JOIN image_attributes ON images.image_id = image_attributes.image_id
    JOIN attributes ON image_attributes.attribute_id = attributes.attribute_id
    WHERE images.preprocessed = TRUE;
    """
    df = pd.read_sql(query, engine)
    return df


def prepare_labels(df):
    """
    Prepares multi-label binarized labels for the dataset.

    Args:
        df (pd.DataFrame): DataFrame containing attribute columns.

    Returns:
        tuple: Binarized labels and label names.
    """
    label_cols = ["type", "material", "color", "brand", "metal"]
    mlb = MultiLabelBinarizer()
    df_labels = df[label_cols].apply(
        lambda x: [str(x[col]).lower() for col in label_cols if pd.notnull(x[col])],
        axis=1,
    )
    labels = mlb.fit_transform(df_labels)
    label_names = mlb.classes_
    return labels, label_names


def build_model(num_classes):
    """
    Builds and compiles the EfficientNetB0-based model for multi-label classification.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        Model: Compiled Keras model.
    """
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="sigmoid")(x)  # Multi-label
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_history(history):
    """
    Plots training accuracy and loss over epochs.

    Args:
        history: Training history object from model.fit().
    """
    plt.figure(figsize=(12, 4))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.legend()
    plt.title("Accuracy")
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss")
    plt.savefig("src/models/training_history.png")
    plt.close()
    logger.info(
        "Training history plotted and saved as 'src/models/training_history.png'."
    )


def save_model_metadata(training_metadata):
    """
    Saves model training metadata to the database.

    Args:
        training_metadata (dict): Metadata dictionary containing model parameters.
    """
    try:
        with psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        ) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO model_metadata (model_name, version, training_date, parameters)
                VALUES (%s, %s, %s, %s) RETURNING model_id;
            """,
                (
                    training_metadata["model_name"],
                    training_metadata["version"],
                    training_metadata["training_date"],
                    json.dumps(training_metadata["parameters"]),
                ),
            )
            model_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
        logger.info(f"Model metadata inserted with Model ID: {model_id}")
    except Exception as e:
        logger.error(f"Error inserting model metadata: {e}")


def main():
    df = fetch_data()
    if df.empty:
        logger.info("No data available for training.")
        return

    labels, label_names = prepare_labels(df)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        df["image_path"], labels, test_size=0.2, random_state=42
    )
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Data generators with enhanced augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({"filename": X_train, "class": list(y_train)}),
        x_col="filename",
        y_col="class",
        target_size=(224, 224),
        batch_size=32,
        class_mode="raw",
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({"filename": X_val, "class": list(y_val)}),
        x_col="filename",
        y_col="class",
        target_size=(224, 224),
        batch_size=32,
        class_mode="raw",
    )

    # Build model
    model = build_model(num_classes=labels.shape[1])
    logger.info("Model architecture built successfully.")

    # Callbacks, including TensorBoard
    checkpoint = ModelCheckpoint(
        "src/models/jewelry_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    early_stop = EarlyStopping(monitor="val_accuracy", patience=10, verbose=1)
    tensorboard_callback = TensorBoard(log_dir="src/logs", histogram_freq=1)

    # Train model with TensorBoard callback
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop, tensorboard_callback],
    )
    logger.info("Model training completed.")

    # Plot training history
    plot_training_history(history)

    # Save model metadata
    training_metadata = {
        "model_name": "EfficientNetB0",
        "version": "1.0",
        "training_date": pd.Timestamp.now().isoformat(),
        "parameters": {
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": len(history.history["accuracy"]),
        },
    }
    save_model_metadata(training_metadata)


if __name__ == "__main__":
    main()
