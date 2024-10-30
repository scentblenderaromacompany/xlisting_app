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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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
DB_PASSWORD = os.getenv("DB_PASSWORD")  # Ensure secure password handling

# Connect to PostgreSQL
engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# Fetch preprocessed image paths and attributes
def fetch_data():
    query = """
    SELECT images.image_path, attributes.type, attributes.material, attributes.color, attributes.brand, attributes.metal
    FROM images
    JOIN image_attributes ON images.image_id = image_attributes.image_id
    JOIN attributes ON image_attributes.attribute_id = attributes.attribute_id
    WHERE images.preprocessed = TRUE;
    """
    return pd.read_sql(query, engine)


def prepare_labels(df):
    label_cols = ["type", "material", "color", "brand", "metal"]
    mlb = MultiLabelBinarizer()
    df_labels = df[label_cols].apply(
        lambda row: [
            str(row[col]).lower() for col in label_cols if pd.notnull(row[col])
        ],
        axis=1,
    )
    labels = mlb.fit_transform(df_labels)
    return labels, mlb.classes_


def build_model(num_classes):
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="sigmoid")(
        x
    )  # Multi-label classification
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

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
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

    # Callbacks
    checkpoint = ModelCheckpoint(
        "src/models/jewelry_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    early_stop = EarlyStopping(monitor="val_accuracy", patience=10, verbose=1)

    # Train model
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop],
    )
    logger.info("Model training completed.")

    # Plot training history
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
            logger.info(f"Model metadata inserted with Model ID: {model_id}")
    except Exception as e:
        logger.error(f"Error inserting model metadata: {e}")


if __name__ == "__main__":
    main()
