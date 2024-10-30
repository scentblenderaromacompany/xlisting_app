import os
import random
import json
import logging
from io import BytesIO
from PIL import Image, ImageEnhance
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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
Session = sessionmaker(bind=engine)
session = Session()


def fetch_images():
    """Fetch images that haven't been preprocessed from the database."""
    query = "SELECT image_id, image_path FROM images WHERE preprocessed = FALSE;"
    return session.execute(query).fetchall()


def preprocess_image(image_url):
    """Process the image by resizing, enhancing brightness, and converting to RGB."""
    response = requests.get(image_url, timeout=10)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))  # Resize to 224x224
    img = ImageEnhance.Brightness(img).enhance(1.2)  # Enhance brightness
    return img.convert("RGB")  # Convert to RGB


def save_image(img, path):
    """Save image to the specified path, ensuring directories exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, format="JPEG")


def augment_image(img, image_id):
    """Create augmentations and save them, returning details for the database."""
    augmentations = [
        img.rotate(random.randint(-30, 30)),
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.transpose(Image.FLIP_TOP_BOTTOM),
        img.resize((random.randint(200, 250), random.randint(200, 250))),
        ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2)),
    ]
    augmented_images = []
    for aug_img in augmentations:
        aug_path = f"data/augmented/{image_id}_{random.randint(1000,9999)}.jpg"
        save_image(aug_img, aug_path)
        augmentation_details = json.dumps(
            {"augmentation": "rotation_flip_resize_contrast"}
        )
        augmented_images.append((image_id, aug_path, augmentation_details))
    return augmented_images


def main():
    images = fetch_images()
    logger.info(f"Found {len(images)} images to preprocess.")
    for image_id, image_url in images:
        try:
            img = preprocess_image(image_url)
            preprocessed_path = f"data/processed/{image_id}.jpg"
            save_image(img, preprocessed_path)
            logger.info(
                f"Preprocessed image ID {image_id} saved to {preprocessed_path}"
            )

            # Update database for preprocessed image
            update_query = "UPDATE images SET image_path = %s, preprocessed = TRUE WHERE image_id = %s;"
            session.execute(update_query, (preprocessed_path, image_id))

            # Data augmentation and insertion into augmented_images table
            augmented_images = augment_image(img, image_id)
            for aug in augmented_images:
                insert_query = """
                    INSERT INTO augmented_images (original_image_id, augmented_image_path, augmentation_details)
                    VALUES (%s, %s, %s)
                """
                session.execute(insert_query, aug)
                logger.info(f"Augmented image saved to {aug[1]} with details {aug[2]}")

            # Mark image as augmented
            mark_augmented_query = (
                "UPDATE images SET augmented = TRUE WHERE image_id = %s;"
            )
            session.execute(mark_augmented_query, (image_id,))
            session.commit()

        except Exception as e:
            session.rollback()
            logger.error(f"Error processing image ID {image_id}: {e}")


if __name__ == "__main__":
    main()
