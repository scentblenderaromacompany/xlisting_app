import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import yaml
import logging
from utils.logger import logger

def load_config():
    """
    Load configuration from config.yaml.
    """
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config.yaml: {e}")
        return {}

# Load configuration
config = load_config()

# Database URL from config
database_url = config['database']['url']

# Create the database engine
engine = create_engine(database_url)

# Create a configured "Session" class
Session = sessionmaker(bind=engine)

# Create a Session
session = Session()
