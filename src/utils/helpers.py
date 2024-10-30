import yaml
import logging

def load_config():
    """
    Load configuration from config.yaml.
    """
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading config.yaml: {e}")
        return {}
