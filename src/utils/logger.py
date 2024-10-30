from loguru import logger
import yaml
import sys

def setup_logger():
    """
    Configure logger using Loguru based on config.yaml.
    """
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Remove default logger
        logger.remove()
        
        # Add file logger with rotation and retention
        logger.add(config['logging']['file'], rotation="1 MB", retention="10 days", level=config['logging']['level'])
        
        # Add stderr logger
        logger.add(sys.stderr, level=config['logging']['level'])
    except Exception as e:
        print(f"Error setting up logger: {e}")
        sys.exit(1)

setup_logger()
