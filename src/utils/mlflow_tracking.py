import mlflow
import yaml
import logging

def setup_mlflow():
    """
    Setup MLflow tracking URI from configuration.
    """
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        logging.info("MLflow tracking URI set successfully")
    except Exception as e:
        logging.error(f"Error setting up MLflow: {e}")

def log_model(model, params, metrics, artifacts=None):
    """
    Log the model, parameters, metrics, and artifacts to MLflow.
    """
    try:
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if artifacts:
                for key, path in artifacts.items():
                    mlflow.log_artifact(path, key)
            mlflow.keras.log_model(model, "model")
            logging.info("Model logged to MLflow successfully")
    except Exception as e:
        logging.error(f"Error logging model to MLflow: {e}")
