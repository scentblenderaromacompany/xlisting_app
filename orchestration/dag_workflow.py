from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import logging
import subprocess

def data_collection():
    """
    Task to collect data by running data_collection.py.
    """
    try:
        subprocess.run(["python", "src/data_collection.py"], check=True)
        logging.info("Data collection completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Data collection failed: {e}")

def data_preprocessing():
    """
    Task to preprocess data by running data_preprocessing.py.
    """
    try:
        subprocess.run(["python", "src/data_preprocessing.py"], check=True)
        logging.info("Data preprocessing completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Data preprocessing failed: {e}")

def model_training():
    """
    Task to train classification model by running jewelry_classification.py.
    """
    try:
        subprocess.run(["python", "src/jewelry_classification.py"], check=True)
        logging.info("Model training completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Model training failed: {e}")

def model_evaluation():
    """
    Task to evaluate the model. This is integrated within the training script.
    """
    logging.info("Model evaluation is integrated within the training script.")

def price_prediction_training():
    """
    Task to train price prediction model by running price_prediction.py.
    """
    try:
        subprocess.run(["python", "src/price_prediction.py"], check=True)
        logging.info("Price prediction model training completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Price prediction model training failed: {e}")

def api_listing():
    """
    Task to perform API listing. This could involve running a specific script or interacting with the API.
    """
    logging.info("API listing task initiated.")
    # Placeholder: Implement actual API listing logic if needed.

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'jewelry_listing_workflow',
    default_args=default_args,
    description='Workflow for jewelry listing automation',
    schedule_interval=timedelta(days=1),
)

t1 = PythonOperator(
    task_id='data_collection',
    python_callable=data_collection,
    dag=dag,
)

t2 = PythonOperator(
    task_id='data_preprocessing',
    python_callable=data_preprocessing,
    dag=dag,
)

t3 = PythonOperator(
    task_id='model_training',
    python_callable=model_training,
    dag=dag,
)

t4 = PythonOperator(
    task_id='model_evaluation',
    python_callable=model_evaluation,
    dag=dag,
)

t5 = PythonOperator(
    task_id='price_prediction_training',
    python_callable=price_prediction_training,
    dag=dag,
)

t6 = PythonOperator(
    task_id='api_listing',
    python_callable=api_listing,
    dag=dag,
)

# Define task dependencies
t1 >> t2 >> t3 >> t4 >> t5 >> t6
