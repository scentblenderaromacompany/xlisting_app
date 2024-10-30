# Base image
FROM python:3.9-slim

# Set environment variables to prevent prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system dependencies
RUN apt-get update && \\
    apt-get install -y build-essential libpq-dev libopencv-dev wget unzip && \\
    apt-get clean && \\
    rm -rf /var/lib/apt/lists/*

# Install Git for Airflow
RUN apt-get update && \\
    apt-get install -y git && \\
    apt-get clean && \\
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download and install Airflow
ENV AIRFLOW_HOME=/usr/local/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False

# Initialize Airflow
RUN airflow db init

# Copy the rest of the application code
COPY . /app

# Expose the FastAPI port
EXPOSE 8000

# Expose Airflow port
EXPOSE 8080

# Expose MLflow port
EXPOSE 5000

# Command to run the FastAPI application and Airflow scheduler and webserver
CMD ["sh", "-c", "uvicorn src.api_integration:app --host 0.0.0.0 --port 8000 & airflow scheduler & airflow webserver"]
