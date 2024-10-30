#!/bin/bash
# Setup environment (for Unix-based systems)

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Initialize the database
psql -U postgres -d jewelry_db -f database/db_schema.sql

echo "Environment setup complete."
