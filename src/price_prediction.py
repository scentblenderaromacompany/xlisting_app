import os
import psycopg2
import json
import pickle
import logging
from sqlalchemy.orm import sessionmaker
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from db_connection import engine, Session
from utils.logger import logger
from utils.mlflow_tracking import log_model

# Initialize session
session = Session()

def load_price_data():
    """
    Load historical pricing data from PostgreSQL.
    """
    try:
        cursor = session.connection().cursor()
        cursor.execute(\"""
            SELECT attributes, price
            FROM jewelry_items
        \""")
        data = cursor.fetchall()
        features = []
        prices = []
        for row in data:
            attributes = json.loads(row[0])
            price_str = row[1].replace('$', '').replace(',', '').strip()
            try:
                price = float(price_str)
            except ValueError:
                price = None
            if price is None:
                continue
            # Extract relevant features
            feature = {
                'type': attributes.get('type', 'unknown'),
                'material': attributes.get('material', 'unknown'),
                'color': attributes.get('color', 'unknown'),
                'brand': attributes.get('brand', 'unknown'),
                'stone': attributes.get('stone', 'unknown'),
                'stone_color': attributes.get('stone_color', 'unknown'),
                'weight': attributes.get('weight', 0.0),
                'metal_purity': attributes.get('metal_purity', 'unknown'),
                'origin': attributes.get('origin', 'unknown'),
                'style': attributes.get('style', 'unknown'),
                'date_made': attributes.get('date_made', '2000-01-01')
            }
            features.append(feature)
            prices.append(price)
        cursor.close()
        return features, prices
    except Exception as e:
        logger.error(f"Error loading price data from database: {e}")
        return None, None

def encode_features(features):
    """
    Encode categorical features using LabelEncoder and handle numerical features.
    """
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    
    df = pd.DataFrame(features)
    
    # Initialize encoders
    encoders = {}
    categorical_features = ['type', 'material', 'color', 'brand', 'stone', 'stone_color', 'metal_purity', 'origin', 'style']
    
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        encoders[feature] = le
    
    # Convert date_made to numerical (e.g., year)
    df['date_made'] = pd.to_datetime(df['date_made'], errors='coerce').dt.year.fillna(2000).astype(int)
    
    return df, encoders

def train_price_model():
    """
    Train the price prediction model.
    """
    features, prices = load_price_data()
    if features is None or prices is None:
        logger.error("No data loaded. Exiting training.")
        return
    
    import pandas as pd
    X, encoders = encode_features(features)
    y = pd.Series(prices)
    
    # Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    
    # Save the model
    with open('models/price_predictor/price_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save encoders
    with open('models/price_predictor/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    # Save performance metrics to PostgreSQL
    try:
        cursor = session.connection().cursor()
        cursor.execute(\"""
            INSERT INTO price_model_performance (mae, rmse)
            VALUES (%s, %s)
        \""", (mae, rmse))
        session.connection().commit()
        cursor.close()
        logger.info("Price model performance metrics stored in database")
    except Exception as e:
        logger.error(f"Error storing price model performance metrics: {e}")
    
    # Log model to MLflow
    metrics = {
        'mae': mae,
        'rmse': rmse
    }
    params = {
        'model_type': 'RandomForestRegressor',
        'n_estimators': 100,
        'random_state': 42
    }
    log_model(model, params, metrics)

if __name__ == "__main__":
    train_price_model()
