from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests
import json
import psycopg2
from db_connection import engine, Session
import yaml
from loguru import logger
from utils.helpers import load_config

app = FastAPI()

# Load configuration
config = load_config()

# Initialize session
session = Session()

class ProductListing(BaseModel):
    listing: dict
    price: float

def get_api_keys():
    return config.get('api_keys', {})

def store_submission(listing_data, price, ebay_response, etsy_response, shopify_response):
    try:
        cursor = session.connection().cursor()
        cursor.execute(\"""
            INSERT INTO product_listings (listing_data, price, ebay_response, etsy_response, shopify_response)
            VALUES (%s, %s, %s, %s, %s)
        \""", (
            json.dumps(listing_data),
            price,
            json.dumps(ebay_response) if ebay_response else None,
            json.dumps(etsy_response) if etsy_response else None,
            json.dumps(shopify_response) if shopify_response else None
        ))
        session.connection().commit()
        cursor.close()
        logger.info("Product listing stored successfully")
    except Exception as e:
        logger.error(f"Error storing product listing: {e}")

def submit_to_ebay(api_key, listing):
    """
    Submit product listing to eBay via API.
    """
    try:
        # Example eBay API endpoint for inventory items
        url = "https://api.ebay.com/sell/inventory/v1/inventory_item"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, json=listing)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"eBay API Error: {e}")
        return None

def submit_to_etsy(api_key, listing):
    """
    Submit product listing to Etsy via API.
    """
    try:
        # Example Etsy API endpoint for creating listings
        url = "https://openapi.etsy.com/v2/listings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, json=listing)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Etsy API Error: {e}")
        return None

def submit_to_shopify(api_key, listing):
    """
    Submit product listing to Shopify via API.
    """
    try:
        # Replace 'yourshopifydomain' with your actual Shopify domain
        shopify_domain = "yourshopifydomain.myshopify.com"
        url = f"https://{shopify_domain}/admin/api/2021-04/products.json"
        headers = {
            "X-Shopify-Access-Token": api_key,
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, json=listing)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Shopify API Error: {e}")
        return None

@app.post("/list_product/")
def list_product(product_data: ProductListing):
    api_keys = get_api_keys()
    listing = product_data.listing
    price = product_data.price
    
    # Submit to eBay
    ebay_response = submit_to_ebay(api_keys.get('ebay'), listing)
    
    # Submit to Etsy
    etsy_response = submit_to_etsy(api_keys.get('etsy'), listing)
    
    # Submit to Shopify
    shopify_response = submit_to_shopify(api_keys.get('shopify'), listing)
    
    # Store submission logs in PostgreSQL
    store_submission(listing, price, ebay_response, etsy_response, shopify_response)
    
    return {
        "status": "success", 
        "ebay_response": ebay_response, 
        "etsy_response": etsy_response, 
        "shopify_response": shopify_response
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
