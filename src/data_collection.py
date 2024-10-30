import os
import requests
from bs4 import BeautifulSoup
import logging
from db_connection import session
import psycopg2
import json

def scrape_ebay(category_url, headers, max_pages=5):
    """
    Scrape jewelry items from eBay.
    """
    items = []
    for page in range(1, max_pages + 1):
        url = f"{category_url}&_pgn={page}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            logging.error(f"Failed to retrieve page {page}: Status code {response.status_code}")
            continue
        soup = BeautifulSoup(response.text, 'html.parser')
        listings = soup.find_all('li', {'class': 's-item'})
        for listing in listings:
            try:
                name = listing.find('h3', {'class': 's-item__title'}).text
                price = listing.find('span', {'class': 's-item__price'}).text
                url = listing.find('a', {'class': 's-item__link'})['href']
                image_url = listing.find('img', {'class': 's-item__image-img'})['src']
                condition = listing.find('span', {'class': 'SECONDARY_INFO'}).text if listing.find('span', {'class': 'SECONDARY_INFO'}) else 'Unknown'
                item = {
                    'name': name,
                    'price': price,
                    'url': url,
                    'image_url': image_url,
                    'condition': condition
                }
                items.append(item)
            except Exception as e:
                logging.error(f"Error parsing listing: {e}")
                continue
    return items

def store_jewelry_items(items):
    """
    Store scraped jewelry items into PostgreSQL.
    """
    try:
        cursor = session.connection().cursor()
        for item in items:
            cursor.execute(
                \"""
                INSERT INTO jewelry_items (name, price, url, image_url, condition)
                VALUES (%s, %s, %s, %s, %s)
                \""",
                (item['name'], item['price'], item['url'], item['image_url'], item['condition'])
            )
        session.connection().commit()
        cursor.close()
        logging.info(f"Stored {len(items)} jewelry items in database")
    except Exception as e:
        logging.error(f"Error storing jewelry items: {e}")

def main():
    """
    Main function to scrape and store jewelry items from eBay.
    """
    category_url = "https://www.ebay.com/sch/i.html?_nkw=jewelry&_sacat=0"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    items = scrape_ebay(category_url, headers, max_pages=3)
    store_jewelry_items(items)

if __name__ == "__main__":
    main()
