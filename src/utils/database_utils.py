import psycopg2
import yaml
import logging

def execute_query(query, params=None):
    """
    Execute a SQL query with optional parameters.
    """
    try:
        with psycopg2.connect(dbname='jewelry_db', user='postgres', password='password', host='db') as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    return cursor.fetchall()
                conn.commit()
    except Exception as e:
        logging.error(f"Database query failed: {e}")
        return None
