import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_database():
    """Create the database if it doesn't exist."""
    # Connect to default postgres database first
    conn = psycopg2.connect(
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        # Connect to 'postgres' database to create our app database
        database='postgres'
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Create database if it doesn't exist
    db_name = os.getenv('POSTGRES_DB')
    try:
        cursor.execute(f"CREATE DATABASE {db_name}")
        print(f"Database '{db_name}' created successfully!")
    except psycopg2.errors.DuplicateDatabase:
        print(f"Database '{db_name}' already exists.")
    except Exception as e:
        print(f"Error creating database: {str(e)}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    create_database() 