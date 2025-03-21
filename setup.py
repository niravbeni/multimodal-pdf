"""
Setup script for the PDF Chat System

This script:
1. Creates the database if it doesn't exist
2. Creates the database tables
3. Ensures the necessary directories exist
"""
import os
import argparse
import logging
from pathlib import Path
import importlib.util
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_database():
    """Create the database if it doesn't exist."""
    logger.info("Creating database...")
    try:
        from database.create_db import create_database as db_create
        db_create()
        logger.info("Database created or already exists.")
    except Exception as e:
        logger.error(f"Error creating database: {str(e)}")
        return False
    return True

def create_tables():
    """Create database tables if they don't exist."""
    logger.info("Creating database tables...")
    try:
        from database.models import Base
        from database.config import engine
        
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully.")
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        return False
    return True

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "pdf_database",   # For storing PDFs
        "temp_pdf_files", # For temporarily storing uploaded PDFs
        "chroma_db",      # For ChromaDB vector storage
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory '{directory}' created or already exists.")
        except Exception as e:
            logger.error(f"Error creating directory '{directory}': {str(e)}")
    
    return True

def check_environment():
    """Check if the environment is properly set up."""
    logger.info("Checking environment...")
    
    # Check OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("OpenAI API key not found in environment variables.")
        logger.info("Please add your OpenAI API key to the .env file or .streamlit/secrets.toml file.")
    else:
        logger.info("OpenAI API key found.")
    
    # Check if required packages are installed
    required_packages = [
        "streamlit", 
        "langchain", 
        "openai", 
        "sqlalchemy", 
        "chromadb",
        "PyPDF2",
        "pandas"
    ]
    
    missing_packages = []
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Please run: pip install -r requirements.txt")
    else:
        logger.info("All required packages are installed.")
    
    return len(missing_packages) == 0

def main():
    parser = argparse.ArgumentParser(description='Setup the PDF Chat System')
    parser.add_argument('--skip-db-creation', action='store_true', help='Skip database creation')
    parser.add_argument('--skip-table-creation', action='store_true', help='Skip table creation')
    args = parser.parse_args()
    
    logger.info("Starting setup...")
    
    # Check environment
    check_environment()
    
    # Create directories
    create_directories()
    
    # Create database
    if not args.skip_db_creation:
        if not create_database():
            logger.error("Failed to create database. Aborting setup.")
            return 1
    
    # Create tables
    if not args.skip_table_creation:
        if not create_tables():
            logger.error("Failed to create tables. Aborting setup.")
            return 1
    
    logger.info("Setup completed successfully!")
    logger.info("\nTo process PDFs and add them to the database:")
    logger.info("  python pdf_processor.py --input /path/to/pdf/folder")
    logger.info("\nTo start the application:")
    logger.info("  streamlit run app.py")
    
    return 0

if __name__ == "__main__":
    exit(main())