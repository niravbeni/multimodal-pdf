"""
Extract text from PDFs and store it in the database
"""
import os
import logging
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from database.models import PDF
from database.config import SessionLocal
from utils.text_processor import extract_text_from_pdf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        return None

def process_pdfs_in_database():
    """Extract text from all PDFs in the database."""
    db = SessionLocal()
    
    # Get all PDFs
    pdfs = db.query(PDF).all()
    logger.info(f"Found {len(pdfs)} PDFs to process")
    
    for i, pdf in enumerate(pdfs, 1):
        logger.info(f"Processing PDF {i}/{len(pdfs)}: {pdf.filename}")
        
        try:
            # Construct the file path
            file_path = os.path.join("pdf_database", pdf.filename)
            
            if not os.path.exists(file_path):
                logger.warning(f"PDF file not found: {file_path}")
                continue
                
            # Extract text
            text = extract_text_from_pdf(file_path)
            if text:
                pdf.full_text = text
                db.commit()
                logger.info(f"Successfully extracted text from {pdf.filename}")
            else:
                logger.warning(f"No text extracted from {pdf.filename}")
                
        except Exception as e:
            logger.error(f"Error processing {pdf.filename}: {str(e)}")
            db.rollback()
            continue
            
    db.close()

if __name__ == "__main__":
    process_pdfs_in_database() 