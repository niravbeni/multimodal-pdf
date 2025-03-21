"""
Process PDFs from the database and generate summaries
"""
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from database.models import PDF
from database.config import SessionLocal
from utils.text_processor import summarize_text_chunks
from langchain.schema.document import Document
import logging
from langchain.chains.summarize import load_summarize_chain

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def process_pdfs_in_database():
    """Process all PDFs in the database to generate summaries."""
    db = SessionLocal()
    
    # Get all PDFs
    pdfs = db.query(PDF).all()
    logger.info(f"Found {len(pdfs)} PDFs to process")
    
    # Initialize the model and chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = load_summarize_chain(llm, chain_type="stuff")
    
    for i, pdf in enumerate(pdfs, 1):
        logger.info(f"Processing PDF {i}/{len(pdfs)}: {pdf.filename}")
        
        if not pdf.full_text:
            logger.warning(f"No full text available for {pdf.filename}, skipping...")
            continue
            
        try:
            # Take first 16,000 characters (about 4,000 tokens)
            text = pdf.full_text[:16000]
            doc = Document(page_content=text)
            
            try:
                # Try to generate summary
                summary = chain.run([doc])
                pdf.summary = f"{summary}\n\n(Summary based on first section of document)"
                db.commit()
                logger.info(f"Successfully generated summary for {pdf.filename}")
            except Exception as e:
                # If failed, try with shorter text
                logger.warning(f"Failed with 16k chars, trying 8k for {pdf.filename}: {str(e)}")
                text = pdf.full_text[:8000]
                doc = Document(page_content=text)
                summary = chain.run([doc])
                pdf.summary = f"{summary}\n\n(Summary based on first section of document)"
                db.commit()
                logger.info(f"Successfully generated summary with shorter text for {pdf.filename}")
                
            # Add delay to avoid rate limits
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error processing {pdf.filename}: {str(e)}")
            db.rollback()
            continue
            
    db.close()

if __name__ == "__main__":
    process_pdfs_in_database() 