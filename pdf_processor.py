"""
PDF Processor - All-in-one script for managing PDFs in the database

This script:
1. Imports PDFs from a specified folder
2. Extracts text and metadata
3. Generates summaries
4. Stores everything in the database

Usage:
python pdf_processor.py --input /path/to/pdf/folder
"""
import os
import hashlib
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

import PyPDF2
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.schema.document import Document

from database.models import PDF, PDFChunk
from database.config import SessionLocal
from utils.text_processor import process_pdf_text

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def parse_pdf_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse PDF date string into datetime object."""
    if not date_str:
        return None
    
    try:
        # Handle common PDF date format: D:YYYYMMDDHHmmSS or D:YYYYMMDDHHmmSS+HH'mm'
        if date_str.startswith('D:'):
            date_str = date_str[2:]
            # Extract the basic date components
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(date_str[8:10]) if len(date_str) > 8 else 0
            minute = int(date_str[10:12]) if len(date_str) > 10 else 0
            second = int(date_str[12:14]) if len(date_str) > 12 else 0
            
            return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except (ValueError, IndexError):
        pass
    
    return None

def extract_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from PDF file."""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            info = reader.metadata or {}
            
            # Clean up and parse the metadata
            creation_date = parse_pdf_date(info.get('/CreationDate', None))
            mod_date = parse_pdf_date(info.get('/ModDate', None))
            
            return {
                'title': info.get('/Title', None),
                'author': info.get('/Author', None),
                'creation_date': creation_date,
                'last_modified': mod_date,
                'page_count': len(reader.pages)
            }
    except Exception as e:
        logger.warning(f"Error extracting metadata from {file_path}: {str(e)}")
        return {
            'title': None,
            'author': None,
            'creation_date': None,
            'last_modified': None,
            'page_count': 0
        }

def extract_full_text(file_path: str) -> str:
    """Extract full text content from a PDF file."""
    full_text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"
        return full_text
    except Exception as e:
        logger.warning(f"Error extracting full text from {file_path}: {str(e)}")
        return ""

def generate_summary(text: str, model: ChatOpenAI) -> str:
    """Generate a summary of the given text."""
    if not text:
        return "No text available to summarize."
    
    try:
        # Take first 16,000 characters (about 4,000 tokens)
        text_sample = text[:16000]
        doc = Document(page_content=text_sample)
        
        # Create summarization chain
        chain = load_summarize_chain(model, chain_type="stuff")
        
        # Generate summary
        try:
            summary = chain.run([doc])
            return f"{summary}\n\n(Summary based on first section of document)"
        except Exception as e:
            # If failed, try with shorter text
            logger.warning(f"Failed summarizing with 16k chars, trying 8k: {str(e)}")
            text_sample = text[:8000]
            doc = Document(page_content=text_sample)
            summary = chain.run([doc])
            return f"{summary}\n\n(Summary based on first section of document)"
            
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return "Error generating summary."

def process_pdf(
    db,
    file_path: str,
    model: ChatOpenAI,
    keep_original: bool = True
) -> Optional[PDF]:
    """Process a PDF file: import, extract text, generate summary, and store in database."""
    logger.info(f"Processing {file_path}")
    
    try:
        # Calculate file hash to check for duplicates
        content_hash = calculate_file_hash(file_path)
        
        # Check if PDF already exists in database
        existing_pdf = db.query(PDF).filter(PDF.content_hash == content_hash).first()
        if existing_pdf:
            logger.info(f"PDF {file_path} already exists in database")
            
            # Check if summary exists, if not, generate it
            if not existing_pdf.summary:
                logger.info(f"Generating summary for existing PDF {existing_pdf.filename}")
                
                # Check if we have the full text, if not extract it
                if not existing_pdf.full_text:
                    logger.info(f"Extracting full text for {existing_pdf.filename}")
                    existing_pdf.full_text = extract_full_text(file_path)
                    
                # Generate summary
                existing_pdf.summary = generate_summary(existing_pdf.full_text, model)
                db.commit()
                logger.info(f"Summary generated for existing PDF {existing_pdf.filename}")
                
            return existing_pdf
        
        # Extract metadata
        metadata = extract_pdf_metadata(file_path)
        
        # Extract full text
        full_text = extract_full_text(file_path)
        
        # Generate summary
        summary = generate_summary(full_text, model)
        
        # Create new PDF record
        new_pdf = PDF(
            filename=os.path.basename(file_path),
            file_path=file_path if keep_original else None,
            content_hash=content_hash,
            file_size=os.path.getsize(file_path),
            page_count=metadata['page_count'],
            title=metadata['title'],
            author=metadata['author'],
            creation_date=metadata['creation_date'],
            last_modified=metadata['last_modified'],
            uploaded_at=datetime.now(timezone.utc),
            full_text=full_text,
            summary=summary
        )
        
        # Process PDF text and create chunks
        documents = process_pdf_text([file_path])
        if documents:
            for idx, doc in enumerate(documents):
                chunk = PDFChunk(
                    chunk_index=idx,
                    content=doc.page_content,
                    page_number=doc.metadata.get('page', 0)
                )
                new_pdf.chunks.append(chunk)
        
        # Add to database
        db.add(new_pdf)
        db.commit()
        
        logger.info(f"Successfully processed and imported {file_path}")
        return new_pdf
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        db.rollback()
        return None

def process_pdf_directory(
    input_dir: str,
    keep_originals: bool = True,
    move_processed: bool = False,
    processed_dir: str = None
):
    """Process all PDFs from a directory and import them to the database."""
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"Input directory {input_dir} does not exist or is not a directory")
        return

    # Create processed directory if needed
    if move_processed and processed_dir:
        processed_path = Path(processed_dir)
        processed_path.mkdir(parents=True, exist_ok=True)

    # Initialize database session
    db = SessionLocal()
    
    # Initialize language model
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    try:
        # Find all PDF files in the directory
        pdf_files = list(input_path.glob("**/*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")
        
        # Process each PDF
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"Processing file {i}/{len(pdf_files)}: {pdf_path}")
            
            pdf = process_pdf(db, str(pdf_path), model, keep_originals)
            
            # Move processed file if requested
            if pdf and move_processed and processed_dir:
                target_path = Path(processed_dir) / pdf_path.name
                try:
                    if not keep_originals:
                        # Move the file
                        pdf_path.rename(target_path)
                        logger.info(f"Moved {pdf_path} to {target_path}")
                    else:
                        # Copy the file
                        import shutil
                        shutil.copy2(pdf_path, target_path)
                        logger.info(f"Copied {pdf_path} to {target_path}")
                except Exception as e:
                    logger.error(f"Error moving/copying file {pdf_path}: {str(e)}")
            
            # Add a delay to prevent rate limiting for the AI model
            time.sleep(2)
            
    finally:
        db.close()

def main():
    parser = argparse.ArgumentParser(description='Process PDF files and import them to the database with summaries')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing PDF files')
    parser.add_argument('--keep', '-k', action='store_true', default=True, help='Keep original PDF files (store path in database)')
    parser.add_argument('--move', '-m', action='store_true', help='Move processed files to a different directory')
    parser.add_argument('--processed-dir', '-p', help='Directory to move processed files to')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.move and not args.processed_dir:
        parser.error("--move requires --processed-dir")
    
    # Process PDFs
    process_pdf_directory(
        args.input,
        keep_originals=args.keep,
        move_processed=args.move,
        processed_dir=args.processed_dir
    )

if __name__ == "__main__":
    main() 