import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import PyPDF2
from sqlalchemy.orm import Session

from models import PDF, PDFChunk
from config import SessionLocal
from utils.text_processor import process_pdf_text

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

def extract_pdf_metadata(file_path: str) -> dict:
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
        print(f"Warning: Error extracting metadata: {str(e)}")
        return {
            'title': None,
            'author': None,
            'creation_date': None,
            'last_modified': None,
            'page_count': 0
        }

def migrate_pdf_to_db(
    db: Session,
    file_path: str,
    keep_original: bool = True
) -> Optional[PDF]:
    """Migrate a single PDF file to the database."""
    try:
        # Calculate file hash to check for duplicates
        content_hash = calculate_file_hash(file_path)
        
        # Check if PDF already exists in database
        existing_pdf = db.query(PDF).filter(PDF.content_hash == content_hash).first()
        if existing_pdf:
            print(f"PDF {file_path} already exists in database")
            return existing_pdf
        
        # Extract metadata
        metadata = extract_pdf_metadata(file_path)
        
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
            uploaded_at=datetime.now(timezone.utc)
        )
        
        # Process PDF text and create chunks
        chunks = process_pdf_text(file_path)
        for idx, (content, page_num) in enumerate(chunks):
            chunk = PDFChunk(
                chunk_index=idx,
                content=content,
                page_number=page_num
            )
            new_pdf.chunks.append(chunk)
        
        # Add to database
        db.add(new_pdf)
        db.commit()
        
        print(f"Successfully migrated {file_path}")
        return new_pdf
        
    except Exception as e:
        print(f"Error migrating {file_path}: {str(e)}")
        db.rollback()
        return None

def migrate_all_pdfs(pdf_dir: str, keep_originals: bool = True):
    """Migrate all PDFs from a directory to the database."""
    db = SessionLocal()
    try:
        pdf_files = Path(pdf_dir).glob("**/*.pdf")
        for pdf_path in pdf_files:
            migrate_pdf_to_db(db, str(pdf_path), keep_originals)
    finally:
        db.close()

if __name__ == "__main__":
    PDF_DIR = "pdf_database"
    print(f"Starting migration of PDFs from {PDF_DIR}...")
    migrate_all_pdfs(PDF_DIR)
    print("Migration completed!") 