from typing import List, Dict, Any
import pandas as pd
from sqlalchemy.orm import Session
from database.models import PDF
from database.config import SessionLocal

def get_pdf_metadata_from_db() -> pd.DataFrame:
    """Get metadata for all PDFs from the database"""
    db = SessionLocal()
    try:
        # Query all PDFs from the database
        pdfs = db.query(PDF).all()
        
        # Convert to list of dictionaries
        pdf_data = []
        for pdf in pdfs:
            # Extract organization name (text before first dash or underscore)
            org_name = pdf.filename.split('-')[0].split('_')[0].strip()
            
            # Get file size in MB
            size_mb = pdf.file_size / (1024 * 1024)
            
            # Get just the date part of last_modified
            last_modified_date = (pdf.last_modified or pdf.uploaded_at).date()
            
            pdf_data.append({
                'Selected': False,
                'Organization': org_name,
                'Filename': pdf.filename,
                'Size (MB)': round(size_mb, 2),
                'Last Modified': last_modified_date,
                'Path': pdf.file_path,
                'ID': pdf.id  # Keep ID for internal use
            })
        
        return pd.DataFrame(pdf_data)
    finally:
        db.close()

def get_selected_pdfs(db: Session, pdf_ids: List[int]) -> List[PDF]:
    """Get PDF objects by their IDs"""
    return db.query(PDF).filter(PDF.id.in_(pdf_ids)).all() 