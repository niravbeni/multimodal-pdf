from typing import List, Dict, Any
from sqlalchemy import or_
from sqlalchemy.orm import Session
from models import PDF, PDFChunk

def search_pdfs(
    db: Session,
    query: str,
    limit: int = 10,
    include_chunks: bool = True
) -> List[Dict[Any, Any]]:
    """
    Search PDFs based on a text query.
    Returns a list of PDFs with their relevance scores and matching chunks.
    """
    # Convert query to lowercase for case-insensitive search
    query = query.lower()
    
    # Search in PDF metadata and chunks
    results = []
    
    # Build the base query
    pdf_query = db.query(PDF)
    
    # Search in PDF metadata
    metadata_matches = pdf_query.filter(
        or_(
            PDF.title.ilike(f"%{query}%"),
            PDF.full_text.ilike(f"%{query}%"),
            PDF.summary.ilike(f"%{query}%")
        )
    ).all()
    
    # Search in PDF chunks if requested
    if include_chunks:
        chunk_matches = db.query(PDFChunk).filter(
            PDFChunk.content.ilike(f"%{query}%")
        ).all()
        
        # Get unique PDFs from chunk matches
        chunk_pdf_ids = set(chunk.pdf_id for chunk in chunk_matches)
        chunk_pdfs = pdf_query.filter(PDF.id.in_(chunk_pdf_ids)).all()
        
        # Combine results, removing duplicates
        all_pdfs = list({pdf.id: pdf for pdf in metadata_matches + chunk_pdfs}.values())
    else:
        all_pdfs = metadata_matches
    
    # Process results
    for pdf in all_pdfs:
        # Find matching chunks for this PDF
        matching_chunks = []
        if include_chunks:
            matching_chunks = [
                {
                    'content': chunk.content,
                    'page_number': chunk.page_number
                }
                for chunk in pdf.chunks
                if query in chunk.content.lower()
            ]
        
        # Calculate a simple relevance score
        # This could be made more sophisticated with proper text similarity
        score = 0
        if pdf.title and query in pdf.title.lower():
            score += 3
        if pdf.summary and query in pdf.summary.lower():
            score += 2
        score += len(matching_chunks) * 0.5
        
        results.append({
            'id': pdf.id,
            'filename': pdf.filename,
            'title': pdf.title or pdf.filename,
            'summary': pdf.summary,
            'relevance_score': score,
            'matching_chunks': matching_chunks[:3],  # Limit to top 3 matching chunks
            'total_matches': len(matching_chunks)
        })
    
    # Sort by relevance score
    results.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return results[:limit] 