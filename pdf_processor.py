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
import random

import PyPDF2
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.schema.document import Document
from concurrent.futures import ThreadPoolExecutor
import asyncio
from langchain_core.prompts import ChatPromptTemplate

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

# Optimized batch summarization function with sampling support
def batch_summarize_chunks(
    documents: List[Document], 
    model: ChatOpenAI, 
    batch_size: int = 20, 
    max_workers: int = 10,
    sample_ratio: float = 1.0,  # 1.0 means process all, 0.5 means process 50%
    chunk_limit: Optional[int] = None  # Optional limit on total chunks to process
) -> List[str]:
    """
    Generate summaries for text chunks using highly optimized batched API calls
    
    Args:
        documents: List of Document objects
        model: LLM model to use for summarization
        batch_size: Number of documents to summarize in a single batch
        max_workers: Maximum number of concurrent workers
        sample_ratio: Fraction of chunks to process (for very large documents)
        chunk_limit: Optional maximum number of chunks to process
        
    Returns:
        List of summaries
    """
    # Determine if sampling is needed
    total_docs = len(documents)
    to_process = total_docs
    
    # Apply sampling if requested
    if sample_ratio < 1.0 or (chunk_limit and chunk_limit < total_docs):
        if chunk_limit and chunk_limit < total_docs:
            to_process = chunk_limit
        else:
            to_process = int(total_docs * sample_ratio)
        
        logger.info(f"Sampling {to_process} chunks out of {total_docs} total chunks (ratio: {sample_ratio:.2f})")
        
        # Create empty summary placeholders for all docs
        summaries = [""] * total_docs
        
        # Choose indices to process, prioritizing important pages
        if sample_ratio < 1.0:
            # Always include first few and last few pages as they often contain important info
            first_n = min(20, int(to_process * 0.2))  # 20% from beginning
            last_n = min(10, int(to_process * 0.1))   # 10% from end
            
            # Force include first and last chunks
            must_include = list(range(first_n)) + list(range(total_docs - last_n, total_docs))
            
            # Randomly sample from the middle
            middle_indices = list(range(first_n, total_docs - last_n))
            remaining_count = to_process - len(must_include)
            
            if remaining_count > 0 and middle_indices:
                random_middle = random.sample(middle_indices, min(remaining_count, len(middle_indices)))
                indices_to_process = sorted(must_include + random_middle)
            else:
                indices_to_process = sorted(must_include[:to_process])
        else:
            # If using chunk_limit but not sampling ratio
            indices_to_process = list(range(min(to_process, total_docs)))
    else:
        # Process all documents
        indices_to_process = list(range(total_docs))
        summaries = [""] * total_docs
    
    # For timing metrics
    start_time = time.time()
    processed_count = 0
    
    logger.info(f"Starting optimized batch summarization of {to_process} text chunks (out of {total_docs})")
    logger.info(f"Using batch size: {batch_size}, max workers: {max_workers}")
    
    # Process in batches
    for batch_start in range(0, len(indices_to_process), batch_size):
        batch_indices = indices_to_process[batch_start:batch_start + batch_size]
        batch_docs = [documents[i] for i in batch_indices]
        
        # Log progress with time estimates
        processed_count += len(batch_indices)
        elapsed = time.time() - start_time
        items_per_second = processed_count / elapsed if elapsed > 0 else 0
        remaining_items = to_process - processed_count
        eta_seconds = remaining_items / items_per_second if items_per_second > 0 else 0
        
        # Calculate progress percentage
        progress_pct = (processed_count / to_process) * 100
        
        # Format time remaining nicely
        if eta_seconds < 60:
            eta_str = f"{eta_seconds:.1f} seconds"
        elif eta_seconds < 3600:
            eta_str = f"{eta_seconds/60:.1f} minutes"
        else:
            eta_str = f"{eta_seconds/3600:.1f} hours"
            
        logger.info(f"Batch {batch_start//batch_size + 1}: Processing chunks {batch_start}-{batch_start + len(batch_indices) - 1} " +
                   f"({progress_pct:.1f}% complete, ETA: {eta_str})")
        
        # Process batch in parallel
        batch_results = []
        
        def summarize_single_doc(batch_idx, doc, original_idx):
            if not doc.page_content or doc.page_content.isspace():
                return original_idx, ""
            
            # Use a simple prompt for summarization
            prompt = f"""
            Create a concise summary that captures the key information in this text chunk.
            Focus on factual information and important details that would be relevant for retrieval.
            Text: {doc.page_content}
            Summary:
            """
            
            try:
                # Generate summary
                response = model.invoke(prompt)
                summary = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                return original_idx, summary
            except Exception as e:
                logger.error(f"Error generating summary for chunk {original_idx}: {str(e)}")
                return original_idx, ""
        
        # Use ThreadPoolExecutor with more workers for parallel processing
        with ThreadPoolExecutor(max_workers=min(len(batch_indices), max_workers)) as executor:
            futures = []
            for batch_idx, (doc, original_idx) in enumerate(zip(batch_docs, batch_indices)):
                futures.append(executor.submit(summarize_single_doc, batch_idx, doc, original_idx))
            
            for future in futures:
                try:
                    idx, summary = future.result()
                    batch_results.append((idx, summary))
                except Exception as e:
                    logger.error(f"Error in thread: {str(e)}")
        
        # Update summaries with results
        for idx, summary in batch_results:
            summaries[idx] = summary
        
        logger.info(f"Completed batch {batch_start//batch_size + 1}, "
                  f"processing at {items_per_second:.2f} chunks/sec")
    
    total_time = time.time() - start_time
    logger.info(f"Completed all batches in {total_time:.2f} seconds, "
              f"average: {to_process/total_time:.2f} chunks/sec")
    
    return summaries

def process_pdf(
    db,
    file_path: str,
    model: ChatOpenAI,
    keep_original: bool = True,
    speed_mode: bool = False,
    max_chunks: Optional[int] = None
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
        
        # Determine if this is a large document that might need optimization
        is_large_document = metadata['page_count'] > 100
        
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
            # Determine processing approach based on document size and speed mode
            sample_ratio = 1.0
            batch_size = 20
            max_workers = 10
            
            if is_large_document:
                if speed_mode:
                    # Ultra-fast processing for very large documents
                    if metadata['page_count'] > 500:
                        sample_ratio = 0.25  # 25% of chunks
                        batch_size = 40
                        max_workers = 20
                    elif metadata['page_count'] > 250:
                        sample_ratio = 0.5  # 50% of chunks
                        batch_size = 30
                        max_workers = 15
                else:
                    # Standard processing for large documents
                    batch_size = 30
                    max_workers = 15
            
            # Log processing approach
            logger.info(f"Using {'SPEED MODE' if speed_mode else 'STANDARD MODE'} with " +
                       f"sample_ratio={sample_ratio}, batch_size={batch_size}, max_workers={max_workers}")
            
            # Use the batch function for faster processing
            logger.info(f"Generating summaries for {len(documents)} chunks using optimized batch processing")
            chunk_summaries = batch_summarize_chunks(
                documents, 
                model, 
                batch_size=batch_size, 
                max_workers=max_workers,
                sample_ratio=sample_ratio,
                chunk_limit=max_chunks
            )
            
            # Progress tracking
            chunk_count = len(documents)
            logger.info(f"Creating {chunk_count} chunk records in database")
            
            # Create chunks in database with progress tracking
            for idx, doc in enumerate(documents):
                if idx % 100 == 0 and idx > 0:
                    # Commit in batches to avoid memory issues with very large documents
                    db.commit()
                    logger.info(f"Saved {idx}/{chunk_count} chunks to database")
                
                # Get the corresponding summary
                chunk_summary = chunk_summaries[idx] if idx < len(chunk_summaries) else ""
                
                chunk = PDFChunk(
                    chunk_index=idx,
                    content=doc.page_content,
                    page_number=doc.metadata.get('page', 0),
                    summary=chunk_summary
                )
                new_pdf.chunks.append(chunk)
            
            logger.info(f"All {chunk_count} chunks processed")
        
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
    processed_dir: str = None,
    speed_mode: bool = False,
    max_chunks_per_pdf: Optional[int] = None
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
        # Find all PDF files in the directory and its subdirectories
        start_time = time.time()
        pdf_files = list(input_path.glob("**/*.pdf"))
        pdf_count = len(pdf_files)
        logger.info(f"Found {pdf_count} PDF files in {input_dir}")
        
        # Optional: Sort PDFs by size (process smaller ones first for quick wins)
        pdf_files.sort(key=lambda p: p.stat().st_size)
        
        for i, pdf_path in enumerate(pdf_files, 1):
            file_path = str(pdf_path.absolute())
            file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
            
            logger.info(f"Processing file {i} of {pdf_count}: {pdf_path.name} ({file_size_mb:.2f} MB)")
            
            # Process the PDF with speed mode if enabled
            processed = process_pdf(
                db, 
                file_path, 
                model, 
                keep_originals, 
                speed_mode=speed_mode,
                max_chunks=max_chunks_per_pdf
            )
            
            # Move processed file if requested
            if processed and move_processed and processed_dir:
                try:
                    target_path = processed_path / pdf_path.name
                    if target_path.exists():
                        # Add timestamp to filename to avoid conflicts
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        target_path = processed_path / f"{pdf_path.stem}_{timestamp}{pdf_path.suffix}"
                    
                    pdf_path.rename(target_path)
                    logger.info(f"Moved {pdf_path.name} to {target_path}")
                    
                    # Update the file_path in the database
                    if processed and keep_originals:
                        processed.file_path = str(target_path)
                        db.commit()
                except Exception as e:
                    logger.error(f"Error moving file {pdf_path}: {str(e)}")
        
        elapsed = time.time() - start_time
        per_pdf_time = elapsed/pdf_count if pdf_count > 0 else 0
        logger.info(f"Processed {pdf_count} PDFs in {elapsed:.2f} seconds ({per_pdf_time:.2f} seconds per PDF)")
        
    except Exception as e:
        logger.error(f"Error processing PDF directory: {str(e)}")
    finally:
        db.close()

def main():
    parser = argparse.ArgumentParser(description="Process PDFs and store them in the database")
    parser.add_argument("--input", "-i", required=True, help="Directory containing PDFs to process")
    parser.add_argument("--keep", "-k", default=True, action="store_true", help="Keep original PDF files (store path in database)")
    parser.add_argument("--move", "-m", action="store_true", help="Move processed files to a different directory")
    parser.add_argument("--processed-dir", "-p", help="Directory to move processed files to (required if using --move)")
    parser.add_argument("--speed", "-s", action="store_true", help="Enable speed mode for faster processing (uses sampling for large documents)")
    parser.add_argument("--max-chunks", "-c", type=int, help="Maximum number of chunks to process per PDF (for very large documents)")
    
    args = parser.parse_args()
    
    if args.move and not args.processed_dir:
        parser.error("--processed-dir is required when using --move")
    
    process_pdf_directory(
        input_dir=args.input,
        keep_originals=args.keep,
        move_processed=args.move,
        processed_dir=args.processed_dir,
        speed_mode=args.speed,
        max_chunks_per_pdf=args.max_chunks
    )

if __name__ == "__main__":
    main() 