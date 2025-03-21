"""
Text processor for PDF files
Contains functions for extracting and processing text from PDF files
"""
import os
from typing import List, Dict, Optional, Tuple
import logging
import warnings
import traceback
import PyPDF2  # Add PyPDF2 import for accurate page counting

from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

def extract_text_from_pdf(pdf_path: str) -> List[Tuple[str, int]]:
    """
    Extract text from a PDF file using PyPDF2 for accurate physical page numbers
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of tuples containing (text, page_number)
    """
    try:
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        # First try extracting with PyPDF2 for accurate physical page numbers
        try:
            pages_text = []
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                pdf_page_count = len(pdf_reader.pages)
                logger.info(f"PDF has {pdf_page_count} physical pages according to PyPDF2")
                
                # Extract text from each physical page directly
                for i in range(pdf_page_count):
                    physical_page_num = i + 1  # 1-indexed page number
                    try:
                        page_text = pdf_reader.pages[i].extract_text()
                        if page_text and not page_text.isspace():
                            pages_text.append((page_text, physical_page_num))
                            logger.info(f"Extracted text from physical page {physical_page_num}: {len(page_text)} chars")
                    except Exception as e:
                        logger.warning(f"Could not extract text from page {physical_page_num}: {str(e)}")
                
                # If we successfully extracted text from any pages, return it
                if pages_text:
                    logger.info(f"Successfully extracted text from {len(pages_text)} physical pages using PyPDF2")
                    return pages_text
                else:
                    logger.warning("No text extracted with PyPDF2, falling back to unstructured")
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {str(e)}, falling back to unstructured")
        
        # If PyPDF2 failed, use unstructured with physical page mapping
        elements = partition_pdf(
            filename=pdf_path,
            strategy="fast",
            extract_images_in_pdf=False,
            infer_table_structure=False,
            include_page_breaks=True,
        )
        
        logger.info(f"Extracted {len(elements)} elements from PDF using unstructured")
        
        # Group elements by the metadata page number first
        raw_page_elements = {}
        
        for element in elements:
            metadata_page = 1  # Default to page 1
            
            if hasattr(element, 'metadata') and element.metadata:
                metadata = element.metadata
                # Try to get page number from metadata
                for key in ['page_number', 'page_num', 'page']:
                    if key in metadata and metadata[key] is not None:
                        try:
                            metadata_page = int(metadata[key])
                            break
                        except (ValueError, TypeError):
                            pass
            
            # Add element to its metadata page group
            if metadata_page not in raw_page_elements:
                raw_page_elements[metadata_page] = []
            raw_page_elements[metadata_page].append(str(element))
        
        # Convert metadata page numbers to physical page numbers
        # Sort metadata pages to ensure they map to consecutive physical pages
        metadata_pages = sorted(raw_page_elements.keys())
        logger.info(f"Found {len(metadata_pages)} unique metadata pages: {metadata_pages}")
        
        # Map metadata pages to physical pages (1-indexed)
        page_mapping = {}
        for i, metadata_page in enumerate(metadata_pages):
            physical_page = i + 1  # 1-indexed physical page
            page_mapping[metadata_page] = physical_page
            logger.info(f"Mapping metadata page {metadata_page} to physical page {physical_page}")
        
        # Now create the actual physical page content
        page_elements = {}
        for metadata_page, elements_text in raw_page_elements.items():
            physical_page = page_mapping.get(metadata_page, 1)  # Default to page 1 if not in mapping
            
            if physical_page not in page_elements:
                page_elements[physical_page] = []
            page_elements[physical_page].extend(elements_text)
        
        # Convert to list of (text, page_number) tuples
        pages_text = []
        for page_num in sorted(page_elements.keys()):
            text = "\n\n".join(page_elements[page_num]).strip()
            if text:  # Only add if there's text content
                pages_text.append((text, page_num))
                logger.info(f"Added physical page {page_num} with {len(text)} characters")
        
        return pages_text
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        traceback.print_exc()
        return []

def split_text(text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split text into chunks using RecursiveCharacterTextSplitter
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Document objects
    """
    if not text or text.isspace():
        return []
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split text
    chunks = text_splitter.split_text(text)
    
    # Debug logging
    logger.info(f"Original text length: {len(text)}")
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i+1} length: {len(chunk)}")
    
    # Convert to Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    return documents

def process_pdf_text(pdf_paths: List[str]) -> List[Document]:
    """
    Process PDF files by extracting text and splitting into chunks
    
    Args:
        pdf_paths: List of paths to PDF files
        
    Returns:
        List of Document objects containing text chunks
    """
    all_documents = []
    
    # Process each PDF file
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF file not found: {pdf_path}")
            continue
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF with page information
        pages_text = extract_text_from_pdf(pdf_path)
        
        if not pages_text:
            logger.warning(f"No text extracted from PDF: {pdf_path} using primary method")
            
            # Try a more direct approach as fallback
            try:
                # Use simple fast extraction without OCR
                all_text = "\n\n".join([str(element) for element in partition_pdf(
                    filename=pdf_path,
                    strategy="fast",
                    include_page_breaks=True,
                )])
                
                if all_text and not all_text.isspace():
                    logger.info(f"Extracted text using fallback method for {pdf_path}")
                    pages_text = [(all_text, 1)]  # Assume single page
                else:
                    logger.warning(f"Fallback extraction also failed for {pdf_path}")
                    continue
            except Exception as e:
                logger.error(f"Fallback extraction failed for {pdf_path}: {str(e)}")
                continue
        
        file_name = os.path.basename(pdf_path)
        
        # Log what pages we're processing
        logger.info(f"Processing {len(pages_text)} pages from {file_name}")
        
        # Process each page
        for page_text, page_number in pages_text:
            if not page_text or page_text.isspace():
                continue
            
            # Log sample of each page's content for debugging
            content_sample = page_text[:200].replace('\n', ' ')
            logger.info(f"Page {page_number} content sample: '{content_sample}...'")
                
            # Split text into chunks
            chunks = split_text(page_text)
            
            # Log number of chunks per page
            logger.info(f"Split page {page_number} into {len(chunks)} chunks")
            
            # Add metadata to documents
            for i, doc in enumerate(chunks):
                doc.metadata = {
                    "source": file_name,
                    "page": page_number,
                    "chunk_id": i,
                    "type": "text"
                }
                
                # Log content sample from each chunk for better traceability
                if i < 5 or i % 10 == 0:  # Log first 5 chunks and every 10th chunk after that
                    chunk_sample = doc.page_content[:100].replace('\n', ' ')
                    logger.info(f"Chunk {i} from {file_name}, page {page_number}: '{chunk_sample}...'")
                    logger.info(f"Metadata: {doc.metadata}")
                
                # Check for specific keywords to help with debugging
                keywords = ["65 or older", "strategic bitcoin reserve", "crypto politicians"]
                for keyword in keywords:
                    if keyword.lower() in doc.page_content.lower():
                        logger.info(f"FOUND KEYWORD '{keyword}' in {file_name}, page {page_number}, chunk {i}")
                        excerpt = doc.page_content[:200].replace('\n', ' ')
                        logger.info(f"Context: '{excerpt}...'")
            
            all_documents.extend(chunks)
    
    # Log info about extraction results
    if all_documents:
        logger.info(f"Successfully extracted {len(all_documents)} text chunks from PDFs")
        # Check page distribution
        page_counts = {}
        source_page_map = {}
        for doc in all_documents:
            page = doc.metadata.get("page", 0)
            source = doc.metadata.get("source", "unknown")
            page_counts[page] = page_counts.get(page, 0) + 1
            
            # Track which sources have which pages
            if source not in source_page_map:
                source_page_map[source] = set()
            source_page_map[source].add(page)
        
        logger.info(f"Page distribution in chunks: {page_counts}")
        logger.info(f"Source to page mapping: {source_page_map}")
    else:
        logger.error("No text could be extracted from any of the provided PDFs")
    
    return all_documents

def summarize_text_chunks(documents: List[Document], model) -> List[str]:
    """
    Generate summaries for text chunks using OpenAI with optimized batch processing
    
    Args:
        documents: List of Document objects
        model: LLM model to use for summarization
        
    Returns:
        List of summaries
    """
    import time
    from concurrent.futures import ThreadPoolExecutor
    
    # Determine if batch processing is needed based on document count
    total_docs = len(documents)
    
    # Use batch processing for more than 20 documents
    if total_docs > 20:
        logger.info(f"Using batch processing for {total_docs} documents")
        summaries = [""] * total_docs
        
        # Optimized batch size and parallelism based on document count
        batch_size = 20
        max_workers = 10
        
        # Configure higher parallelism for larger documents
        if total_docs > 100:
            batch_size = 30
            max_workers = 15
        if total_docs > 500:
            batch_size = 40
            max_workers = 20
            
        logger.info(f"Batch size: {batch_size}, Max workers: {max_workers}")
        
        # Track timing for progress updates
        start_time = time.time()
        processed_count = 0
        
        # Process in batches
        for batch_start in range(0, total_docs, batch_size):
            batch_end = min(batch_start + batch_size, total_docs)
            batch_docs = documents[batch_start:batch_end]
            batch_size_actual = len(batch_docs)
            
            # Progress tracking with ETA
            processed_count += batch_size_actual
            elapsed = time.time() - start_time
            items_per_second = processed_count / elapsed if elapsed > 0 else 0
            remaining_items = total_docs - processed_count
            eta_seconds = remaining_items / items_per_second if items_per_second > 0 else 0
            
            # Format time remaining
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.1f} seconds"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds/60:.1f} minutes"
            else:
                eta_str = f"{eta_seconds/3600:.1f} hours"
                
            # Calculate progress percentage
            progress_pct = (processed_count / total_docs) * 100
                
            logger.info(f"Processing batch {batch_start//batch_size + 1}: " +
                       f"chunks {batch_start}-{batch_end-1} ({progress_pct:.1f}% complete, ETA: {eta_str})")
            
            # Process batch in parallel
            batch_results = []
            
            def summarize_single_doc(doc_idx, doc):
                if not doc.page_content or doc.page_content.isspace():
                    return doc_idx, ""
                
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
                    return doc_idx, summary
                except Exception as e:
                    logger.error(f"Error generating summary for chunk {doc_idx}: {str(e)}")
                    return doc_idx, ""
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=min(batch_size_actual, max_workers)) as executor:
                futures = []
                for j, doc in enumerate(batch_docs):
                    doc_idx = batch_start + j
                    futures.append(executor.submit(summarize_single_doc, doc_idx, doc))
                
                for future in futures:
                    try:
                        idx, summary = future.result()
                        batch_results.append((idx, summary))
                    except Exception as e:
                        logger.error(f"Error in thread: {str(e)}")
            
            # Update summaries with results
            for idx, summary in batch_results:
                summaries[idx] = summary
                
            logger.info(f"Completed batch {batch_start//batch_size + 1}, " +
                      f"processing at {items_per_second:.2f} chunks/sec")
        
        total_time = time.time() - start_time
        logger.info(f"Completed all batches in {total_time:.2f} seconds, " +
                  f"average: {total_docs/total_time:.2f} chunks/sec")
        
        return summaries
        
    # For smaller document sets, use the simpler approach
    else:
        logger.info(f"Processing {total_docs} documents sequentially")
        summaries = []
        
        for i, doc in enumerate(documents):
            try:
                # Skip empty documents
                if not doc.page_content or doc.page_content.isspace():
                    summaries.append("")
                    continue
                
                # Use a simple prompt for summarization
                prompt = f"""
                Create a concise summary that captures the key information in this text chunk.
                Focus on factual information and important details that would be relevant for retrieval.
                Text: {doc.page_content}
                Summary:
                """
                
                # Generate summary
                response = model.invoke(prompt)
                summary = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                
                # Add summary to list
                summaries.append(summary)
                
            except Exception as e:
                logger.error(f"Error generating summary for chunk {i}: {str(e)}")
                summaries.append("")
        
        return summaries 