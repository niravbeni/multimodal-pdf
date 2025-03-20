"""
Text processor for PDF files
Contains functions for extracting and processing text from PDF files
"""
import os
from typing import List, Dict, Optional, Tuple
import logging
import warnings

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
    Extract text from a PDF file using Unstructured
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of tuples containing (text, page_number)
    """
    try:
        # Use simpler extraction method that was working before
        elements = partition_pdf(
            filename=pdf_path,
            strategy="fast",
            extract_images_in_pdf=False,
            infer_table_structure=False,
            # Don't use page breaks which might be causing issues
            include_page_breaks=False,
        )
        
        # Group elements by page number
        page_elements = {}
        for element in elements:
            if hasattr(element, 'metadata') and 'page_number' in element.metadata:
                page_num = element.metadata['page_number']
                if page_num not in page_elements:
                    page_elements[page_num] = []
                page_elements[page_num].append(str(element))
        
        # Convert to list of (text, page_number) tuples
        pages_text = []
        for page_num in sorted(page_elements.keys()):
            text = "\n\n".join(page_elements[page_num]).strip()
            if text:  # Only add if there's text content
                pages_text.append((text, page_num))
        
        # If we didn't get any pages with the metadata approach, fall back to simpler method
        if not pages_text:
            logger.warning(f"No pages with metadata found in {pdf_path}, using fallback method")
            all_text = "\n\n".join([str(element) for element in elements]).strip()
            if all_text:
                pages_text = [(all_text, 1)]  # Assume it's all page 1
        
        return pages_text
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return []

def split_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 150) -> List[Document]:
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
                # Simple extraction of all text
                all_text = "\n\n".join([str(element) for element in partition_pdf(
                    filename=pdf_path,
                    strategy="fast",
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
        
        # Process each page
        for page_text, page_number in pages_text:
            if not page_text or page_text.isspace():
                continue
                
            # Split text into chunks
            chunks = split_text(page_text)
            
            # Add metadata to documents
            for i, doc in enumerate(chunks):
                doc.metadata = {
                    "source": file_name,
                    "page": page_number,
                    "chunk_id": i,
                    "type": "text"
                }
            
            all_documents.extend(chunks)
    
    # Log info about extraction results
    if all_documents:
        logger.info(f"Successfully extracted {len(all_documents)} text chunks from PDFs")
    else:
        logger.error("No text could be extracted from any of the provided PDFs")
    
    return all_documents

def summarize_text_chunks(documents: List[Document], model) -> List[str]:
    """
    Generate summaries for text chunks using OpenAI
    
    Args:
        documents: List of Document objects
        model: LLM model to use for summarization
        
    Returns:
        List of summaries
    """
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