"""
Text PDF Chat - Preprocessing Script

This script processes PDF files to extract text, create document chunks,
and store the processed data for later use in the chat application.
"""
import os
import argparse
import glob
import logging
import joblib
import uuid
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema.document import Document

# Import local modules
from utils.text_processor import process_pdf_text, summarize_text_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_collection_from_pdfs(pdf_paths, output_file, model_name="gpt-4o-mini"):
    """
    Process PDFs and create a collection for later use
    
    Args:
        pdf_paths: List of paths to PDF files
        output_file: Path to save the processed data
        model_name: OpenAI model to use for summarization
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if not pdf_paths:
            logger.error("No PDF files provided")
            return False
        
        # Create OpenAI model
        logger.info(f"Using OpenAI model: {model_name}")
        model = ChatOpenAI(model=model_name, temperature=0.2)
        
        # Process PDFs to extract text chunks
        logger.info(f"Processing {len(pdf_paths)} PDF files")
        text_chunks = process_pdf_text(pdf_paths)
        
        if not text_chunks:
            logger.error("No text content extracted from PDFs")
            return False
        
        logger.info(f"Extracted {len(text_chunks)} text chunks")
        
        # Generate summaries for text chunks
        logger.info("Generating summaries for text chunks")
        summaries = summarize_text_chunks(text_chunks, model)
        
        # Create metadata
        metadata = {
            "summaries": summaries,
            "sources": [os.path.basename(path) for path in pdf_paths],
            "processed_by": "text_preprocess_script",
            "collection_id": str(uuid.uuid4()),
        }
        
        # Save processed data
        logger.info(f"Saving processed data to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        joblib.dump((text_chunks, metadata), output_file)
        
        logger.info("Preprocessing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error preprocessing PDFs: {str(e)}", exc_info=True)
        return False

def main():
    """Main function to run the preprocessing script"""
    parser = argparse.ArgumentParser(description="Preprocess PDF files for Text PDF Chat")
    parser.add_argument("--input", "-i", type=str, help="Input directory containing PDF files or specific PDF file")
    parser.add_argument("--output", "-o", type=str, default="./preprocessed_data/primary_collection.joblib", 
                         help="Output joblib file path")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o-mini", 
                         help="OpenAI model to use for summarization")
    
    args = parser.parse_args()
    
    if not args.input:
        logger.error("Input directory or file is required")
        return 1
    
    # Check if input is a directory or file
    if os.path.isdir(args.input):
        pdf_paths = glob.glob(os.path.join(args.input, "*.pdf"))
        if not pdf_paths:
            logger.error(f"No PDF files found in directory: {args.input}")
            return 1
    elif os.path.isfile(args.input) and args.input.lower().endswith(".pdf"):
        pdf_paths = [args.input]
    else:
        logger.error(f"Invalid input: {args.input}")
        return 1
    
    # Process PDFs
    logger.info(f"Found {len(pdf_paths)} PDF files")
    success = create_collection_from_pdfs(pdf_paths, args.output, args.model)
    
    if success:
        logger.info(f"Preprocessing completed. Output saved to {args.output}")
        return 0
    else:
        logger.error("Preprocessing failed")
        return 1

if __name__ == "__main__":
    exit(main())