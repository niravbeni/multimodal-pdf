"""
Text PDF Chat - Batch Preprocessing Script

This script processes PDF files in batches for better memory management.
"""
import os
import joblib
import traceback
import glob
import argparse
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load environment variables
load_dotenv()

# Constants
PRELOADED_PDF_DIRECTORY = "./preloaded_pdfs"
PREPROCESSED_DATA_PATH = "./preprocessed_data"
BATCH_DATA_PATH = os.path.join(PREPROCESSED_DATA_PATH, "batches")
PREPROCESSED_COLLECTION_FILE = os.path.join(PREPROCESSED_DATA_PATH, "primary_collection.joblib")

class BatchTextPreprocessor:
    def __init__(self, model_name="gpt-4o-mini", batch_size=5):
        # Ensure output directories exist
        os.makedirs(PREPROCESSED_DATA_PATH, exist_ok=True)
        os.makedirs(BATCH_DATA_PATH, exist_ok=True)
        
        self.batch_size = batch_size
        
        # Initialize embedding and chat models
        self.embeddings = OpenAIEmbeddings()
        self.model = ChatOpenAI(model=model_name, temperature=0.2)
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file using PyPDF2"""
        documents = []
        try:
            print(f"Processing {os.path.basename(pdf_path)} with PyPDF2...")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            doc = Document(
                                page_content=text.strip(),
                                metadata={
                                    "source": os.path.basename(pdf_path),
                                    "page": page_num + 1
                                }
                            )
                            documents.append(doc)
                    except Exception as page_error:
                        print(f"Error extracting text from page {page_num} of {pdf_path}: {page_error}")
        except Exception as e:
            print(f"Error processing {pdf_path} with PyPDF2: {e}")
        
        print(f"Extracted {len(documents)} text chunks from {os.path.basename(pdf_path)}")
        return documents
    
    def chunk_documents(self, documents):
        """Split documents into smaller chunks for better processing"""
        print(f"Chunking {len(documents)} documents...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunked_docs = []
        for doc in documents:
            chunks = text_splitter.split_documents([doc])
            chunked_docs.extend(chunks)
        
        print(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs
    
    def summarize_documents(self, documents):
        """Generate summaries for documents"""
        print(f"Generating summaries for {len(documents)} documents...")
        summaries = []
        for i, doc in enumerate(documents):
            # Print progress
            if i % 10 == 0:
                print(f"Summarizing document {i+1}/{len(documents)}...")
                
            # Create a summary prompt
            summary_prompt = f"""
            Provide a concise, informative summary of the following text.
            Focus on the key points and main ideas.
            
            Text:
            {doc.page_content[:1000]}  # Limit to first 1000 characters
            """
            
            try:
                summary = self.model.invoke(summary_prompt).content
                summaries.append(summary)
            except Exception as e:
                print(f"Error generating summary for document {i+1}: {e}")
                print(traceback.format_exc())
                # Add a placeholder summary to maintain alignment with documents
                summaries.append("Summary not available due to processing error.")
        
        print(f"Generated {len(summaries)} summaries")
        return summaries
    
    def process_batch(self, pdf_files, batch_num):
        """Process a batch of PDF files"""
        print(f"\n{'='*60}")
        print(f"Processing Batch #{batch_num} with {len(pdf_files)} PDFs")
        print(f"{'='*60}")
        
        # Process PDFs
        all_documents = []
        processed_files = []
        failed_files = []
        
        for pdf_file in pdf_files:
            print(f"\nProcessing {pdf_file} ({len(processed_files) + 1}/{len(pdf_files)})")
            pdf_path = os.path.join(PRELOADED_PDF_DIRECTORY, pdf_file)
            
            try:
                # Process individual PDF
                pdf_documents = self.extract_text_from_pdf(pdf_path)
                
                if pdf_documents:
                    all_documents.extend(pdf_documents)
                    processed_files.append(pdf_file)
                    print(f"Successfully processed {pdf_file}. Extracted {len(pdf_documents)} chunks.")
                else:
                    print(f"No documents extracted from {pdf_file}")
                    failed_files.append(pdf_file)
            except Exception as e:
                print(f"Failed to process {pdf_file}: {e}")
                print(traceback.format_exc())
                failed_files.append(pdf_file)
        
        print(f"\nDocument extraction complete. Processed {len(processed_files)}/{len(pdf_files)} files.")
        print(f"Total documents extracted: {len(all_documents)}")
        
        if not all_documents:
            print("No documents were extracted from any PDFs in this batch. Skipping.")
            return None
        
        # Chunk documents for better processing
        chunked_docs = self.chunk_documents(all_documents)
        
        # Generate summaries
        summaries = self.summarize_documents(chunked_docs)
        
        # Prepare data for saving
        batch_data = {
            "documents": chunked_docs,
            "summaries": summaries,
            "processed_files": processed_files,
            "failed_files": failed_files
        }
        
        # Save batch data
        batch_file = os.path.join(BATCH_DATA_PATH, f"batch_{batch_num}.joblib")
        print(f"Saving batch {batch_num} to {batch_file}")
        joblib.dump(batch_data, batch_file)
        
        return batch_file
    
    def process_in_batches(self, start_batch=1):
        """Process all PDFs in batches"""
        # Ensure the directories exist
        os.makedirs(PRELOADED_PDF_DIRECTORY, exist_ok=True)
        
        # Get all PDF files
        pdf_files = [
            f for f in os.listdir(PRELOADED_PDF_DIRECTORY) 
            if f.lower().endswith('.pdf')
        ]
        
        print(f"Found {len(pdf_files)} PDF files in {PRELOADED_PDF_DIRECTORY}")
        
        if not pdf_files:
            print(f"No PDF files found in {PRELOADED_PDF_DIRECTORY}! Please add some PDF files.")
            return
        
        # Process PDFs in batches
        batch_files = []
        for i in range(0, len(pdf_files), self.batch_size):
            batch_num = (i // self.batch_size) + start_batch
            batch = pdf_files[i:i + self.batch_size]
            
            batch_file = self.process_batch(batch, batch_num)
            if batch_file:
                batch_files.append(batch_file)
        
        if batch_files:
            # Merge all batches
            self.merge_batches(batch_files)
        else:
            print("No batches were processed successfully.")
    
    def merge_batches(self, batch_files=None):
        """Merge all batch files into a single collection"""
        if batch_files is None:
            # Get all batch files
            batch_files = glob.glob(os.path.join(BATCH_DATA_PATH, "batch_*.joblib"))
        
        print(f"\n{'='*60}")
        print(f"Merging {len(batch_files)} batches into primary collection")
        print(f"{'='*60}")
        
        if not batch_files:
            print("No batch files found to merge.")
            return
        
        # Initialize merged data
        all_documents = []
        all_summaries = []
        all_processed_files = []
        all_failed_files = []
        
        # Load and merge batch data
        for batch_file in batch_files:
            print(f"Loading {os.path.basename(batch_file)}...")
            try:
                batch_data = joblib.load(batch_file)
                
                # Add batch data to merged data
                all_documents.extend(batch_data["documents"])
                all_summaries.extend(batch_data["summaries"])
                all_processed_files.extend(batch_data.get("processed_files", []))
                all_failed_files.extend(batch_data.get("failed_files", []))
                
            except Exception as e:
                print(f"Error loading {batch_file}: {e}")
                print(traceback.format_exc())
        
        # Verify document and summary counts match
        if len(all_documents) != len(all_summaries):
            print("WARNING: Document and summary counts don't match!")
            print(f"Documents: {len(all_documents)}, Summaries: {len(all_summaries)}")
            
            # Trim to match
            min_count = min(len(all_documents), len(all_summaries))
            all_documents = all_documents[:min_count]
            all_summaries = all_summaries[:min_count]
            
            print(f"Trimmed to {min_count} items for alignment")
        
        # Save merged data
        merged_data = {
            "documents": all_documents,
            "summaries": all_summaries,
            "processed_files": all_processed_files,
            "failed_files": all_failed_files
        }
        
        print(f"Saving merged collection with {len(all_documents)} documents to {PREPROCESSED_COLLECTION_FILE}")
        joblib.dump(merged_data, PREPROCESSED_COLLECTION_FILE)
        
        print("Merge complete!")
        print(f"Successfully processed {len(all_processed_files)} files")
        print(f"Failed to process {len(all_failed_files)} files")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch process PDF documents for text-based RAG')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of PDFs to process in each batch')
    parser.add_argument('--model', default='gpt-4o-mini', help='OpenAI model to use for summarization')
    parser.add_argument('--start-batch', type=int, default=1, help='Starting batch number')
    parser.add_argument('--merge-only', action='store_true', help='Only merge existing batches')
    args = parser.parse_args()
    
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key in the .env file or as an environment variable.")
        return
    
    # Create preprocessor
    preprocessor = BatchTextPreprocessor(
        model_name=args.model,
        batch_size=args.batch_size
    )
    
    if args.merge_only:
        # Just merge existing batches
        preprocessor.merge_batches()
    else:
        # Process all PDFs in batches
        preprocessor.process_in_batches(start_batch=args.start_batch)

if __name__ == "__main__":
    main()