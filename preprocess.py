"""
Text PDF Chat - Simple Preprocessing Script
"""
import os
import joblib
import traceback
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
PREPROCESSED_COLLECTION_FILE = os.path.join(PREPROCESSED_DATA_PATH, "primary_collection.joblib")

class TextPreprocessor:
    def __init__(self, model_name="gpt-4o-mini"):
        # Ensure output directory exists
        os.makedirs(PREPROCESSED_DATA_PATH, exist_ok=True)
        
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
        """
        Split documents into smaller chunks for better processing
        """
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
        """
        Generate summaries for documents
        """
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
    
    def preprocess_collection(self):
        """
        Preprocess all PDFs in the primary collection
        """
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
            print("No documents were extracted from any PDFs. Cannot continue.")
            return
        
        # Chunk documents for better processing
        chunked_docs = self.chunk_documents(all_documents)
        
        # Generate summaries
        summaries = self.summarize_documents(chunked_docs)
        
        # Prepare data for saving
        preprocessed_data = {
            "documents": chunked_docs,
            "summaries": summaries,
            "processed_files": processed_files,
            "failed_files": failed_files
        }
        
        # Save preprocessed data
        output_path = PREPROCESSED_COLLECTION_FILE
        print(f"\nSaving preprocessed data to {output_path}")
        joblib.dump(preprocessed_data, output_path)
        print("Preprocessing complete!")
        
        return preprocessed_data

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocess PDF documents for text-based RAG')
    parser.add_argument('--model', default='gpt-4o-mini', help='OpenAI model to use for summarization')
    args = parser.parse_args()
    
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key in the .env file or as an environment variable.")
        return
    
    # Create preprocessor and run
    preprocessor = TextPreprocessor(model_name=args.model)
    preprocessor.preprocess_collection()

if __name__ == "__main__":
    main()