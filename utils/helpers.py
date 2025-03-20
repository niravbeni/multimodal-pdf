"""
Utility helper functions for Text PDF Chat
"""
import os
import uuid
import traceback
import streamlit as st
import joblib
from langchain.schema.document import Document
from langchain_core.messages import HumanMessage, AIMessage

def save_uploaded_file(uploaded_file, temp_dir="./temp_pdf_files"):
    """Save uploaded file to a temporary location and return the path"""
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def load_preprocessed_data(file_path):
    """
    Load preprocessed data from joblib file
    
    Returns:
    - documents: list of Document objects
    - metadata: dictionary with additional info
    """
    try:
        # Get file size for logging
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        
        # Load the data
        data = joblib.load(file_path)
        
        documents = []
        summaries = []
        metadata = {"file_size_mb": file_size}
        
        # Handle dictionary format
        if isinstance(data, dict):
            # Extract documents
            if "documents" in data:
                documents = data["documents"]
            
            # Extract summaries
            if "summaries" in data:
                summaries = data["summaries"]
            
            # Extract other metadata
            for key in data:
                if key not in ["documents", "summaries"]:
                    metadata[key] = data[key]
        
        # Add summaries to metadata
        metadata["summaries"] = summaries
        metadata["doc_count"] = len(documents)
        metadata["sum_count"] = len(summaries)
        
        return documents, metadata
        
    except Exception as e:
        error_msg = f"Error loading file: {str(e)}"
        st.error(error_msg)
        return [], {"error": error_msg, "traceback": traceback.format_exc()}

def ensure_chroma_directory():
    """Ensure the ChromaDB directory exists"""
    chroma_dir = "./chroma_db"
    os.makedirs(chroma_dir, exist_ok=True)
    return chroma_dir