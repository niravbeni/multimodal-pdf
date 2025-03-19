"""
Simple text processing functions for the Text PDF Chat
"""
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyPDF2"""
    try:
        pdf_reader = PdfReader(pdf_path)
        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        return text
    except Exception as e:
        st.error(f"Error extracting text from {os.path.basename(pdf_path)}: {str(e)}")
        return ""

def chunk_text(text, source, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks for processing"""
    if not text.strip():
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = text_splitter.create_documents([text], metadatas=[{"source": source}])
    return chunks

def process_pdf_text(pdf_paths):
    """Process PDF files and extract text"""
    all_chunks = []
    
    with st.status("Extracting text from PDFs...") as status:
        for i, pdf_path in enumerate(pdf_paths):
            status.update(label=f"Processing {os.path.basename(pdf_path)} ({i+1}/{len(pdf_paths)})")
            
            # Extract text from PDF
            text = extract_text_from_pdf(pdf_path)
            
            if not text:
                status.update(label=f"No text content found in {os.path.basename(pdf_path)}")
                continue
                
            # Split text into chunks
            source = os.path.basename(pdf_path)
            chunks = chunk_text(text, source)
            
            if chunks:
                all_chunks.extend(chunks)
                status.update(label=f"Extracted {len(chunks)} text chunks from {source}")
            else:
                status.update(label=f"Failed to extract text chunks from {source}")
    
    return all_chunks

def summarize_text_chunks(text_chunks, model, batch_size=5):
    """Generate summaries for text chunks"""
    summaries = []
    total_chunks = len(text_chunks)
    
    with st.status(f"Generating summaries for {total_chunks} text chunks...") as status:
        # Create a summary prompt
        prompt_text = """
        Summarize the following text in a concise and informative way.
        Focus on the key points and main ideas.
        
        Text:
        {text}
        
        Summary:
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_text)
        summarize_chain = prompt | model | StrOutputParser()
        
        # Process in batches to show progress
        for i in range(0, total_chunks, batch_size):
            batch = text_chunks[i:i + batch_size]
            status.update(label=f"Summarizing chunks {i+1}-{min(i+batch_size, total_chunks)} of {total_chunks}")
            
            for doc in batch:
                try:
                    # Get the text content from the document
                    if hasattr(doc, 'page_content'):
                        text = doc.page_content
                    else:
                        text = str(doc)
                        
                    # Generate summary
                    summary = summarize_chain.invoke({"text": text})
                    summaries.append(summary)
                except Exception as e:
                    print(f"Error summarizing chunk: {str(e)}")
                    # Add a placeholder summary
                    summaries.append(f"Summary not available for text from {doc.metadata.get('source', 'unknown source')}")
    
    return summaries 