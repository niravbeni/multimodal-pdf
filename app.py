"""
Text PDF Chat - Simple Text-based RAG System
"""
from utils.sqlite_fix import fix_sqlite
fix_sqlite()

import sys
import os
import warnings
import traceback
import logging
import uuid
import joblib
import pandas as pd
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

# Add the project directory to the Python path to fix import issues
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from database.pdf_manager import get_pdf_metadata_from_db
from database.config import SessionLocal
from utils.helpers import save_uploaded_file, load_preprocessed_data, ensure_chroma_directory
from utils.html_templates import inject_css
from utils.text_processor import process_pdf_text, summarize_text_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Get OpenAI API key from Streamlit secrets or environment variable
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set it in .streamlit/secrets.toml or .env file.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # Set for OpenAI client

# Constants for preloaded collection
PREPROCESSED_DATA_PATH = "./preprocessed_data"
PREPROCESSED_COLLECTION_FILE = os.path.join(PREPROCESSED_DATA_PATH, "primary_collection.joblib")

# Check if preprocessed_data directory exists
if not os.path.exists(PREPROCESSED_DATA_PATH):
    os.makedirs(PREPROCESSED_DATA_PATH, exist_ok=True)
    print(f"Created preprocessed data directory at {PREPROCESSED_DATA_PATH}")

# Cache expensive operations
@st.cache_resource
def get_openai_model(model_name):
    """Cache the OpenAI model to avoid reloading it"""
    return ChatOpenAI(model=model_name, temperature=0.2)

@st.cache_resource
def get_embeddings():
    """Cache the embeddings model to avoid reloading it"""
    return OpenAIEmbeddings()

# Define custom avatar paths
USER_AVATAR = "images/user.png"
ASSISTANT_AVATAR = "images/ideo.png"

def build_rag_prompt(context, question, chat_history=None):
    """Build a prompt for the RAG system"""
    context_text = ""
    
    # Process all context documents with source information
    for i, doc in enumerate(context):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        filename = os.path.basename(source) if source != "Unknown" else source
        
        # Log each context entry for debugging
        logger.info(f"Context entry {i}: source={source}, filename={filename}, page={page}")
        
        # Format with exact filename
        context_text += f"[{filename}, Page {page}]\n{doc.page_content}\n\n"

    # Format chat history if provided
    chat_history_text = ""
    if chat_history:
        for message in chat_history:
            role = "Human" if message["role"] == "user" else "Assistant"
            chat_history_text += f"{role}: {message['content']}\n"

    # Construct prompt with exact filename citation instructions
    prompt = f"""
    Answer the question based only on the following context:
    
    Context: {context_text}
    
    {chat_history_text}
    
    Question: {question}
    
    Provide a clear, detailed answer that directly addresses the question.
    
    When citing sources, use the exact filename shown in the context above, including the .pdf extension.
    For example, if the context shows [example.pdf, Page 5], cite it exactly as [example.pdf, Page 5].
    
    If multiple sources support a statement, cite them all like: [document1.pdf, Page 5; document2.pdf, Page 3]
    
    Make sure you:
    1. Use the exact filenames as shown in the context
    2. Include the .pdf extension in citations
    3. Use the exact page numbers shown in the context
    4. Do not modify or abbreviate the filenames
    
    If you can't answer based on the provided context, simply state that you don't have enough information.
    """

    return prompt

def get_rag_chain(retriever, model):
    """Create the RAG chain for conversation"""
    def generate_response(query):
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(query)
        
        # Debug: Log the retrieved documents with their metadata
        logger.info(f"Retrieved {len(docs)} documents for query: {query}")
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', 'Unknown')
            
            # Log detailed information about each retrieved document
            logger.info(f"Retrieved Doc {i}: source={source}, page={page}, chunk_id={chunk_id}")
            
            # Log a sample of the content to see what text is being used
            content_sample = doc.page_content[:150].replace('\n', ' ')
            logger.info(f"Content sample: '{content_sample}...'")
            
            # Check for specific keywords to track where certain content is found
            keywords = ["65 or older", "strategic bitcoin reserve", "crypto politicians"]
            for keyword in keywords:
                if keyword.lower() in doc.page_content.lower():
                    logger.info(f"FOUND KEYWORD '{keyword}' in retrieved doc from {source}, page {page}")
        
        # Get chat history from session state
        chat_history = st.session_state.conversation if "conversation" in st.session_state else []
        
        # Build the prompt
        prompt = build_rag_prompt(docs, query, chat_history)
        
        # Log the prompt for debugging
        logger.info(f"Prompt sent to model: {prompt[:500]}...")
        
        # Generate the response
        response = model.invoke(prompt)
        
        # Log the response
        logger.info(f"Model response: {response.content}")
        
        # Store the retrieved documents in session state for citation tracking
        st.session_state.last_retrieved_docs = docs
        
        return response
    
    return generate_response

def handle_user_input(user_question, rag_chain):
    """Process the user question and update the chat"""
    if not user_question:
        return
    
    # Add user message to conversation
    st.session_state.conversation.append({"role": "user", "content": user_question, "formatted_content": user_question})
    
    # Display user message immediately
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_question)
    
    # Display a typing indicator for the assistant's response
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ Thinking...")
        
        try:
            # Generate response
            response = rag_chain(user_question)
            
            # Get response content
            response_content = response.content
            
            # Format citations for better readability
            import re
            
            # Log the raw response to see what citations are present
            logger.info(f"Raw response with citations: {response_content}")
            
            # Get the last retrieved documents to ensure correct filenames
            last_docs = st.session_state.get('last_retrieved_docs', [])
            # Create a mapping of both full paths and basenames to the correct filename
            filename_map = {}
            for doc in last_docs:
                source = doc.metadata.get('source', '')
                if source:
                    filename = os.path.basename(source)
                    filename_map[source] = filename  # Map full path to filename
                    filename_map[filename] = filename  # Map filename to itself
                    # Also map without .pdf extension
                    name_without_ext = os.path.splitext(filename)[0]
                    filename_map[name_without_ext] = filename
            
            logger.info(f"Filename map: {filename_map}")
            
            # Apply formatting to each citation pattern in order of specificity:
            
            # 1. Single page citation: [filename.pdf, Page X]
            single_page_pattern = r'\[([^,;\]]+?)(?:\.pdf)?,\s*Page\s*(\d+)\]'
            def single_page_replacer(match):
                pdf_path = match.group(1)
                page_num = match.group(2)
                pdf_filename = filename_map.get(pdf_path, pdf_path)
                if not pdf_filename.lower().endswith('.pdf'):
                    pdf_filename += '.pdf'
                return f'<span class="citation">[{pdf_filename}, Page {page_num}]</span>'
            
            formatted_response = re.sub(single_page_pattern, single_page_replacer, response_content)
            
            # 2. Multiple pages with 'and': [filename.pdf, Pages X and Y]
            pages_and_pattern = r'\[([^,;\]]+?)(?:\.pdf)?,\s*Pages\s*(\d+)\s*and\s*(\d+)\]'
            def pages_and_replacer(match):
                pdf_path = match.group(1)
                page1 = match.group(2)
                page2 = match.group(3)
                pdf_filename = filename_map.get(pdf_path, pdf_path)
                if not pdf_filename.lower().endswith('.pdf'):
                    pdf_filename += '.pdf'
                return f'<span class="citation">[{pdf_filename}, Pages {page1} and {page2}]</span>'
            
            formatted_response = re.sub(pages_and_pattern, pages_and_replacer, formatted_response)
            
            # 3. Multiple pages with semicolon: [filename.pdf, Page X; Page Y]
            pages_semi_pattern = r'\[([^,;\]]+?)(?:\.pdf)?,\s*Page\s*(\d+);\s*Page\s*(\d+)\]'
            def pages_semi_replacer(match):
                pdf_path = match.group(1)
                page1 = match.group(2)
                page2 = match.group(3)
                pdf_filename = filename_map.get(pdf_path, pdf_path)
                if not pdf_filename.lower().endswith('.pdf'):
                    pdf_filename += '.pdf'
                return f'<span class="citation">[{pdf_filename}, Page {page1}; Page {page2}]</span>'
            
            formatted_response = re.sub(pages_semi_pattern, pages_semi_replacer, formatted_response)
            
            # 4. Multiple pages with comma: [filename.pdf, Pages X, Y, Z]
            pages_comma_pattern = r'\[([^,;\]]+?)(?:\.pdf)?,\s*Pages\s*(\d+(?:\s*,\s*\d+)+)\]'
            def pages_comma_replacer(match):
                pdf_path = match.group(1)
                pages = match.group(2).replace(' ', '')
                pdf_filename = filename_map.get(pdf_path, pdf_path)
                if not pdf_filename.lower().endswith('.pdf'):
                    pdf_filename += '.pdf'
                return f'<span class="citation">[{pdf_filename}, Pages {pages}]</span>'
            
            formatted_response = re.sub(pages_comma_pattern, pages_comma_replacer, formatted_response)
            
            # 5. Multiple pages with range: [filename.pdf, Pages X-Y]
            pages_range_pattern = r'\[([^,;\]]+?)(?:\.pdf)?,\s*Pages\s*(\d+)-(\d+)\]'
            def pages_range_replacer(match):
                pdf_path = match.group(1)
                start_page = match.group(2)
                end_page = match.group(3)
                pdf_filename = filename_map.get(pdf_path, pdf_path)
                if not pdf_filename.lower().endswith('.pdf'):
                    pdf_filename += '.pdf'
                return f'<span class="citation">[{pdf_filename}, Pages {start_page}-{end_page}]</span>'
            
            formatted_response = re.sub(pages_range_pattern, pages_range_replacer, formatted_response)
            
            # 6. Multiple files with single pages: [file1.pdf, Page X; file2.pdf, Page Y]
            multi_file_pattern = r'\[([^,;\]]+?)(?:\.pdf)?,\s*Page\s*(\d+);\s*([^,;\]]+?)(?:\.pdf)?,\s*Page\s*(\d+)\]'
            def multi_file_replacer(match):
                pdf1_path = match.group(1)
                page1 = match.group(2)
                pdf2_path = match.group(3)
                page2 = match.group(4)
                pdf1_filename = filename_map.get(pdf1_path, pdf1_path)
                pdf2_filename = filename_map.get(pdf2_path, pdf2_path)
                if not pdf1_filename.lower().endswith('.pdf'):
                    pdf1_filename += '.pdf'
                if not pdf2_filename.lower().endswith('.pdf'):
                    pdf2_filename += '.pdf'
                return f'<span class="citation">[{pdf1_filename}, Page {page1}; {pdf2_filename}, Page {page2}]</span>'
            
            formatted_response = re.sub(multi_file_pattern, multi_file_replacer, formatted_response)
            
            # Add the formatted response to the conversation (with HTML for persistence)
            st.session_state.conversation.append({
                "role": "assistant", 
                "content": response_content,  # Plain content
                "formatted_content": formatted_response  # HTML-formatted content with styled citations
            })
            
            # Display the formatted response with HTML
            message_placeholder.markdown(formatted_response, unsafe_allow_html=True)
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            message_placeholder.markdown(f"❌ {error_msg}")
            st.error(error_msg)

def load_preloaded_collection():
    """
    Load the preprocessed collection from the joblib file and create a retriever
    """
    # Check if the joblib file exists
    if not os.path.exists(PREPROCESSED_COLLECTION_FILE):
        st.error(f"Preprocessed collection file not found at {PREPROCESSED_COLLECTION_FILE}")
        return None
    
    try:
        with st.status("Loading preprocessed collection...") as status:
            # Load data from file
            status.update(label="Loading data from file...")
            documents, metadata = load_preprocessed_data(PREPROCESSED_COLLECTION_FILE)
            
            # Check if we have documents but no summaries
            summaries = metadata.get("summaries", [])
            
            if documents and (not summaries or len(summaries) == 0):
                status.update(label="No summaries found. Generating summaries...")
                model = get_openai_model("gpt-4o-mini")
                summaries = summarize_text_chunks(documents, model)
            
            # Handle mismatch between documents and summaries
            elif documents and summaries and len(documents) != len(summaries):
                if len(documents) > len(summaries):
                    # Generate missing summaries
                    status.update(label=f"Generating {len(documents) - len(summaries)} missing summaries...")
                    model = get_openai_model("gpt-4o-mini")
                    missing_summaries = summarize_text_chunks(documents[len(summaries):], model)
                    summaries.extend(missing_summaries)
                else:
                    # Trim excess summaries
                    status.update(label="Trimming excess summaries...")
                    summaries = summaries[:len(documents)]
            
            # Create retriever
            status.update(label="Creating retriever...")
            retriever = create_text_retriever(documents, summaries, get_embeddings())
            
            status.update(label="Collection loaded successfully!", state="complete")
            return retriever
    except Exception as e:
        error_msg = f"Error loading preloaded collection: {str(e)}\n{traceback.format_exc()}"
        st.error(error_msg)
        return None

def create_text_retriever(text_chunks, summaries, embeddings):
    """Create a MultiVectorRetriever for text-only content"""
    # Create a unique ChromaDB collection
    chroma_dir = ensure_chroma_directory()
    collection_name = f"text_rag_{uuid.uuid4().hex[:8]}"
    
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=chroma_dir
    )
    
    store = InMemoryStore()
    id_key = "doc_id"
    
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    
    # Add text documents with summaries
    doc_ids = [str(uuid.uuid4()) for _ in range(len(summaries))]
    
    # Log page number debug info
    page_numbers = {}
    for i, chunk in enumerate(text_chunks[:5]):  # Log first few for debugging
        if hasattr(chunk, 'metadata') and chunk.metadata:
            page = chunk.metadata.get('page', 'Unknown')
            source = chunk.metadata.get('source', 'Unknown')
            page_numbers[f"{source}-{i}"] = page
            logger.info(f"Original chunk {i} - Source: {source}, Page: {page}")
    
    # Ensure we preserve all metadata from original text chunks in both places
    summary_docs = []
    for i, summary in enumerate(summaries):
        if i < len(text_chunks):
            # Copy metadata from text chunk to summary doc
            metadata = {}
            if hasattr(text_chunks[i], 'metadata') and text_chunks[i].metadata:
                metadata = text_chunks[i].metadata.copy()
                
                # Debug log for first few docs
                if i < 5:
                    page = metadata.get('page', 'Unknown')
                    source = metadata.get('source', 'Unknown')
                    logger.info(f"Preserving metadata for summary {i} - Source: {source}, Page: {page}")
            
            # Add the id_key for the retriever
            metadata[id_key] = doc_ids[i]
            summary_doc = Document(page_content=summary, metadata=metadata)
            summary_docs.append(summary_doc)
        else:
            # Fallback if there's a mismatch (shouldn't happen)
            summary_docs.append(Document(page_content=summary, metadata={id_key: doc_ids[i]}))
    
    # Add docs to vectorstore
    retriever.vectorstore.add_documents(summary_docs)
    
    # Store original docs with their full metadata in the docstore
    original_docs_with_ids = list(zip(doc_ids, text_chunks[:len(summaries)]))
    retriever.docstore.mset(original_docs_with_ids)
    
    # Verify a few documents were stored with correct metadata
    for i, doc_id in enumerate(doc_ids[:5]):
        if i < len(text_chunks):
            # Use mget instead of get - InMemoryStore uses mget
            stored_docs = retriever.docstore.mget([doc_id])
            if stored_docs and doc_id in stored_docs:
                stored_doc = stored_docs[doc_id]
                if hasattr(stored_doc, 'metadata') and stored_doc.metadata:
                    page = stored_doc.metadata.get('page', 'Unknown')
                    source = stored_doc.metadata.get('source', 'Unknown')
                    logger.info(f"Verified stored doc {i} - ID: {doc_id}, Source: {source}, Page: {page}")
    
    # Log page number mapping for reference
    logger.info(f"Page numbers for first chunks: {page_numbers}")
    
    return retriever

def initialize_session_state():
    """Initialize session state variables"""
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    else:
        # Add formatted_content field to existing messages if needed
        for message in st.session_state.conversation:
            if "formatted_content" not in message:
                message["formatted_content"] = message["content"]
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    if "mode" not in st.session_state:
        st.session_state.mode = "upload"  # Default to upload mode
    
    if "temp_pdf_files" not in st.session_state:
        st.session_state.temp_pdf_files = []

def reset_conversation():
    """Reset the conversation"""
    st.session_state.conversation = []

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files and create a retriever"""
    temp_dir = "./temp_pdf_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    pdf_paths = []
    for uploaded_file in uploaded_files:
        temp_path = save_uploaded_file(uploaded_file, temp_dir)
        pdf_paths.append(temp_path)
    
    if not pdf_paths:
        st.error("No valid PDF files uploaded!")
        return None
    
    # Process the PDFs to extract text
    with st.status("Processing PDFs...") as status:
        try:
            # Extract text from PDFs
            status.update(label="Extracting text from PDFs...")
            text_chunks = process_pdf_text(pdf_paths)
            
            if not text_chunks:
                status.update(label="❌ No text content extracted from PDFs!", state="error")
                # Add more detailed information about the PDFs
                pdf_info = "\n".join([f"- {os.path.basename(path)} (Size: {os.path.getsize(path) / 1024:.1f} KB)" for path in pdf_paths])
                error_msg = f"""
                Could not extract any text from the provided PDFs. Please try different files.
                
                PDF Files attempted:
                {pdf_info}
                
                This could be due to:
                1. The PDFs containing only scanned images without OCR
                2. The PDFs having security restrictions
                3. Text encoded in a non-standard way
                """
                st.error(error_msg)
                return None
            
            # Generate summaries for better retrieval
            status.update(label=f"Generating summaries for {len(text_chunks)} text chunks...")
            model = get_openai_model("gpt-4o-mini")
            summaries = summarize_text_chunks(text_chunks, model)
            
            # Create retriever
            status.update(label="Creating retriever...")
            embeddings = get_embeddings()
            retriever = create_text_retriever(text_chunks, summaries, embeddings)
            
            # Save the temporary PDF paths for cleanup
            st.session_state.temp_pdf_files = pdf_paths
            
            status.update(label="✅ PDFs processed successfully!", state="complete")
            return retriever
            
        except Exception as e:
            error_msg = f"Error processing PDFs: {str(e)}\n{traceback.format_exc()}"
            status.update(label=f"❌ {error_msg}", state="error")
            st.error(error_msg)
            return None

def display_conversation():
    """Display the conversation history using Streamlit's chat interface"""
    # Get messages
    messages = st.session_state.conversation
    if not messages:  # If no messages, return early
        return

    # Create a container for the messages
    with st.container():
        # Show full history
        for message in messages:
            avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
            with st.chat_message(message["role"], avatar=avatar):
                # Use formatted content if available, otherwise use plain content
                content = message.get("formatted_content", message["content"])
                st.markdown(content, unsafe_allow_html=True)

def get_pdf_metadata():
    """Get metadata for PDFs from the database"""
    return get_pdf_metadata_from_db()

def main():
    # Initialize session state
    initialize_session_state()
    
    # Page configuration
    st.set_page_config(page_title="Text PDF Chat", page_icon="📄", layout="wide")
    
    # Inject custom CSS
    st.markdown(inject_css(), unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # Mode selection - Simplified state handling
        mode = st.radio(
            "Choose mode:",
            ["Upload PDFs", "Search PDF Collection"],
            key="mode_radio",
            on_change=reset_conversation
        )
        
        # Remove the redundant session state update
        is_upload_mode = mode == "Upload PDFs"
        
        if is_upload_mode:
            # File uploader widget
            uploaded_files = st.file_uploader(
                "Upload your PDFs",
                accept_multiple_files=True,
                type="pdf",
                help="Upload one or more PDF files to chat with",
                on_change=reset_conversation
            )
            
            process_button = st.button("Process PDFs", type="primary")
            
            if process_button and uploaded_files:
                # Process the uploaded files
                retriever = process_uploaded_files(uploaded_files)
                
                if retriever:
                    # Create the RAG chain
                    model = get_openai_model("gpt-4o-mini")
                    st.session_state.rag_chain = get_rag_chain(retriever, model)
            
        else:  # Search PDF Collection mode
            st.markdown("### Search PDF Collection")
            st.info("Browse and search through our curated collection of PDFs.")
            
            # Get PDF metadata from database
            pdf_df = get_pdf_metadata()
            
            if not pdf_df.empty:
                # Create a container for the dataframe with fixed height
                with st.container():
                    st.markdown("#### Available PDFs")
                    st.markdown("Select one or more PDFs to chat with:")
                    
                    # Display the dataframe with selection column
                    edited_df = st.data_editor(
                        pdf_df,
                        hide_index=True,
                        column_config={
                            "Selected": "checkbox",
                            "Organization": {
                                "label": "Organization",
                                "help": "Organization that published the PDF",
                            },
                            "Filename": {
                                "label": "Filename",
                                "help": "Name of the PDF file",
                            },
                            "Size (MB)": {
                                "label": "Size (MB)",
                                "help": "File size in megabytes",
                                "format": "%.2f MB",
                            },
                            "Last Modified": {
                                "label": "Last Modified",
                                "help": "Last modification date",
                                "format": "D MMM YYYY",
                            },
                            "Path": {
                                "label": "Path",
                                "help": "File path",
                                "hidden": True,
                            },
                            "ID": {
                                "label": "ID",
                                "help": "Database ID",
                                "hidden": True,
                            }
                        },
                        disabled=["Organization", "Filename", "Size (MB)", "Last Modified"],
                        key="pdf_selection",
                        height=400,
                    )
                    
                    # Process selected PDFs button
                    selected_pdfs = edited_df[edited_df['Selected']]['Path'].tolist()
                    if st.button("Process Selected PDFs", type="primary", disabled=len(selected_pdfs) == 0):
                        # Process the selected PDFs
                        with st.status("Processing PDFs...") as status:
                            try:
                                # Get database session
                                db = SessionLocal()
                                
                                # Get selected PDF IDs
                                selected_ids = edited_df[edited_df['Selected']]['ID'].tolist()
                                
                                # Extract text from PDFs
                                status.update(label="Extracting text from PDFs...")
                                documents = process_pdf_text(selected_pdfs)
                                
                                if documents:
                                    # Generate summaries for better retrieval
                                    status.update(label=f"Generating summaries for {len(documents)} text chunks...")
                                    model = get_openai_model("gpt-4o-mini")
                                    summaries = summarize_text_chunks(documents, model)
                                    
                                    # Create retriever
                                    status.update(label="Creating retriever...")
                                    embeddings = get_embeddings()
                                    retriever = create_text_retriever(documents, summaries, embeddings)
                                    
                                    if retriever:
                                        # Create the RAG chain
                                        st.session_state.rag_chain = get_rag_chain(retriever, model)
                                        status.update(label="✅ PDFs processed successfully!", state="complete")
                                        
                                        # Clear the search question since we're moving to chat mode
                                        st.session_state.search_question = None
                                        
                                        # Show success message
                                        st.success(f"Successfully processed {len(selected_pdfs)} PDFs. You can now start asking questions!")
                                    else:
                                        status.update(label="❌ Failed to create retriever!", state="error")
                                else:
                                    status.update(label="❌ No text content extracted from PDFs!", state="error")
                                    # Add more detailed information about the PDFs
                                    pdf_info = "\n".join([f"- {os.path.basename(path)} (Size: {os.path.getsize(path) / 1024:.1f} KB)" for path in selected_pdfs])
                                    error_msg = f"""
                                    Could not extract any text from the selected PDFs. Please try different files.
                                    
                                    PDF Files attempted:
                                    {pdf_info}
                                    
                                    This could be due to:
                                    1. The PDFs containing only scanned images without OCR
                                    2. The PDFs having security restrictions
                                    3. Text encoded in a non-standard way
                                    """
                                    st.error(error_msg)
                            except Exception as e:
                                error_msg = f"Error processing PDFs: {str(e)}\n{traceback.format_exc()}"
                                status.update(label=f"❌ {error_msg}", state="error")
                                st.error(error_msg)
                            finally:
                                db.close()
            else:
                st.warning("No PDFs found in the database. Please upload some PDFs first.")
        
        # Clear conversation button
        if st.button("Clear Conversation"):
            reset_conversation()

    # Main content area
    st.header("Chat with your PDFs")
    st.markdown("---")
    
    # Show the conversation
    display_conversation()
    
    # Chat input
    if is_upload_mode:
        if st.session_state.rag_chain:
            user_question = st.chat_input("Ask a question about your PDFs", key="upload_mode_chat")
            if user_question:
                handle_user_input(user_question, st.session_state.rag_chain)
        else:
            # Display help message if no RAG chain is available
            if "uploaded_files" not in locals() or not uploaded_files:
                st.info("👆 Upload your PDFs using the sidebar and click 'Process PDFs' to start chatting")
            else:
                st.info("👆 Click 'Process PDFs' to start chatting")
    else:  # Search mode
        if st.session_state.rag_chain:
            user_question = st.chat_input("Ask a question about your PDFs", key="search_mode_chat")
            if user_question:
                handle_user_input(user_question, st.session_state.rag_chain)
        else:
            st.info("👆 Select PDFs from the list above and click 'Process Selected PDFs' to start chatting")

if __name__ == "__main__":
    main()