"""
Text PDF Chat - Simple Text-based RAG System
"""
from utils.sqlite_fix import fix_sqlite
fix_sqlite()

import sys
import os
import time
import shutil

import uuid
import warnings
import traceback
import logging
import joblib
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Add the project directory to the Python path to fix import issues
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from utils.helpers import save_uploaded_file, load_preprocessed_data, ensure_chroma_directory, display_conversation
from utils.html_templates import inject_css, bot_template, user_template
from utils.text_processor import process_pdf_text, summarize_text_chunks

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Workaround for PyTorch error in Streamlit file watcher
import streamlit.watcher.path_watcher
original_watch_dir = streamlit.watcher.path_watcher.watch_dir

def patched_watch_dir(path, *args, **kwargs):
    if "torch" in path or "_torch" in path or "site-packages" in path:
        # Skip watching PyTorch-related directories
        return None
    return original_watch_dir(path, *args, **kwargs)

streamlit.watcher.path_watcher.watch_dir = patched_watch_dir

# Load environment variables
load_dotenv()  # Keep this for local development

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

# Check if collection file exists
if not os.path.exists(PREPROCESSED_COLLECTION_FILE):
    print(f"Warning: Preprocessed collection file not found at {PREPROCESSED_COLLECTION_FILE}")
    print("You'll need to run the preprocessing script first or switch to upload mode.")

# Cache expensive operations
@st.cache_resource
def get_openai_model(model_name):
    """Cache the OpenAI model to avoid reloading it"""
    return ChatOpenAI(model=model_name, temperature=0.2)

@st.cache_resource
def get_embeddings():
    """Cache the embeddings model to avoid reloading it"""
    return OpenAIEmbeddings()

def build_prompt(kwargs):
    """Build a prompt for the RAG system"""
    context = kwargs["context"]
    user_question = kwargs["question"]
    chat_history = kwargs.get("chat_history", [])

    context_text = ""
    for document in context:
        if hasattr(document, 'page_content'):
            context_text += document.page_content + "\n\n"
        else:
            context_text += str(document) + "\n\n"

    # Format chat history
    chat_history_text = ""
    if chat_history:
        for msg in chat_history:
            role = "Human" if msg.type == "human" else "Assistant"
            chat_history_text += f"{role}: {msg.content}\n"

    # Construct prompt with context
    prompt_template = f"""
    Answer the question based only on the following context:
    
    Context: {context_text}
    
    {chat_history_text}
    
    Question: {user_question}
    
    Provide a clear, detailed answer that directly addresses the question. If you can't answer based on the provided context, simply state that you don't have enough information.
    """

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_template),
        ]
    )

def get_conversational_rag_chain(retriever, model):
    """Create the RAG chain for conversation"""
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "chat_history": lambda x: memory.chat_memory.messages
        }
        | RunnableLambda(build_prompt)
        | model
        | StrOutputParser()
    )
    
    return chain, memory

def handle_userinput(user_question, rag_chain, memory):
    """Process the user question and update the chat"""
    if not user_question:
        return
    
    with st.spinner("Thinking..."):
        try:
            response = rag_chain.invoke(user_question)
            
            # Update memory
            memory.chat_memory.add_user_message(user_question)
            memory.chat_memory.add_ai_message(response)
            
            # Rerun to show updated conversation
            st.rerun()
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
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
            # Use our flexible data loading function
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
    summary_docs = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]})
        for i, summary in enumerate(summaries)
    ]
    
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, text_chunks[:len(summaries)])))
    
    return retriever

def initialize_session_state():
    """Initialize session state variables"""
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    if "memory" not in st.session_state:
        st.session_state.memory = None
    
    if "mode" not in st.session_state:
        st.session_state.mode = "upload"  # Default to upload mode
    
    if "temp_pdf_files" not in st.session_state:
        st.session_state.temp_pdf_files = []

def on_file_upload():
    """Clear conversation when new files are uploaded"""
    st.session_state.conversation = []
    st.session_state.rag_chain = None
    st.session_state.memory = None

def handle_mode_change():
    """Handle changes in the application mode"""
    st.session_state.conversation = []
    st.session_state.rag_chain = None
    st.session_state.memory = None

def cleanup_temp_files(pdf_paths):
    """Clean up temporary PDF files"""
    try:
        for path in pdf_paths:
            if os.path.exists(path) and path.startswith("./temp_pdf_files/"):
                os.remove(path)
                print(f"Removed temporary file: {path}")
    except Exception as e:
        print(f"Error cleaning up temp files: {e}")

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
                st.error("Could not extract any text from the provided PDFs. Please try different files.")
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

def main():
    # Initialize session state
    initialize_session_state()
    
    # Page configuration
    st.set_page_config(page_title="Text PDF Chat", page_icon="📄", layout="wide")
    
    # Inject custom CSS
    inject_css()
    
    # Sidebar
    with st.sidebar:
        st.title("Text PDF Chat")
        st.markdown("---")
        
        # Mode selection
        mode = st.radio(
            "Choose mode:",
            ["Upload PDFs", "Use Preloaded Collection"],
            key="mode_radio",
            index=0 if st.session_state.mode == "upload" else 1,
            on_change=handle_mode_change
        )
        
        st.session_state.mode = "upload" if mode == "Upload PDFs" else "preloaded"
        
        if st.session_state.mode == "upload":
            # File uploader widget
            uploaded_files = st.file_uploader(
                "Upload your PDFs",
                accept_multiple_files=True,
                type="pdf",
                help="Upload one or more PDF files to chat with",
                on_change=on_file_upload
            )
            
            process_button = st.button("Process PDFs", type="primary")
            
            if process_button and uploaded_files:
                # Process the uploaded files
                retriever = process_uploaded_files(uploaded_files)
                
                if retriever:
                    # Create the RAG chain
                    model = get_openai_model("gpt-4o-mini")
                    st.session_state.rag_chain, st.session_state.memory = get_conversational_rag_chain(
                        retriever, model
                    )
            
        else:  # Preloaded collection mode
            st.info("Using preloaded PDF collection")
            
            load_button = st.button("Load Collection", type="primary")
            
            if load_button:
                # Load the preloaded collection
                retriever = load_preloaded_collection()
                
                if retriever:
                    # Create the RAG chain
                    model = get_openai_model("gpt-4o-mini")
                    st.session_state.rag_chain, st.session_state.memory = get_conversational_rag_chain(
                        retriever, model
                    )
        
        # About section
        st.markdown("---")
        st.markdown(
            """
            ### About
            
            This application allows you to chat with your PDF documents.
            Upload PDFs or use the preloaded collection, then ask questions
            about their content.
            
            Powered by:
            - LangChain
            - OpenAI
            - ChromaDB
            - Streamlit
            """
        )
    
    # Main content area
    st.header("Chat with your PDFs")
    
    # Show the conversation
    display_conversation()
    
    # Chat input
    if st.session_state.rag_chain:
        user_question = st.chat_input("Ask a question about your PDFs")
        if user_question:
            handle_userinput(user_question, st.session_state.rag_chain, st.session_state.memory)
    else:
        # Display help message if no RAG chain is available
        if st.session_state.mode == "upload":
            if "uploaded_files" not in locals() or not uploaded_files:
                st.info("👆 Upload your PDFs using the sidebar and click 'Process PDFs' to start chatting")
            else:
                st.info("👆 Click 'Process PDFs' in the sidebar to start chatting")
        else:
            st.info("👆 Click 'Load Collection' in the sidebar to start chatting")

if __name__ == "__main__":
    main()