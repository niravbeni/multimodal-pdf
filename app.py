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
from utils.helpers import save_uploaded_file, load_preprocessed_data, ensure_chroma_directory
from utils.html_templates import inject_css
from utils.text_processor import process_pdf_text, summarize_text_chunks

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

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
        context_text += f"[Document {i+1}: {source}, Page {page}]\n{doc.page_content}\n\n"

    # Format chat history if provided
    chat_history_text = ""
    if chat_history:
        for message in chat_history:
            role = "Human" if message["role"] == "user" else "Assistant"
            chat_history_text += f"{role}: {message['content']}\n"

    # Construct prompt with context and citation instructions
    prompt = f"""
    Answer the question based only on the following context:
    
    Context: {context_text}
    
    {chat_history_text}
    
    Question: {question}
    
    Provide a clear, detailed answer that directly addresses the question.
    Include citations to the source documents in your answer using the format [Document title, Page X].
    For example: "According to [Document A, Page 5], the main concept is..."
    
    If multiple sources support a statement, cite them all like: [Document A, Page 5; Document B, Page 3]
    
    If you can't answer based on the provided context, simply state that you don't have enough information.
    """

    return prompt

def get_rag_chain(retriever, model):
    """Create the RAG chain for conversation"""
    def generate_response(query):
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(query)
        
        # Get chat history from session state
        chat_history = st.session_state.conversation if "conversation" in st.session_state else []
        
        # Build the prompt
        prompt = build_rag_prompt(docs, query, chat_history)
        
        # Generate the response
        response = model.invoke(prompt)
        
        # Store the retrieved documents in session state for citation tracking
        st.session_state.last_retrieved_docs = docs
        
        return response
    
    return generate_response

def handle_user_input(user_question, rag_chain):
    """Process the user question and update the chat"""
    if not user_question:
        return
    
    # Add user message to conversation
    st.session_state.conversation.append({"role": "user", "content": user_question})
    
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
            # This replaces citations like [Document.pdf, Page 5] with formatted ones
            import re
            citation_pattern = r'\[(.*?), Page (\d+)\]'
            
            def citation_replacer(match):
                doc_name = match.group(1)
                page_num = match.group(2)
                return f'<span class="citation">[{doc_name}, Page {page_num}]</span>'
            
            # Apply the formatting to citations
            formatted_response = re.sub(citation_pattern, citation_replacer, response_content)
            
            # Add the formatted response to the conversation (without HTML)
            st.session_state.conversation.append({"role": "assistant", "content": response_content})
            
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
    
    if "mode" not in st.session_state:
        st.session_state.mode = "upload"  # Default to upload mode
    
    if "temp_pdf_files" not in st.session_state:
        st.session_state.temp_pdf_files = []
        
    if "show_chat_history" not in st.session_state:
        st.session_state.show_chat_history = True

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
        # If chat history is disabled, only show the latest exchange
        if not st.session_state.show_chat_history:
            if len(messages) >= 2:  # Make sure we have at least one exchange
                with st.chat_message(messages[-2]["role"], avatar=USER_AVATAR if messages[-2]["role"] == "user" else ASSISTANT_AVATAR):
                    st.markdown(messages[-2]["content"])
                with st.chat_message(messages[-1]["role"], avatar=USER_AVATAR if messages[-1]["role"] == "user" else ASSISTANT_AVATAR):
                    st.markdown(messages[-1]["content"])
            return
        
        # Otherwise, show full history
        for message in messages:
            avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

def main():
    # Initialize session state
    initialize_session_state()
    
    # Page configuration
    st.set_page_config(page_title="Text PDF Chat", page_icon="📄", layout="wide")
    
    # Inject custom CSS
    st.markdown(inject_css(), unsafe_allow_html=True)
    
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
            on_change=reset_conversation
        )
        
        st.session_state.mode = "upload" if mode == "Upload PDFs" else "preloaded"
        
        if st.session_state.mode == "upload":
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
            
        else:  # Preloaded collection mode
            st.info("Using preloaded PDF collection")
            
            load_button = st.button("Load Collection", type="primary")
            
            if load_button:
                # Load the preloaded collection
                retriever = load_preloaded_collection()
                
                if retriever:
                    # Create the RAG chain
                    model = get_openai_model("gpt-4o-mini")
                    st.session_state.rag_chain = get_rag_chain(retriever, model)
        
        st.markdown("---")
        
        # Show history toggle
        st.checkbox("Show chat history", value=st.session_state.show_chat_history, key="show_history_toggle", 
                    on_change=lambda: setattr(st.session_state, "show_chat_history", st.session_state.show_history_toggle))
        
        # Clear conversation button
        if st.button("Clear Conversation"):
            reset_conversation()

    # Main content area
    st.header("Chat with your PDFs")
    
    # Show the conversation
    display_conversation()
    
    # Chat input
    if st.session_state.rag_chain:
        user_question = st.chat_input("Ask a question about your PDFs")
        if user_question:
            handle_user_input(user_question, st.session_state.rag_chain)
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