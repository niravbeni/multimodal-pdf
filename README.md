# PDF Chat System

A streamlined PDF management and chat system that enables users to upload PDF documents, search them by content, and ask questions about their content.

## Overview

This application extracts text from PDF files, processes it into chunks, and uses a retrieval-augmented generation (RAG) approach to answer questions about the content. It stores PDF metadata and content in a database for efficient searching and retrieval.

## Features

- **PDF Management**: Import PDFs from a folder to the database with a single command
- **Automatic Text Extraction**: Extract text content from PDFs for search and analysis
- **Automatic Summary Generation**: Create summaries of PDFs to improve searchability
- **Content Search**: Search PDF content and metadata to find relevant documents
- **Conversational Interface**: Clean chat interface for asking questions about PDF content
- **Semantic Search**: Use embeddings to find the most relevant content for user queries

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- PostgreSQL database (optional, SQLite will be used by default)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdf-chat.git
   cd pdf-chat
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Create a `.env` file in the project root:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```
   - Alternatively, add your API key to `.streamlit/secrets.toml`:
     ```
     OPENAI_API_KEY="your_openai_api_key_here"
     ```

### Database Setup

```bash
python -m database.create_db
```

## Usage

### 1. Process PDFs (Import to Database)

Using the all-in-one PDF processor, you can import PDFs from a folder, extract their text, generate summaries, and store everything in the database:

```bash
python pdf_processor.py --input /path/to/pdf/folder
```

Options:
- `--input`, `-i`: Directory containing PDFs (required)
- `--keep`, `-k`: Keep original PDF files (store path in database) (default: True)
- `--move`, `-m`: Move processed files to a different directory
- `--processed-dir`, `-p`: Directory to move processed files to (required if using --move)

### 2. Launch the Chat Application

```bash
streamlit run app.py
```

### 3. Using the Application

1. **Search Mode**:
   - Select "Search PDF Collection" in the sidebar
   - Search for PDFs by content, filename, or summary
   - Select PDFs from the search results
   - Click "Process Selected PDFs"
   - Start asking questions about your documents

2. **Upload Mode**:
   - Select "Upload PDFs" in the sidebar
   - Upload one or more PDF files
   - Click "Process PDFs"
   - Start asking questions about your documents

## Project Structure

```
pdf-chat/
├── app.py                     # Main Streamlit application
├── pdf_processor.py           # All-in-one PDF processing script
├── requirements.txt           # Python dependencies
├── database/                  # Database models and utilities
│   ├── config.py              # Database configuration
│   ├── create_db.py           # Database creation script
│   ├── models.py              # SQLAlchemy database models
│   └── pdf_manager.py         # PDF database operations
├── utils/                     # Utility functions
│   ├── helpers.py             # Helper functions
│   ├── html_templates.py      # HTML/CSS templates for the UI
│   ├── sqlite_fix.py          # Fix for SQLite issues
│   └── text_processor.py      # Text extraction and processing functions
└── pdf_database/              # Default directory for PDF storage
```

## Technical Details

- **Database**: SQLAlchemy ORM with SQLite or PostgreSQL
- **Text Extraction**: PyPDF2 for extracting text from PDFs
- **Vector Database**: ChromaDB for storing and retrieving document embeddings
- **Embeddings**: OpenAI embeddings for semantic search capabilities
- **Language Model**: OpenAI's GPT models for generating responses and summaries
- **Retrieval Strategy**: Multi-vector retrieval with summaries for better results

## License

MIT License

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI](https://openai.com/)
- [Streamlit](https://streamlit.io/)
- [Unstructured](https://github.com/Unstructured-IO/unstructured)
- [ChromaDB](https://github.com/chroma-core/chroma) 
- See `requirements.txt` for full dependencies 