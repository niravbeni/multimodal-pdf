# PDF Chat System Project Structure

This document explains the structure and components of the PDF Chat System.

## Core Components

### 1. Main Application Files

- **app.py**: The main Streamlit application that provides the user interface
- **pdf_processor.py**: All-in-one script for processing PDFs and importing them to the database
- **setup.py**: Setup script to initialize the database and prepare the environment

### 2. Database

The database layer is in the `database/` directory:

- **database/config.py**: Database connection configuration
- **database/create_db.py**: Script to create the database
- **database/models.py**: SQLAlchemy models defining the database schema
- **database/pdf_manager.py**: Functions for managing PDFs in the database

### 3. Utilities

Helper functions and utilities in the `utils/` directory:

- **utils/helpers.py**: Generic helper functions
- **utils/html_templates.py**: HTML and CSS templates for the UI
- **utils/sqlite_fix.py**: Fixes for SQLite compatibility issues
- **utils/text_processor.py**: Functions for processing PDF text

## Database Schema

The system uses the following database models:

### PDF

Stores PDF documents and their metadata:

- `id`: Unique identifier
- `filename`: Original filename
- `file_path`: Path to the PDF file (if stored on disk)
- `content_hash`: SHA-256 hash of file content for deduplication
- `file_size`: Size in bytes
- `page_count`: Number of pages
- `title`: PDF title (from metadata)
- `author`: PDF author (from metadata)
- `creation_date`: When the PDF was created
- `last_modified`: When the PDF was last modified
- `uploaded_at`: When the PDF was added to the database
- `summary`: Generated summary of the PDF content
- `full_text`: Complete extracted text content
- `embedding`: Vector embedding for semantic search

### PDFChunk

Stores chunks of text from PDFs for more granular retrieval:

- `id`: Unique identifier
- `pdf_id`: Reference to the parent PDF
- `chunk_index`: Position in the sequence of chunks
- `content`: Text content of the chunk
- `embedding`: Vector embedding for the chunk
- `page_number`: Page number where this chunk appears

### Tag

Stores tags that can be applied to PDFs:

- `id`: Unique identifier
- `name`: Tag name

## Flow of Data

1. **PDF Processing**:
   - PDFs are imported using `pdf_processor.py`
   - Text is extracted and chunked
   - Summaries are generated
   - Everything is stored in the database

2. **Search and Retrieval**:
   - User searches for PDFs by content/metadata using `pdf_manager.py`
   - Selected PDFs are processed for RAG (Retrieval Augmented Generation)
   - PDF chunks are embedded and stored in ChromaDB
   - The MultiVectorRetriever is created for semantic search

3. **Chat Interface**:
   - User asks questions in the chat interface
   - System retrieves relevant chunks using vector similarity
   - LLM generates responses using the retrieved context
   - Responses include citations to relevant pages in PDFs

## Directory Structure

```
pdf-chat/
├── app.py                     # Main Streamlit application
├── pdf_processor.py           # All-in-one PDF processing script
├── setup.py                   # Setup and initialization script
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (OpenAI API key, etc.)
├── database/                  # Database layer
│   ├── config.py              # Database configuration
│   ├── create_db.py           # Database creation script
│   ├── models.py              # SQLAlchemy models (schema)
│   └── pdf_manager.py         # PDF database operations
├── utils/                     # Utilities
│   ├── helpers.py             # Helper functions
│   ├── html_templates.py      # HTML/CSS templates
│   ├── sqlite_fix.py          # SQLite compatibility fixes
│   └── text_processor.py      # Text processing functions
├── pdf_database/              # Default PDF storage directory
├── temp_pdf_files/            # Temporary storage for uploads
└── chroma_db/                 # Vector database storage
```

## Runtime Workflow

1. **Setup**: Run `python setup.py` to initialize the environment
2. **Import PDFs**: Run `python pdf_processor.py --input /path/to/pdfs`
3. **Start App**: Run `streamlit run app.py`
4. **Use App**:
   - Search for PDFs by content or metadata
   - Select and process PDFs of interest
   - Chat with the selected PDFs using natural language 