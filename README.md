# Text PDF Chat

A streamlined, text-based PDF chat application that enables users to upload PDF documents and ask questions about their content.

## Overview

This application extracts text from PDF files, processes it into chunks, and uses a retrieval-augmented generation (RAG) approach to answer questions about the content. It focuses solely on text extraction and processing, making it more efficient and producing smaller output files compared to multimodal approaches.

## Features

- **PDF Text Extraction**: Extracts text from PDF files using the Unstructured library
- **Text Chunking**: Splits large documents into manageable chunks for better processing
- **Semantic Search**: Uses embeddings to find the most relevant content for user queries
- **Conversational Interface**: Clean, intuitive chat interface for asking questions
- **Preloaded Collection Support**: Option to use a preprocessed collection of documents
- **Summary Generation**: Creates summaries of text chunks to improve retrieval quality

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/text-pdf-chat.git
   cd text-pdf-chat
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

### Usage

#### Running the Application

```bash
streamlit run app.py
```

#### Using the Application

1. **Upload Mode**:
   - Select "Upload PDFs" in the sidebar
   - Upload one or more PDF files
   - Click "Process PDFs"
   - Start asking questions about your documents

2. **Preloaded Collection Mode**:
   - First, preprocess your PDFs (see below)
   - Select "Use Preloaded Collection" in the sidebar
   - Click "Load Collection"
   - Start asking questions about the preloaded documents

#### Preprocessing PDFs

To create a preprocessed collection:

```bash
python preprocess.py --input /path/to/your/pdfs --output ./preprocessed_data/primary_collection.joblib
```

Options:
- `--input`, `-i`: Directory containing PDFs or path to a specific PDF file (required)
- `--output`, `-o`: Output path for the processed data (default: ./preprocessed_data/primary_collection.joblib)
- `--model`, `-m`: OpenAI model to use for summarization (default: gpt-4o-mini)

## Project Structure

```
text-pdf-chat/
├── app.py                     # Main Streamlit application
├── preprocess.py              # Script to preprocess PDFs
├── requirements.txt           # Python dependencies
├── utils/
│   ├── helpers.py             # Helper functions for the application
│   ├── html_templates.py      # HTML/CSS templates for the UI
│   ├── sqlite_fix.py          # Fix for SQLite issues
│   └── text_processor.py      # Text extraction and processing functions
├── preprocessed_data/         # Directory for preprocessed collections
└── temp_pdf_files/            # Temporary directory for uploaded PDFs
```

## Technical Details

- **Text Extraction**: Uses Unstructured library to extract text from PDFs
- **Vector Database**: ChromaDB for storing and retrieving document embeddings
- **Embeddings**: OpenAI embeddings for semantic search capabilities
- **Language Model**: OpenAI's GPT models for generating responses
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