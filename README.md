# Text PDF Chat

A streamlined application that allows users to chat with PDF documents using text-based RAG (Retrieval-Augmented Generation).

## Features

- Chat with PDF documents using natural language
- Efficient text extraction and processing
- Two modes of operation:
  - Use a preloaded collection of documents
  - Upload your own PDFs for analysis
- Powered by OpenAI's advanced language models

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key in `.env` or `.streamlit/secrets.toml`
4. Run the application:
   ```
   streamlit run app.py
   ```

## Preprocessing Documents

To preprocess a collection of documents for the preloaded mode:

1. Place PDF files in the `preloaded_pdfs` directory
2. Run the preprocessing script:
   ```
   python batch_preprocess.py
   ```

## Project Structure

- `app.py`: Main Streamlit application
- `utils/`: Utility functions for text processing and UI
- `utils/text_processor.py`: Text extraction and summarization
- `preprocessed_data/`: Directory for storing preprocessed document collections

## Requirements

- Python 3.8+
- OpenAI API key
- See `requirements.txt` for full dependencies 