# Multimodal PDF AI Assistant

A powerful PDF management system that allows you to import, search, and chat with your PDF documents using AI.

## Features

- **PDF Import**: Import PDF files, extract text and metadata, and store in a database
- **Text Extraction**: Extract text from PDF files, handling various PDF formats
- **Summarization**: Generate summaries of PDF documents for quick reference
- **Search**: Search across your PDF documents using semantic search
- **Chat**: Ask questions about your PDFs and get AI-powered responses
- **Batch Processing**: Process large PDF documents efficiently with optimized batch processing
- **Parallel API Calls**: Handle multiple chunks of text concurrently for faster processing

## Getting Started

### Prerequisites

- Python 3.9+
- [pip](https://pip.pypa.io/en/stable/)
- OpenAI API Key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/niravbeni/multimodal-pdf.git
cd multimodal-pdf
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### Setup

Run the setup script to initialize the database, create necessary directories and verify dependencies:

```bash
python setup.py
```

### Usage

#### 1. Process PDFs

Process a directory of PDF files:

```bash
python pdf_processor.py --input /path/to/your/pdfs
```

Options:
- `--input` or `-i`: Directory containing PDFs to process (required)
- `--keep` or `-k`: Keep original PDF files (default: True)
- `--move` or `-m`: Move processed files to a different directory
- `--processed-dir` or `-p`: Directory to move processed files to (required if using --move)

For large PDFs (1000+ pages), the batch processing system will automatically handle chunks in parallel for improved performance.

#### 2. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

#### 3. Search and Chat

- Use the **Search PDFs** tab to search across your imported PDFs
- Use the **Chat with PDFs** tab to ask questions about your PDFs

## Project Structure

For detailed information about the project structure and components, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 