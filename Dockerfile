# Use Python 3.9 slim image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    poppler-data \
    libmagic1 \
    ghostscript \
    libheif-dev \
    python3-dev \
    build-essential \
    pkg-config \
    libpoppler-dev \
    libpoppler-cpp-dev \
    poppler-utils \
    libreoffice \
    pandoc \
    libmagic1 \
    libxml2-dev \
    libxslt1-dev \
    antiword \
    unrtf \
    && rm -rf /var/lib/apt/lists/*

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
ENV OCR_AGENT=pytesseract
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV MAGIC=/usr/lib/file/magic.mgc

# Set up working directory
WORKDIR /app

# Install dependencies in order
COPY requirements.txt .
RUN pip install --no-cache-dir unstructured[all] && \
    pip install --no-cache-dir unstructured-inference && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"] 