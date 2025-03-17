# Use Python 3.9 slim image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Tesseract and its dependencies
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    # PDF processing
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
    libreoffice \
    pandoc \
    libxml2-dev \
    libxslt1-dev \
    antiword \
    unrtf \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/tesseract-ocr/tessdata \
    && cp /usr/share/tesseract-ocr/4.00/tessdata/eng.traineddata /usr/share/tesseract-ocr/tessdata/ \
    && tesseract --version \
    && tesseract --list-langs

# Verify tesseract installation and data
RUN tesseract --version && \
    ls -la /usr/share/tesseract-ocr/tessdata/eng.traineddata && \
    tesseract --list-langs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata
ENV OCR_AGENT=pytesseract
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV MAGIC=/usr/lib/file/magic.mgc
ENV TESSERACT_CMD=/usr/bin/tesseract

# Set up working directory
WORKDIR /app

# Install dependencies in order
COPY requirements.txt .
RUN pip install --no-cache-dir unstructured[all] && \
    pip install --no-cache-dir unstructured-inference && \
    pip install --no-cache-dir unstructured-pytesseract && \
    pip install --no-cache-dir -r requirements.txt

# Test OCR setup
RUN python -c "import pytesseract; print('Tesseract path:', pytesseract.get_tesseract_cmd()); print('Version:', pytesseract.get_tesseract_version())" && \
    python -c "import os; print('TESSDATA_PREFIX:', os.environ.get('TESSDATA_PREFIX')); print('Directory exists:', os.path.exists(os.environ.get('TESSDATA_PREFIX', '')));"

# Copy the rest of the application
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"] 