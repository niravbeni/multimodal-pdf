# Use Python 3.9 slim image
FROM python:3.9-slim-buster

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    curl \
    poppler-utils \
    poppler-data \
    libmagic1 \
    libreoffice \
    pandoc \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/tesseract-ocr/tessdata \
    && curl -L -o /usr/share/tesseract-ocr/tessdata/eng.traineddata \
       https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata \
    && chmod 644 /usr/share/tesseract-ocr/tessdata/eng.traineddata \
    && tesseract --version \
    && tesseract --list-langs

# Verify tesseract installation and data
RUN ls -la /usr/share/tesseract-ocr/tessdata/ && \
    echo "TESSDATA_PREFIX=$TESSDATA_PREFIX" && \
    tesseract --version && \
    which tesseract && \
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

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Test OCR setup
RUN python -c "import pytesseract; print('Tesseract path:', pytesseract.get_tesseract_cmd()); print('Version:', pytesseract.get_tesseract_version())" && \
    python -c "import os; print('TESSDATA_PREFIX:', os.environ.get('TESSDATA_PREFIX')); print('Directory exists:', os.path.exists(os.environ.get('TESSDATA_PREFIX', '')));" && \
    # Test actual OCR functionality
    python -c "import pytesseract; print('OCR test:', bool(pytesseract.get_languages()))"

# Copy the rest of the application
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Set Streamlit config
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "app.py"] 