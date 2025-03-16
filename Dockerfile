# Use Python 3.9 slim image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libmagic1 \
    libreoffice \
    pandoc \
    ghostscript \
    libgl1-mesa-glx \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir "unstructured[all]" "unstructured[pdf]" && \
    pip install --no-cache-dir -r requirements.txt

# Install additional PDF dependencies
RUN pip install --no-cache-dir \
    "pdfminer.six>=20221105" \
    "pdf2image>=1.16.3" \
    "pypdfium2>=4.20.0" \
    "pypdf>=3.17.1" \
    "pdfplumber>=0.10.2" \
    "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2"

# Copy the rest of the application
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"] 