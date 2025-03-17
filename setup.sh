#!/bin/bash

echo "Setting up Tesseract..."

# Create tessdata directory
mkdir -p /usr/share/tesseract-ocr/tessdata

# Download English language data directly to the correct location
curl -L -o /usr/share/tesseract-ocr/tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata

# Set permissions
chmod 755 /usr/share/tesseract-ocr/tessdata
chmod 644 /usr/share/tesseract-ocr/tessdata/eng.traineddata

# Verify
echo "Tesseract installation info:"
tesseract --version
ls -la /usr/share/tesseract-ocr/tessdata
tesseract --list-langs 