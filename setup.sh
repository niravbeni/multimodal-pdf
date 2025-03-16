#!/bin/bash

echo "Installing Tesseract from prebuilt binaries..."

# Create directories
mkdir -p /usr/local/bin
mkdir -p /usr/local/share/tessdata

# Download prebuilt tesseract binary
curl -L -o /usr/local/bin/tesseract https://github.com/tesseract-ocr/tesseract/releases/download/5.3.3/tesseract-5.3.3-linux-x86_64.tar.gz
chmod +x /usr/local/bin/tesseract

# Download English language data
curl -L -o /usr/local/share/tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata

# Set permissions
chmod -R 755 /usr/local/share/tessdata

# Print debug info
echo "Tesseract installation info:"
/usr/local/bin/tesseract --version
ls -l /usr/local/bin/tesseract
ls -l /usr/local/share/tessdata 