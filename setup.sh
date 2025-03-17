#!/bin/bash

echo "Starting Tesseract setup..."

# Check if tesseract is installed
which tesseract || echo "Tesseract not found in PATH"

# Check all possible tessdata locations
echo "Checking tessdata locations..."
LOCATIONS=(
    "/usr/share/tesseract-ocr/tessdata"
    "/usr/share/tesseract-ocr/4.00/tessdata"
    "/usr/local/share/tessdata"
    "/usr/share/tessdata"
)

for loc in "${LOCATIONS[@]}"; do
    echo "Checking $loc"
    ls -la $loc 2>/dev/null || echo "$loc not found"
done

# Create our tessdata directory
echo "Creating tessdata directory..."
mkdir -p /usr/share/tesseract-ocr/tessdata

# Download language data
echo "Downloading language data..."
curl -L -o /usr/share/tesseract-ocr/tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata

# Set permissions
echo "Setting permissions..."
chmod 755 /usr/share/tesseract-ocr/tessdata
chmod 644 /usr/share/tesseract-ocr/tessdata/eng.traineddata

# Verify setup
echo "Verifying setup..."
tesseract --version
echo "Language files:"
ls -la /usr/share/tesseract-ocr/tessdata
echo "Available languages:"
tesseract --list-langs

# Export environment variables
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata
export TESSERACT_CMD=$(which tesseract)

echo "Setup complete. TESSDATA_PREFIX=$TESSDATA_PREFIX" 