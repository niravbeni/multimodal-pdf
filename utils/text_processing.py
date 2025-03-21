import PyPDF2
from typing import List, Tuple

def process_pdf_text(file_path: str) -> List[Tuple[str, int]]:
    """
    Process a PDF file and return a list of (chunk_text, page_number) tuples.
    Each chunk is a meaningful section of text from the PDF.
    """
    chunks = []
    
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            
            if not text.strip():
                continue
            
            # For now, we'll treat each page as one chunk
            # In a more sophisticated implementation, you might want to:
            # 1. Split pages into smaller chunks based on content
            # 2. Merge small pages into larger chunks
            # 3. Use NLP to find natural break points
            chunks.append((text.strip(), page_num + 1))
    
    return chunks 