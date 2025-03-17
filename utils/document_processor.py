"""
Document processing functions for the Streamlit app
"""
import os
import traceback
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
import subprocess
from PyPDF2 import PdfReader
from unstructured.partition.auto import partition
import pytesseract

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set tesseract environment variables
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/tessdata'
os.environ['TESSERACT_CMD'] = '/usr/bin/tesseract'

# Set tesseract path directly in unstructured_pytesseract
# unstructured_pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Check if Unstructured is available
try:
    logger.info("Attempting to import Unstructured...")
    from unstructured.partition.auto import partition
    UNSTRUCTURED_AVAILABLE = True
    logger.info("Unstructured successfully imported")
except Exception as e:
    UNSTRUCTURED_AVAILABLE = False
    logger.error(f"Failed to import Unstructured: {str(e)}")

# Before any processing
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def check_tesseract():
    """Check if tesseract is installed and in PATH"""
    try:
        # Debug: Print current environment
        logger.info(f"Current environment:")
        logger.info(f"PATH: {os.environ.get('PATH')}")
        logger.info(f"TESSDATA_PREFIX: {os.environ.get('TESSDATA_PREFIX')}")
        logger.info(f"PWD: {os.getcwd()}")
        
        # Debug: Check tesseract binary
        result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
        logger.info(f"Tesseract binary location: {result.stdout}")
        
        # Check possible tessdata locations
        possible_paths = [
            '/usr/share/tesseract-ocr/tessdata',
            '/usr/share/tesseract-ocr/4.00/tessdata',
            '/usr/local/share/tessdata',
            '/usr/share/tessdata'
        ]
        
        for path in possible_paths:
            logger.info(f"Checking path: {path}")
            if os.path.exists(path):
                logger.info(f"Found tessdata at: {path}")
                os.environ['TESSDATA_PREFIX'] = path
                break
        else:
            logger.error("No valid tessdata directory found")
            return False
            
        # Try to run tesseract
        logger.info("Attempting to run tesseract --version...")
        result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
        logger.info(f"Tesseract version output: {result.stdout}")
        logger.info(f"Tesseract error output: {result.stderr}")
        
        # Check if eng.traineddata exists in the found path
        eng_data = os.path.join(os.environ['TESSDATA_PREFIX'], 'eng.traineddata')
        if not os.path.exists(eng_data):
            logger.error(f"English language data not found: {eng_data}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Tesseract check failed: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def process_pdfs_with_unstructured(pdf_paths):
    """Process PDFs using Unstructured"""
    logger.info(f"Starting PDF processing with Unstructured. Paths: {pdf_paths}")
    
    # Log environment variables
    logger.info(f"TESSDATA_PREFIX: {os.environ.get('TESSDATA_PREFIX')}")
    logger.info(f"TESSERACT_CMD: {os.environ.get('TESSERACT_CMD')}")
    logger.info(f"OCR_AGENT: {os.environ.get('OCR_AGENT')}")
    
    if not UNSTRUCTURED_AVAILABLE:
        logger.error("Unstructured is not available - initialization failed")
        return [], [], []
        
    logger.info("Unstructured is available, proceeding with PDF processing")
    
    # Check if tesseract is working
    logger.info("Checking tesseract installation...")
    if not check_tesseract():
        logger.error("Tesseract is not properly configured")
        return [], [], []
    
    all_texts = []
    all_tables = []
    all_images = []
    
    with st.status("Processing PDFs...") as status:
        for i, pdf_path in enumerate(pdf_paths):
            try:
                logger.info(f"\n{'='*50}\nProcessing file {i+1}/{len(pdf_paths)}: {pdf_path}")
                
                # Verify file exists and is readable
                if not os.path.exists(pdf_path):
                    logger.error(f"File not found: {pdf_path}")
                    continue
                    
                file_size = os.path.getsize(pdf_path)
                logger.info(f"File size: {file_size} bytes")
                
                # Process file using partition with OCR
                logger.info("\nStarting partition with OCR...")
                logger.info("Partition settings:")
                logger.info("- Strategy: hi_res")
                logger.info("- OCR Languages: eng")
                logger.info("- DPI: 300")
                logger.info("- Extract Images: True")
                logger.info("- Extract Tables: True")
                
                try:
                    elements = partition(
                        file=pdf_path,
                        strategy="hi_res",
                        include_metadata=True,
                        include_page_breaks=True,
                        encoding='utf-8',
                        ocr_languages=['eng'],
                        extract_images_in_pdf=True,
                        extract_tables=True,
                        infer_table_structure=True,
                        pdf_image_dpi=300,
                        ocr_level=2  # Force OCR on everything
                    )
                    logger.info(f"\nPartition successful, got {len(elements)} elements")
                except Exception as partition_error:
                    logger.error(f"\nPartition failed: {str(partition_error)}")
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")
                    raise partition_error
                
                # Log element details for debugging
                logger.info("\nElement Details:")
                for idx, element in enumerate(elements):
                    logger.info(f"\nElement {idx}:")
                    logger.info(f"- Type: {type(element)}")
                    logger.info(f"- Has text: {hasattr(element, 'text')}")
                    if hasattr(element, 'text'):
                        text_preview = element.text[:100] + "..." if len(element.text) > 100 else element.text
                        logger.info(f"- Text preview: {text_preview}")
                    if hasattr(element, 'metadata'):
                        logger.info(f"- Metadata keys: {element.metadata.keys()}")
                
                # Process elements
                pdf_tables = []
                pdf_texts = []
                pdf_images = []
                
                logger.info("\nProcessing elements...")
                for element in elements:
                    element_type = str(type(element))
                    try:
                        if "Table" in element_type and hasattr(element, 'text'):
                            pdf_tables.append(element)
                            logger.info(f"Added table element ({len(element.text)} chars)")
                        elif hasattr(element, 'metadata') and element.metadata.get('image_base64'):
                            pdf_images.append(element.metadata['image_base64'])
                            logger.info("Added image element")
                        elif hasattr(element, 'text'):
                            pdf_texts.append(element)
                            logger.info(f"Added text element ({len(element.text)} chars)")
                        else:
                            logger.warning(f"Skipped element of type {element_type} - no text or image")
                    except Exception as element_error:
                        logger.error(f"Error processing element: {str(element_error)}")
                        logger.error(f"Element traceback:\n{traceback.format_exc()}")
                        continue
                
                # Log processing results
                logger.info(f"\nProcessing Results for {pdf_path}:")
                logger.info(f"- Text elements: {len(pdf_texts)}")
                logger.info(f"- Table elements: {len(pdf_tables)}")
                logger.info(f"- Image elements: {len(pdf_images)}")
                
                # Add to collections
                all_texts.extend(pdf_texts)
                all_tables.extend(pdf_tables)
                all_images.extend(pdf_images)
                
            except Exception as e:
                logger.error(f"\nError in unstructured processing: {str(e)}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                return [], [], []
        
        status.update(label=f"PDF processing complete! Extracted {len(all_texts)} text chunks, {len(all_tables)} tables, {len(all_images)} images.", state="complete")
    
    return all_texts, all_tables, all_images

def get_pdf_text_fallback(pdf_paths):
    """Fallback method using PyPDF2"""
    all_docs = []
    with st.status("Processing PDFs using fallback method...") as status:
        for i, pdf_path in enumerate(pdf_paths):
            status.update(label=f"Processing PDF {i+1}/{len(pdf_paths)}: {os.path.basename(pdf_path)}")
            
            try:
                pdf_reader = PdfReader(pdf_path)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": os.path.basename(pdf_path),
                                "page": page_num + 1
                            }
                        )
                        all_docs.append(doc)
            except Exception as e:
                st.error(f"Error processing {os.path.basename(pdf_path)}: {str(e)}")
    
    return all_docs

def summarize_elements(texts, tables, images, model):
    """Generate summaries for elements just like the example code"""
    text_summaries = []
    table_summaries = []
    image_summaries = []
    
    with st.status("Generating summaries...") as status:
        # Text and table summary prompt (same as example)
        prompt_text = """
        You are an assistant tasked with summarizing tables and text.
        Give a concise summary of the table or text.

        Respond only with the summary, no additional comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.

        Table or text chunk: {element}
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_text)
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        
        # Summarize texts
        if texts:
            status.update(label=f"Summarizing {len(texts)} text chunks...")
            for text in texts:
                try:
                    summary = summarize_chain.invoke(text)
                    text_summaries.append(summary)
                except Exception as e:
                    print(f"Error summarizing text: {str(e)}")
        
        # Summarize tables
        if tables:
            status.update(label=f"Summarizing {len(tables)} tables...")
            # Get HTML representation of tables like the example
            tables_html = [table.metadata.text_as_html for table in tables if hasattr(table, 'metadata') and hasattr(table.metadata, 'text_as_html')]
            
            for table_html in tables_html:
                try:
                    summary = summarize_chain.invoke(table_html)
                    table_summaries.append(summary)
                except Exception as e:
                    print(f"Error summarizing table: {str(e)}")
        
        # Summarize images
        if images:
            status.update(label=f"Analyzing {len(images)} images...")
            # Image summary prompt (similar to example)
            img_prompt_template = """Describe the image in detail. For context, 
                            the image is part of a PDF document. Be specific about 
                            any graphs, tables, or visual elements."""
            
            for i, image in enumerate(images):
                try:
                    messages = [
                        (
                            "user",
                            [
                                {"type": "text", "text": img_prompt_template},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                                },
                            ],
                        )
                    ]
                    
                    img_prompt = ChatPromptTemplate.from_messages(messages)
                    img_chain = img_prompt | model | StrOutputParser()
                    summary = img_chain.invoke("")
                    image_summaries.append(summary)
                except Exception as e:
                    print(f"Error summarizing image {i}: {str(e)}")
        
        print(f"Generated summaries: {len(text_summaries)} texts, {len(table_summaries)} tables, {len(image_summaries)} images")
        status.update(label="All summaries generated!", state="complete")
    
    return text_summaries, table_summaries, image_summaries

def process_fallback_documents(documents, model):
    """Process documents from the fallback method"""
    text_chunks = []
    summaries = []
    
    with st.status("Processing document content...") as status:
        # Text splitter for long documents
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Split long documents into chunks
        for doc in documents:
            chunks = text_splitter.split_documents([doc])
            text_chunks.extend(chunks)
        
        # Summarize chunks
        prompt_text = """
        You are an assistant tasked with summarizing text.
        Give a concise summary of the text.

        Respond only with the summary, no additional comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.

        Text chunk: {element}
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_text)
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        
        for chunk in text_chunks:
            try:
                summary = summarize_chain.invoke(chunk.page_content)
                summaries.append(summary)
            except Exception as e:
                print(f"Error summarizing chunk: {str(e)}")
    
    return text_chunks, summaries

def generate_missing_summaries(documents, model):
    """
    Generate summaries for documents that don't have them
    """
    summaries = []
    with st.status("Generating missing summaries...") as status:
        # Create a summary prompt
        prompt_text = """
        You are an assistant tasked with summarizing text.
        Give a concise summary of the text.

        Respond only with the summary, no additional comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.

        Text: {element}
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_text)
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        
        total = len(documents)
        for i, doc in enumerate(documents):
            status.update(label=f"Generating summary {i+1}/{total}...")
            try:
                # Get the content from the document
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                elif hasattr(doc, 'text'):
                    content = doc.text
                else:
                    content = str(doc)
                
                # Limit content length if needed
                content = content[:5000]  # Limit to 5000 chars to avoid token limits
                
                summary = summarize_chain.invoke(content)
                summaries.append(summary)
            except Exception as e:
                print(f"Error generating summary: {str(e)}")
                summaries.append("Summary not available.")
        
        status.update(label=f"Generated {len(summaries)} summaries", state="complete")
    return summaries