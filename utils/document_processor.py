"""
Document processing functions for the Streamlit app
"""
import os
import traceback
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.document import Document
import logging
from langchain.schema import HumanMessage

# At the top of the file, add debug logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if Unstructured is available
try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
    logger.info("Unstructured successfully imported")
except Exception as e:
    UNSTRUCTURED_AVAILABLE = False
    logger.error(f"Failed to import Unstructured: {str(e)}")

def process_pdfs_with_unstructured(pdf_paths):
    """Process PDFs using Unstructured"""
    elements = []
    
    # Create a status container with spinner
    with st.status("Processing PDFs...", expanded=True) as status:
        for pdf_path in pdf_paths:
            # Update status for each file
            status.update(label=f"📄 Processing {os.path.basename(pdf_path)}...", expanded=True)
            
            try:
                file_elements = partition_pdf(
                    filename=pdf_path,
                    include_metadata=True,
                    strategy="hi_res",
                    extract_images_in_pdf=True,
                    extract_tables=True
                )
                elements.extend(file_elements)
                
                # Show success for each file
                status.update(label=f"✅ Processed {os.path.basename(pdf_path)}")
                
            except Exception as e:
                # Show error if processing fails
                status.error(f"❌ Error processing {os.path.basename(pdf_path)}: {str(e)}")
                continue
    
        # Show final stats
        total_elements = len(elements)
        texts = [e for e in elements if "Text" in str(type(e))]
        tables = [e for e in elements if "Table" in str(type(e))]
        images = [e for e in elements if "Image" in str(type(e))]
        
        status.update(
            label=f"✨ Processing complete! Found {len(texts)} text blocks, {len(tables)} tables, and {len(images)} images",
            state="complete"
        )
    
    return texts, tables, images

def get_pdf_text_fallback(pdf_paths):
    """Fallback method using PyPDF2"""
    from PyPDF2 import PdfReader
    
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

        Table or text chunk: {text}
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_text)
        summarize_chain = prompt | model | StrOutputParser()
        
        # Summarize texts
        if texts:
            status.update(label=f"Summarizing {len(texts)} text chunks...")
            for text in texts:
                try:
                    # Convert text to string if needed
                    text_content = text.text if hasattr(text, 'text') else str(text)
                    summary = summarize_chain.invoke({"text": text_content})
                    text_summaries.append(summary)
                except Exception as e:
                    print(f"Error summarizing text: {str(e)}")

        # Summarize tables
        if tables:
            status.update(label=f"Summarizing {len(tables)} tables...")
            tables_html = [table.metadata.text_as_html for table in tables if hasattr(table, 'metadata') and hasattr(table.metadata, 'text_as_html')]
            
            for table_html in tables_html:
                try:
                    summary = summarize_chain.invoke({"text": table_html})
                    table_summaries.append(summary)
                except Exception as e:
                    print(f"Error summarizing table: {str(e)}")

        # Summarize images
        if images:
            status.update(label=f"Analyzing {len(images)} images...")
            for i, img in enumerate(images):
                try:
                    # Get base64 data from the image
                    image_data = None
                    if hasattr(img, 'metadata') and hasattr(img.metadata, 'image_base64'):
                        image_data = img.metadata.image_base64
                    elif hasattr(img, 'image'):
                        import base64
                        from io import BytesIO
                        buffered = BytesIO()
                        img.image.save(buffered, format="JPEG")
                        image_data = base64.b64encode(buffered.getvalue()).decode()
                    
                    if image_data:
                        messages = [
                            HumanMessage(
                                content=[
                                    {
                                        "type": "text",
                                        "text": "Describe this image in detail, focusing on any graphs, tables, or visual elements."
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                                    }
                                ]
                            )
                        ]
                        img_prompt = ChatPromptTemplate.from_messages(messages)
                        img_chain = img_prompt | model | StrOutputParser()
                        summary = img_chain.invoke({})
                        image_summaries.append(summary)
                except Exception as e:
                    print(f"Error summarizing image: {str(e)}")
                    image_summaries.append("Failed to summarize image")

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