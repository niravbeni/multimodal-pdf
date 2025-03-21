from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, LargeBinary, Float, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

# Association table for many-to-many relationship between PDFs and tags
pdf_tags = Table(
    'pdf_tags',
    Base.metadata,
    Column('pdf_id', Integer, ForeignKey('pdfs.id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id'), primary_key=True)
)

class PDF(Base):
    __tablename__ = 'pdfs'

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=True)  # Path to PDF file if stored on disk
    content_hash = Column(String(64), nullable=False, unique=True)  # SHA-256 hash of file content
    file_size = Column(Integer, nullable=False)  # Size in bytes
    page_count = Column(Integer, nullable=False)
    
    # Metadata
    title = Column(String(512), nullable=True)
    author = Column(String(255), nullable=True)
    creation_date = Column(DateTime, nullable=True)
    last_modified = Column(DateTime, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Content
    summary = Column(Text, nullable=True)
    full_text = Column(Text, nullable=True)  # Extracted text content
    embedding = Column(LargeBinary, nullable=True)  # Vector embedding for semantic search
    
    # Relationships
    tags = relationship('Tag', secondary=pdf_tags, back_populates='pdfs')
    chunks = relationship('PDFChunk', back_populates='pdf', cascade='all, delete-orphan')

class PDFChunk(Base):
    __tablename__ = 'pdf_chunks'

    id = Column(Integer, primary_key=True)
    pdf_id = Column(Integer, ForeignKey('pdfs.id'), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)  # Summary of the chunk content
    embedding = Column(LargeBinary, nullable=True)  # Vector embedding for semantic search
    page_number = Column(Integer, nullable=False)
    
    # Relationships
    pdf = relationship('PDF', back_populates='chunks')

class Tag(Base):
    __tablename__ = 'tags'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    
    # Relationships
    pdfs = relationship('PDF', secondary=pdf_tags, back_populates='tags') 