#!/usr/bin/env python3
"""
Knowledge Base Vectorization Script - Sentence Transformers Version
Processes source documents into ChromaDB vector database using sentence transformers

DEPENDENCIES:
    pip install PyPDF2 beautifulsoup4 markdown chromadb sentence-transformers

NO API KEYS REQUIRED - fully self-contained!

Usage:
    cd knowledge-base/
    python build-kb.py [--rebuild] [--test-mode]
    
Arguments:
    --rebuild: Force rebuild of existing vector DB
    --test-mode: Process only a subset of documents for testing
    --source-dir: Source documents directory (default: source-docs)
    --output-dir: Output vector database directory (default: vector-db)
    --model: Sentence transformer model (default: all-mpnet-base-v2)

Examples:
    # Test with subset of documents
    cd knowledge-base/
    python build-kb.py --test-mode
    
    # Full knowledge base build
    python build-kb.py
    
    # Force complete rebuild
    python build-kb.py --rebuild
    
    # Use different model
    python build-kb.py --model all-MiniLM-L6-v2
    
Cost: FREE! No API calls, fully offline after initial model download.
"""

import os
import sys
import json
import logging
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Document processing imports
import PyPDF2
from bs4 import BeautifulSoup
import markdown
import re

# Vector DB and embeddings
import chromadb
from chromadb.config import Settings

# Sentence transformers for local embeddings
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - KB-BUILD - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeBaseBuilder:
    """
    Builds Census expertise knowledge base from source documents.
    
    Uses sentence transformers for high-quality local embeddings with NO external API dependencies.
    Processes PDFs, HTML, markdown, and text files into a ChromaDB vector database.
    """
    
    def __init__(self, source_dir: Path, output_dir: Path, test_mode: bool = False, model_name: str = "all-mpnet-base-v2"):
        """Initialize the knowledge base builder."""
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.test_mode = test_mode
        self.model_name = model_name
        
        # Document processing stats
        self.stats = {
            'files_processed': 0,
            'chunks_created': 0,
            'embedding_batches': 0,
            'total_embeddings': 0,
            'errors': 0
        }
        
        # Initialize sentence transformer model
        self._init_embedding_model()
        
        # Initialize ChromaDB
        self._init_vector_db()
        
        logger.info(f"Knowledge base builder initialized")
        logger.info(f"Source directory: {self.source_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Embedding model: {self.model_name}")
        logger.info(f"Test mode: {self.test_mode}")
    
    def _init_embedding_model(self):
        """Initialize sentence transformer model for local embeddings."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            logger.info("Note: First run will download model (~90-420MB depending on model)")
            
            # Load sentence transformer model (downloads on first use)
            self.embedding_model = SentenceTransformer(self.model_name)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            
            logger.info(f"✅ Embedding model loaded successfully")
            logger.info(f"   Model: {self.model_name}")
            logger.info(f"   Dimensions: {self.embedding_dimension}")
            logger.info(f"   Max sequence length: {self.embedding_model.max_seq_length}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load sentence transformer model: {str(e)}")
            raise
    
    def _init_vector_db(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.output_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection
            collection_name = "census_knowledge"
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
            except Exception:  # ChromaDB raises various exceptions for missing collections
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={
                        "description": "Census expertise knowledge base",
                        "embedding_model": self.model_name,
                        "embedding_dimension": self.embedding_dimension
                    }
                )
                logger.info(f"Created new collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def build_knowledge_base(self, rebuild: bool = False):
        """Main entry point to build the knowledge base."""
        
        logger.info("=" * 60)
        logger.info("BUILDING CENSUS KNOWLEDGE BASE (SENTENCE TRANSFORMERS)")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Check if rebuild is needed
            if rebuild:
                logger.info("Rebuild requested - clearing existing collection")
                # Delete the collection and recreate it
                try:
                    self.chroma_client.delete_collection(self.collection.name)
                    logger.info("Deleted existing collection")
                except ValueError:
                    pass  # Collection doesn't exist
                except Exception as e:
                    logger.error(f"Failed to delete collection: {e}")
                    raise
                
                # Recreate the collection
                self.collection = self.chroma_client.create_collection(
                    name="census_knowledge",
                    metadata={
                        "description": "Census expertise knowledge base",
                        "embedding_model": self.model_name,
                        "embedding_dimension": self.embedding_dimension
                    }
                )
                logger.info("Created fresh collection for rebuild")
            
            existing_count = self.collection.count()
            if existing_count > 0 and not rebuild:
                logger.info(f"Collection already has {existing_count} documents")
                response = input("Continue building? (y/n): ")
                if response.lower() != 'y':
                    logger.info("Build cancelled")
                    return
            
            # Process all source documents
            self._process_source_documents()
            
            # Generate build manifest
            self._generate_build_manifest()
            
            # Display final statistics
            build_time = time.time() - start_time
            self._display_final_stats(build_time)
            
            logger.info("=" * 60)
            logger.info("KNOWLEDGE BASE BUILD COMPLETE")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Knowledge base build failed: {str(e)}")
            raise
    
    def _process_source_documents(self):
        """Process all source documents by category."""
        
        # Define document categories and their priorities
        categories = {
            'tidycensus-complete': {'priority': 1, 'max_files': 50 if self.test_mode else None},
            'tigris-complete': {'priority': 2, 'max_files': 20 if self.test_mode else None},
            'census-r-book': {'priority': 1, 'max_files': 30 if self.test_mode else None},
            'census-methodology': {'priority': 1, 'max_files': 10 if self.test_mode else None},
            'variable-definitions': {'priority': 1, 'max_files': 15 if self.test_mode else None},
            'equity-guidance': {'priority': 1, 'max_files': 15 if self.test_mode else None},
            'geographic-reference': {'priority': 2, 'max_files': 10 if self.test_mode else None},
            'data-privacy': {'priority': 2, 'max_files': 5 if self.test_mode else None},
            'training-best-practices': {'priority': 1, 'max_files': 10 if self.test_mode else None}
        }
        
        # Process each category
        for category, config in categories.items():
            category_path = self.source_dir / category
            if category_path.exists():
                logger.info(f"Processing category: {category}")
                self._process_category(category_path, category, config)
            else:
                logger.warning(f"Category directory not found: {category}")
    
    def _process_category(self, category_path: Path, category_name: str, config: Dict):
        """Process all files in a document category."""
        
        files_processed = 0
        max_files = config.get('max_files')
        
        # Get all files in category (recursively)
        all_files = []
        for pattern in ['*.pdf', '*.html', '*.md', '*.txt', '*.Rmd']:
            all_files.extend(category_path.rglob(pattern))
        
        # Sort by priority (smaller files first for testing)
        if self.test_mode:
            all_files.sort(key=lambda f: f.stat().st_size)
        
        for file_path in all_files:
            if max_files and files_processed >= max_files:
                logger.info(f"Reached max files limit for {category_name}: {max_files}")
                break
            
            try:
                # Skip hidden files and directories
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                
                # Skip very large files in test mode
                if self.test_mode and file_path.stat().st_size > 5 * 1024 * 1024:  # 5MB
                    logger.info(f"Skipping large file in test mode: {file_path.name}")
                    continue
                
                logger.info(f"Processing: {file_path.relative_to(self.source_dir)}")
                self._process_document(file_path, category_name)
                files_processed += 1
                
                # Small delay to prevent overheating
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                self.stats['errors'] += 1
                continue
        
        logger.info(f"Completed category {category_name}: {files_processed} files processed")
    
    def _process_document(self, file_path: Path, category: str):
        """Process a single document into chunks and embeddings."""
        
        try:
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                text = self._extract_pdf_text(file_path)
            elif file_path.suffix.lower() in ['.html', '.htm']:
                text = self._extract_html_text(file_path)
            elif file_path.suffix.lower() in ['.md', '.rmd']:
                text = self._extract_markdown_text(file_path)
            else:
                # Plain text
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            
            if not text or len(text.strip()) < 100:
                logger.warning(f"Insufficient text extracted from {file_path.name}")
                return
            
            # Create chunks
            chunks = self._create_chunks(text, file_path, category)
            
            if not chunks:
                logger.warning(f"No chunks created from {file_path.name}")
                return
            
            # Generate embeddings and store
            self._store_chunks(chunks)
            
            self.stats['files_processed'] += 1
            logger.info(f"Processed {file_path.name}: {len(chunks)} chunks created")
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:  # Guard against None
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            return text
        except Exception as e:
            logger.error(f"PDF extraction failed for {file_path}: {str(e)}")
            return ""
    
    def _extract_html_text(self, file_path: Path) -> str:
        """Extract text from HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"HTML extraction failed for {file_path}: {str(e)}")
            return ""
    
    def _extract_markdown_text(self, file_path: Path) -> str:
        """Extract text from Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                md_content = f.read()
            
            # Convert markdown to HTML then extract text
            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
        except Exception as e:
            logger.error(f"Markdown extraction failed for {file_path}: {str(e)}")
            return ""
    
    def _create_chunks(self, text: str, file_path: Path, category: str) -> List[Dict[str, Any]]:
        """Split document into chunks with aggressive token-aware sizing to preserve all content."""
        
        chunks = []
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) < 100:
            return chunks
        
        # Get actual model limits
        max_tokens = getattr(self.embedding_model, 'max_seq_length', 512)
        
        # Very conservative token estimation: 3 chars per token with 30% buffer
        max_chars = int(max_tokens * 3 * 0.7)
        chunk_size = min(200, max_chars // 3)  # Much smaller target chunks
        overlap = min(30, chunk_size // 10)
        
        logger.debug(f"Chunking {file_path.name}: max_tokens={max_tokens}, max_chars={max_chars}, chunk_size={chunk_size}")
        
        # For structured data files (like Census variables), use line-based chunking
        is_structured = any(indicator in file_path.name.lower()
                           for indicator in ['variables', 'api', 'definitions', 'zcta', 'rel'])
        
        if is_structured and len(text) > max_chars * 2:
            return self._chunk_structured_document(text, file_path, category, max_chars)
        
        # Standard paragraph-based chunking for regular documents
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_num = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding paragraph exceeds limits
            test_chunk = current_chunk + " " + paragraph if current_chunk else paragraph
            
            if len(test_chunk) > chunk_size and current_chunk:
                # Always save current chunk, force-splitting if needed
                if len(current_chunk) <= max_chars:
                    chunks.append(self._create_chunk_metadata(
                        current_chunk.strip(), file_path, category, chunk_num
                    ))
                    chunk_num += 1
                else:
                    # Force split oversized chunk
                    split_chunks = self._force_split_chunk(current_chunk, max_chars)
                    for split_chunk in split_chunks:
                        chunks.append(self._create_chunk_metadata(
                            split_chunk, file_path, category, chunk_num
                        ))
                        chunk_num += 1
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                current_chunk = overlap_text + " " + paragraph if overlap_text else paragraph
            else:
                current_chunk = test_chunk
            
            # Proactive split if chunk approaches limit
            if len(current_chunk) > max_chars * 0.9:  # Split at 90% of limit
                split_chunks = self._force_split_chunk(current_chunk, max_chars)
                for split_chunk in split_chunks:
                    chunks.append(self._create_chunk_metadata(
                        split_chunk, file_path, category, chunk_num
                    ))
                    chunk_num += 1
                current_chunk = ""
        
        # Add final chunk - always split if too large
        if current_chunk and len(current_chunk.strip()) > 100:
            if len(current_chunk) <= max_chars:
                chunks.append(self._create_chunk_metadata(
                    current_chunk.strip(), file_path, category, chunk_num
                ))
            else:
                # Force split final chunk
                split_chunks = self._force_split_chunk(current_chunk, max_chars)
                for split_chunk in split_chunks:
                    chunks.append(self._create_chunk_metadata(
                        split_chunk, file_path, category, chunk_num
                    ))
                    chunk_num += 1
        
        return chunks
    
    def _chunk_structured_document(self, text: str, file_path: Path, category: str, max_chars: int) -> List[Dict[str, Any]]:
        """Handle large structured documents by splitting on natural boundaries."""
        
        chunks = []
        chunk_num = 0
        
        # Try splitting on common structured boundaries
        split_patterns = [
            r'\n(?=[A-Z]\d{5}_\d{3})',  # Census variable codes
            r'\n(?=Table [A-Z]\d+)',    # Table definitions
            r'\n(?=\w+:)',              # Key-value pairs
            r'\n\n',                    # Paragraph breaks
            r'\n'                       # Line breaks (last resort)
        ]
        
        sections = [text]  # Start with full text
        
        # Try each split pattern until chunks are small enough
        for pattern in split_patterns:
            new_sections = []
            for section in sections:
                if len(section) <= max_chars:
                    new_sections.append(section)
                else:
                    # Split this section
                    parts = re.split(pattern, section)
                    new_sections.extend(parts)
            sections = new_sections
            
            # Check if we're small enough now
            if all(len(s) <= max_chars for s in sections):
                break
        
        # Create chunks from sections
        for section in sections:
            section = section.strip()
            if len(section) >= 100:  # Minimum chunk size
                if len(section) <= max_chars:
                    chunks.append(self._create_chunk_metadata(
                        section, file_path, category, chunk_num
                    ))
                    chunk_num += 1
                else:
                    # Force split oversized sections
                    split_chunks = self._force_split_chunk(section, max_chars)
                    for split_chunk in split_chunks:
                        chunks.append(self._create_chunk_metadata(
                            split_chunk, file_path, category, chunk_num
                        ))
                        chunk_num += 1
        
        return chunks
    
    def _force_split_chunk(self, text: str, max_chars: int) -> List[str]:
        """Aggressively split oversized chunks to preserve all content."""
        
        chunks = []
        
        # If text is still way too large, split by sentences first
        if len(text) > max_chars * 3:
            sentences = re.split(r'[.!?]+\s+', text)
            text_parts = []
            current_part = ""
            
            for sentence in sentences:
                test_part = current_part + " " + sentence if current_part else sentence
                if len(test_part) > max_chars * 2 and current_part:
                    text_parts.append(current_part.strip())
                    current_part = sentence
                else:
                    current_part = test_part
            
            if current_part:
                text_parts.append(current_part.strip())
        else:
            text_parts = [text]
        
        # Now split each part by words
        for part in text_parts:
            words = part.split()
            current_chunk = ""
            
            for word in words:
                test_chunk = current_chunk + " " + word if current_chunk else word
                
                if len(test_chunk) > max_chars and current_chunk:
                    if len(current_chunk.strip()) >= 50:  # Lower minimum for aggressive splitting
                        chunks.append(current_chunk.strip())
                    current_chunk = word
                else:
                    current_chunk = test_chunk
            
            if current_chunk and len(current_chunk.strip()) >= 50:
                chunks.append(current_chunk.strip())
        
        # Final safety - if any chunk is still too large, character-split it
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                final_chunks.append(chunk)
            else:
                # Last resort: character splitting with word boundaries
                while len(chunk) > max_chars:
                    # Find last space before limit
                    split_point = chunk.rfind(' ', 0, max_chars)
                    if split_point == -1:  # No space found, force split
                        split_point = max_chars
                    
                    final_chunks.append(chunk[:split_point].strip())
                    chunk = chunk[split_point:].strip()
                
                if chunk and len(chunk) >= 50:
                    final_chunks.append(chunk)
        
        return [chunk for chunk in final_chunks if len(chunk) >= 50]
    
    def _create_chunk_metadata(self, text: str, file_path: Path, category: str, chunk_num: int) -> Dict[str, Any]:
        """Create metadata for a text chunk."""
        
        # Generate unique ID with full hash for better traceability
        content_hash = hashlib.md5(text.encode()).hexdigest()
        chunk_id = f"{file_path.stem}_{chunk_num}_{content_hash}"
        
        return {
            'id': chunk_id,
            'text': text,
            'metadata': {
                'source_file': str(file_path.relative_to(self.source_dir)),
                'category': category,
                'chunk_number': chunk_num,
                'file_name': file_path.name,
                'file_type': file_path.suffix,
                'text_length': len(text)
            }
        }
    
    def _store_chunks(self, chunks: List[Dict[str, Any]]):
        """Generate embeddings and store chunks in ChromaDB using sentence transformers."""
        
        if not chunks:
            return
        
        try:
            # Prepare data for batch processing
            texts = [chunk['text'] for chunk in chunks]
            chunk_ids = [chunk['id'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            logger.info(f"Generating embeddings for {len(texts)} chunks using {self.model_name}...")
            
            # All chunks should be properly sized now, but add minimal safety check
            max_tokens = getattr(self.embedding_model, 'max_seq_length', 512)
            oversized_count = 0
            
            for text in texts:
                estimated_tokens = len(text) // 3  # Same conservative estimation
                if estimated_tokens > max_tokens:
                    oversized_count += 1
            
            if oversized_count > 0:
                logger.warning(f"Found {oversized_count} potentially oversized chunks - chunking logic needs adjustment")
            
            # Generate embeddings using sentence transformers (batch processing)
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=32,  # Process in batches for memory efficiency
                show_progress_bar=False,  # We have our own logging
                convert_to_numpy=True
            )
            
            # Convert to list format for ChromaDB
            embeddings_list = embeddings.tolist()
            
            # Store in ChromaDB
            self.collection.add(
                documents=texts,
                embeddings=embeddings_list,
                metadatas=metadatas,
                ids=chunk_ids
            )
            
            # Track usage
            self.stats['embedding_batches'] += 1
            self.stats['total_embeddings'] += len(texts)
            self.stats['chunks_created'] += len(texts)
            
            logger.info(f"✅ Stored {len(texts)} chunks in vector database")
            
        except Exception as e:
            logger.error(f"Error storing chunks: {str(e)}")
            raise
    
    def _generate_build_manifest(self):
        """Generate build manifest with statistics and metadata."""
        
        manifest = {
            'build_timestamp': time.time(),
            'build_date': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'test_mode': self.test_mode,
            'stats': self.stats,
            'collection_info': {
                'name': self.collection.name,
                'document_count': self.collection.count(),
            },
            'embedding_model': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'source_directory': str(self.source_dir),
            'output_directory': str(self.output_dir),
            'dependencies': 'sentence-transformers (local, no API key required)'
        }
        
        # Save manifest
        manifest_path = self.output_dir / 'build_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Build manifest saved: {manifest_path}")
    
    def _display_final_stats(self, build_time: float):
        """Display final build statistics."""
        
        logger.info("BUILD STATISTICS:")
        logger.info(f"  Files processed: {self.stats['files_processed']}")
        logger.info(f"  Chunks created: {self.stats['chunks_created']}")
        logger.info(f"  Embedding batches: {self.stats['embedding_batches']}")
        logger.info(f"  Total embeddings: {self.stats['total_embeddings']:,}")
        logger.info(f"  Model used: {self.model_name}")
        logger.info(f"  Embedding dimension: {self.embedding_dimension}")
        logger.info(f"  Build time: {build_time:.2f} seconds")
        logger.info(f"  Errors: {self.stats['errors']}")
        logger.info(f"  💰 Cost: FREE (no API calls)")
        
        final_count = self.collection.count()
        logger.info(f"  Final collection size: {final_count} documents")

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description='Build Census Knowledge Base with Sentence Transformers')
    parser.add_argument('--rebuild', action='store_true',
                       help='Force rebuild of existing vector DB')
    parser.add_argument('--test-mode', action='store_true',
                       help='Process only a subset of documents for testing')
    parser.add_argument('--source-dir', type=str, default='source-docs',
                       help='Source documents directory')
    parser.add_argument('--output-dir', type=str, default='vector-db',
                       help='Output vector database directory')
    parser.add_argument('--model', type=str, default='all-mpnet-base-v2',
                       help='Sentence transformer model (all-mpnet-base-v2, all-MiniLM-L6-v2, etc.)')
    
    args = parser.parse_args()
    
    # Validate source directory
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        sys.exit(1)
    
    # Build knowledge base
    try:
        builder = KnowledgeBaseBuilder(
            source_dir=source_dir,
            output_dir=Path(args.output_dir),
            test_mode=args.test_mode,
            model_name=args.model
        )
        
        builder.build_knowledge_base(rebuild=args.rebuild)
        
        logger.info("✅ Knowledge base build completed successfully!")
        logger.info("💰 No API costs - fully self-contained!")
        
    except KeyboardInterrupt:
        logger.info("Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Build failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
