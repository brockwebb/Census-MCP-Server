"""
Vector Database and Knowledge Base for Census MCP Server

Handles RAG (Retrieval-Augmented Generation) functionality using:
- Sentence transformers for high-quality local embeddings (NO API key required)
- ChromaDB for vector storage and similarity search
- R documentation corpus processing and indexing
- Context enrichment for Census queries
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import hashlib

# Vector DB and embedding imports
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available. Install with: pip install sentence-transformers")

from utils.config import get_config

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    Vector database and knowledge management for Census expertise.
    
    Provides:
    - Document indexing and similarity search using sentence transformers (local, no API key)
    - Variable context lookup and enrichment
    - Location parsing and geographic knowledge
    - R documentation corpus integration
    """
    
    def __init__(self, corpus_path: Optional[Path] = None, vector_db_path: Optional[Path] = None):
        """
        Initialize knowledge base.
        
        Args:
            corpus_path: Path to R documentation corpus
            vector_db_path: Path to vector database storage
        """
        self.config = get_config()
        self.corpus_path = corpus_path or self.config.r_docs_corpus_path
        self.vector_db_path = vector_db_path or self.config.vector_db_path
        
        # Initialize components
        self._init_embedding_model()
        self._init_vector_db()
        self._init_variable_contexts()
        self._init_location_knowledge()
        
        # Load or build corpus index
        self._ensure_corpus_indexed()
        
        logger.info("Knowledge Base initialized successfully")
    
    def _init_embedding_model(self):
        """Initialize sentence transformer for local embeddings."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("SentenceTransformers not available. RAG functionality disabled.")
            self.embedding_model = None
            self.model_name = None
            return
        
        try:
            # Use the model specified in config, with fallback
            model_name = getattr(self.config, 'embedding_model', 'all-mpnet-base-v2')
            
            # Handle old OpenAI model names in config
            if 'text-embedding' in model_name:
                logger.warning(f"OpenAI model '{model_name}' detected in config, switching to sentence transformers")
                model_name = 'all-mpnet-base-v2'
            
            logger.info(f"Loading sentence transformer model: {model_name}")
            
            # Load sentence transformer model (downloads on first use)
            self.embedding_model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            
            logger.info(f"✅ Sentence transformer model loaded successfully")
            logger.info(f"   Model: {model_name}")
            logger.info(f"   Dimensions: {self.embedding_dimension}")
            logger.info(f"   Max sequence length: {self.embedding_model.max_seq_length}")
            
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {str(e)}")
            self.embedding_model = None
            self.model_name = None
    
    def _init_vector_db(self):
        """Initialize ChromaDB for vector storage."""
        if not CHROMADB_AVAILABLE:
            logger.error("ChromaDB not available. Vector search disabled.")
            self.vector_db = None
            self.collection = None
            return
        
        try:
            # Create vector DB directory
            self.vector_db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.vector_db = chromadb.PersistentClient(
                path=str(self.vector_db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            collection_name = "census_knowledge"
            try:
                self.collection = self.vector_db.get_collection(collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
                
                # Check if collection metadata matches current model
                metadata = self.collection.metadata or {}
                stored_model = metadata.get('embedding_model', 'unknown')
                if self.model_name and stored_model != self.model_name:
                    logger.warning(f"Vector DB was built with '{stored_model}', current model is '{self.model_name}'")
                    logger.warning("Consider rebuilding vector DB for optimal performance")
                
            except:
                self.collection = self.vector_db.create_collection(
                    name=collection_name,
                    metadata={
                        "description": "Census R documentation and knowledge corpus",
                        "embedding_model": self.model_name or "unknown",
                        "embedding_dimension": getattr(self, 'embedding_dimension', 0)
                    }
                )
                logger.info(f"Created new collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {str(e)}")
            self.vector_db = None
            self.collection = None
    
    def _init_variable_contexts(self):
        """Initialize variable context knowledge."""
        # Enhanced variable contexts with definitions and usage notes
        self.variable_contexts = {
            'population': {
                'label': 'Total Population',
                'definition': 'Total count of people residing in the area',
                'variable_code': 'B01003_001',
                'table': 'B01003',
                'universe': 'Total population',
                'notes': 'Includes all residents regardless of citizenship status'
            },
            'median_income': {
                'label': 'Median Household Income',
                'definition': 'Median income of households in the past 12 months (in inflation-adjusted dollars)',
                'variable_code': 'B19013_001',
                'table': 'B19013',
                'universe': 'Households',
                'notes': 'Inflation-adjusted to survey year dollars. Excludes group quarters population.'
            },
            'poverty_rate': {
                'label': 'Population Below Poverty Level',
                'definition': 'Number of people with income below the federal poverty threshold',
                'variable_code': 'B17001_002',
                'table': 'B17001',
                'universe': 'Population for whom poverty status is determined',
                'notes': 'Excludes people in institutions, military group quarters, and college dormitories'
            },
            'median_home_value': {
                'label': 'Median Home Value',
                'definition': 'Median value of owner-occupied housing units',
                'variable_code': 'B25077_001',
                'table': 'B25077',
                'universe': 'Owner-occupied housing units',
                'notes': 'Self-reported by homeowners. May not reflect current market values.'
            },
            'unemployment_rate': {
                'label': 'Unemployment Rate',
                'definition': 'Percentage of labor force that is unemployed',
                'variable_code': 'B23025_005',
                'table': 'B23025',
                'universe': 'Population 16 years and over',
                'notes': 'Calculated as unemployed / (employed + unemployed) * 100'
            }
        }
    
    def _init_location_knowledge(self):
        """Initialize geographic knowledge and patterns."""
        # Common location patterns and disambiguation
        self.location_patterns = {
            'major_cities': {
                'new york': {'state': 'NY', 'type': 'place', 'full_name': 'New York city, New York'},
                'los angeles': {'state': 'CA', 'type': 'place', 'full_name': 'Los Angeles city, California'},
                'chicago': {'state': 'IL', 'type': 'place', 'full_name': 'Chicago city, Illinois'},
                'houston': {'state': 'TX', 'type': 'place', 'full_name': 'Houston city, Texas'},
                'philadelphia': {'state': 'PA', 'type': 'place', 'full_name': 'Philadelphia city, Pennsylvania'},
                'phoenix': {'state': 'AZ', 'type': 'place', 'full_name': 'Phoenix city, Arizona'},
                'san antonio': {'state': 'TX', 'type': 'place', 'full_name': 'San Antonio city, Texas'},
                'san diego': {'state': 'CA', 'type': 'place', 'full_name': 'San Diego city, California'},
                'dallas': {'state': 'TX', 'type': 'place', 'full_name': 'Dallas city, Texas'},
                'san jose': {'state': 'CA', 'type': 'place', 'full_name': 'San Jose city, California'},
                'baltimore': {'state': 'MD', 'type': 'place', 'full_name': 'Baltimore city, Maryland'},
            },
            'ambiguous_names': {
                'springfield': ['IL', 'MA', 'MO', 'OH'],  # Multiple Springfields
                'washington': ['DC', 'WA'],  # Washington DC vs Washington State
                'baltimore': ['city', 'county'],  # Baltimore city vs Baltimore County
            }
        }
    
    def _ensure_corpus_indexed(self):
        """Check if corpus is already indexed in vector DB."""
        if not self.embedding_model or not self.collection:
            logger.warning("Cannot check corpus: sentence transformer or vector DB not available")
            return
        
        try:
            # Check if corpus is already indexed
            count = self.collection.count()
            if count > 0:
                logger.info(f"Corpus already indexed: {count} documents")
                return
            else:
                logger.warning("No documents found in knowledge base. Run build-kb.py to create vector database.")
                
        except Exception as e:
            logger.error(f"Error checking corpus index: {str(e)}")
    
    async def get_variable_context(self, variables: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get enhanced context for variables using both static knowledge and RAG.
        
        Args:
            variables: List of variable names
            
        Returns:
            Dictionary mapping variables to context information
        """
        context = {}
        
        for var in variables:
            var_lower = var.lower().strip()
            
            # Start with static context
            var_context = self.variable_contexts.get(var_lower, {
                'label': var.title(),
                'definition': f"Census variable: {var}",
                'notes': 'Variable definition not available in local knowledge base'
            })
            
            # Enhance with RAG if available
            if self.embedding_model and self.collection:
                try:
                    rag_context = await self._search_variable_documentation(var)
                    if rag_context:
                        var_context.update(rag_context)
                except Exception as e:
                    logger.warning(f"RAG search failed for variable {var}: {str(e)}")
            
            context[var] = var_context
        
        return context
    
    async def _search_variable_documentation(self, variable: str) -> Optional[Dict[str, Any]]:
        """Search documentation for variable-specific context."""
        try:
            query = f"Census variable {variable} definition methodology"
            results = await self.search_documentation(query, context="variable_lookup")
            
            if results:
                # Extract relevant information from search results
                best_result = results[0]
                return {
                    'documentation': best_result.get('content', ''),
                    'source': best_result.get('source', ''),
                    'rag_enhanced': True
                }
        except Exception as e:
            logger.warning(f"Variable documentation search failed: {str(e)}")
        
        return None
    
    async def parse_location(self, location: str) -> Dict[str, Any]:
        """
        Parse and enhance location information.
        
        Args:
            location: Location string
            
        Returns:
            Enhanced location information with geographic context
        """
        location_lower = location.lower().strip()
        
        # Check major cities
        if location_lower in self.location_patterns['major_cities']:
            city_info = self.location_patterns['major_cities'][location_lower]
            return {
                'original': location,
                'normalized': city_info['full_name'],
                'state': city_info['state'],
                'type': city_info['type'],
                'confidence': 'high'
            }
        
        # Check for ambiguous names
        for ambiguous, states in self.location_patterns['ambiguous_names'].items():
            if ambiguous in location_lower:
                return {
                    'original': location,
                    'ambiguous': True,
                    'possible_states': states,
                    'recommendation': f"Specify state for {ambiguous} (e.g., '{ambiguous}, {states[0]}')",
                    'confidence': 'low'
                }
        
        # Default parsing
        return {
            'original': location,
            'normalized': location,
            'confidence': 'medium'
        }
    
    async def search_documentation(self, query: str, context: str = "",
                                 top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search documentation using vector similarity with sentence transformers.
        
        Args:
            query: Search query
            context: Additional context for the search
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.embedding_model or not self.collection:
            logger.warning("Cannot search: sentence transformer or vector DB not available")
            return []
        
        try:
            # Prepare search query
            search_text = f"{query} {context}".strip()
            top_k = top_k or self.config.vector_search_top_k
            
            # Generate query embedding using sentence transformers
            query_embedding = self.embedding_model.encode([search_text])
            
            # Search vector database
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'source': results['metadatas'][0][i].get('source', 'Unknown'),
                    'title': results['metadatas'][0][i].get('title', 'Untitled')
                })
            
            # Filter by threshold
            threshold = self.config.vector_search_threshold
            filtered_results = [r for r in formatted_results if r['score'] >= threshold]
            
            logger.info(f"Documentation search: {len(filtered_results)} relevant results from {len(results['documents'][0])} total")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Documentation search failed: {str(e)}")
            return []
    
    async def add_document(self, title: str, content: str, metadata: Optional[Dict] = None):
        """Add a document to the knowledge base using sentence transformers."""
        if not self.embedding_model or not self.collection:
            logger.warning("Cannot add document: sentence transformer or vector DB not available")
            return
        
        try:
            # Generate unique ID
            doc_id = hashlib.md5(f"{title}_{content[:100]}".encode()).hexdigest()
            
            # Generate embedding using sentence transformers
            embedding = self.embedding_model.encode([content])
            
            # Prepare metadata
            doc_metadata = metadata or {}
            doc_metadata.update({
                'title': title,
                'source': 'user_added',
                'timestamp': str(asyncio.get_event_loop().time())
            })
            
            # Add to collection
            self.collection.add(
                documents=[content],
                metadatas=[doc_metadata],
                embeddings=embedding.tolist(),
                ids=[doc_id]
            )
            
            logger.info(f"Added document: {title}")
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        stats = {
            'embedding_model': self.model_name or 'Not available',
            'embedding_dimension': getattr(self, 'embedding_dimension', 0),
            'vector_db_available': self.collection is not None,
            'corpus_path': str(self.corpus_path),
            'variable_contexts': len(self.variable_contexts),
            'location_patterns': len(self.location_patterns['major_cities']),
            'dependencies': 'sentence-transformers (local, no API key required)'
        }
        
        if self.collection:
            try:
                stats['document_count'] = self.collection.count()
                metadata = self.collection.metadata or {}
                stats['collection_metadata'] = metadata
            except:
                stats['document_count'] = 'Unknown'
        
        return stats
