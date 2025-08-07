"""
Advanced FAISS-based semantic search system for intelligent document retrieval
and clause matching optimized for insurance, legal, HR, and compliance domains.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import faiss
from sentence_transformers import SentenceTransformer
import torch
import gc
from dataclasses import dataclass
import logging
from src.utils.memory_manager import MemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""
    content: str
    chunk_id: str
    document_id: str
    document_type: str
    start_pos: int
    end_pos: int
    importance_score: float = 0.0
    metadata: Dict[str, Any] = None

@dataclass
class SearchResult:
    """Represents a search result with relevance information"""
    chunk: DocumentChunk
    similarity_score: float
    rank: int
    explanation: str = ""

class AdvancedFAISSSearch:
    """
    Advanced FAISS-based semantic search system optimized for HackRX requirements.
    Provides high-accuracy clause retrieval with explainable decision rationale.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.memory_manager = MemoryManager()
        self.model_name = model_name
        self.dimension = dimension
        
        # Initialize sentence transformer model (lightweight but effective)
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.encoder.max_seq_length = 512  # Optimize for speed
        
        # FAISS index for semantic search
        self.index = None
        self.document_chunks = []
        self.chunk_metadata = {}
        
        # Domain-specific optimization
        self.domain_keywords = {
            'insurance': ['policy', 'coverage', 'premium', 'claim', 'deductible', 'benefit', 'exclusion'],
            'legal': ['contract', 'agreement', 'clause', 'terms', 'conditions', 'liability'],
            'hr': ['employee', 'employment', 'benefits', 'compensation', 'leave', 'policy'],
            'compliance': ['regulation', 'requirement', 'standard', 'audit', 'compliance', 'violation']
        }
        
        # Initialize FAISS index
        self._initialize_faiss_index()
        
    def _initialize_faiss_index(self):
        """Initialize FAISS index with optimized settings"""
        try:
            # Use IndexFlatIP for cosine similarity (normalized vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"Initialized FAISS index with dimension {self.dimension}")
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {str(e)}")
            raise
    
    def smart_chunk_document(self, text: str, document_id: str, document_type: str = "unknown") -> List[DocumentChunk]:
        """
        Intelligently chunk document based on content structure and domain-specific patterns.
        Optimized for insurance, legal, HR, and compliance documents.
        """
        chunks = []
        
        # Clean and preprocess text
        text = self._preprocess_text(text)
        
        # Domain-aware chunking strategy
        if document_type.lower() in ['insurance', 'policy']:
            chunks = self._chunk_insurance_document(text, document_id, document_type)
        elif document_type.lower() in ['legal', 'contract']:
            chunks = self._chunk_legal_document(text, document_id, document_type)
        else:
            # Default semantic chunking
            chunks = self._semantic_chunk_document(text, document_id, document_type)
        
        # Calculate importance scores for each chunk
        for chunk in chunks:
            chunk.importance_score = self._calculate_importance_score(chunk.content, document_type)
        
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better chunking and embedding"""
        import re
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
        
        # Preserve important structural elements
        text = re.sub(r'(\d+\.)\s*', r'\n\1 ', text)  # Numbered lists
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', text)  # Sentence boundaries
        
        return text.strip()
    
    def _chunk_insurance_document(self, text: str, document_id: str, document_type: str) -> List[DocumentChunk]:
        """Specialized chunking for insurance documents"""
        import re
        chunks = []
        
        # Insurance-specific patterns
        section_patterns = [
            r'(coverage|benefit|exclusion|limitation|condition|term|definition)s?\s*[:.]',
            r'(grace period|waiting period|claim procedure|premium payment)',
            r'(policy|coverage)\s+(details|terms|conditions)',
            r'(section|article|clause)\s+\d+'
        ]
        
        # Split by insurance sections
        sections = self._split_by_patterns(text, section_patterns)
        
        chunk_id_counter = 0
        for section in sections:
            if len(section.strip()) > 50:  # Minimum chunk size
                # Further split long sections
                subsections = self._split_by_length(section, max_length=800, overlap=100)
                
                for subsection in subsections:
                    chunk = DocumentChunk(
                        content=subsection.strip(),
                        chunk_id=f"{document_id}_chunk_{chunk_id_counter}",
                        document_id=document_id,
                        document_type=document_type,
                        start_pos=text.find(subsection),
                        end_pos=text.find(subsection) + len(subsection),
                        metadata={'section_type': 'insurance_clause'}
                    )
                    chunks.append(chunk)
                    chunk_id_counter += 1
        
        return chunks
    
    def _chunk_legal_document(self, text: str, document_id: str, document_type: str) -> List[DocumentChunk]:
        """Specialized chunking for legal documents"""
        import re
        chunks = []
        
        # Legal-specific patterns
        section_patterns = [
            r'(section|article|clause|paragraph)\s+\d+',
            r'(whereas|therefore|notwithstanding|provided that)',
            r'(agreement|contract|terms|conditions|obligations)',
            r'(liability|indemnification|termination|breach)'
        ]
        
        sections = self._split_by_patterns(text, section_patterns)
        
        chunk_id_counter = 0
        for section in sections:
            if len(section.strip()) > 50:
                subsections = self._split_by_length(section, max_length=900, overlap=150)
                
                for subsection in subsections:
                    chunk = DocumentChunk(
                        content=subsection.strip(),
                        chunk_id=f"{document_id}_chunk_{chunk_id_counter}",
                        document_id=document_id,
                        document_type=document_type,
                        start_pos=text.find(subsection),
                        end_pos=text.find(subsection) + len(subsection),
                        metadata={'section_type': 'legal_clause'}
                    )
                    chunks.append(chunk)
                    chunk_id_counter += 1
        
        return chunks
    
    def _semantic_chunk_document(self, text: str, document_id: str, document_type: str) -> List[DocumentChunk]:
        """Default semantic chunking for general documents"""
        chunks = []
        
        # Split by paragraphs and sentences
        paragraphs = text.split('\n\n')
        
        chunk_id_counter = 0
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < 800:  # Optimal chunk size
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk.strip():
                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        chunk_id=f"{document_id}_chunk_{chunk_id_counter}",
                        document_id=document_id,
                        document_type=document_type,
                        start_pos=text.find(current_chunk),
                        end_pos=text.find(current_chunk) + len(current_chunk),
                        metadata={'section_type': 'general'}
                    )
                    chunks.append(chunk)
                    chunk_id_counter += 1
                
                current_chunk = paragraph + "\n\n"
        
        # Add remaining chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                chunk_id=f"{document_id}_chunk_{chunk_id_counter}",
                document_id=document_id,
                document_type=document_type,
                start_pos=text.find(current_chunk),
                end_pos=text.find(current_chunk) + len(current_chunk),
                metadata={'section_type': 'general'}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Split text by regex patterns"""
        import re
        
        # Combine all patterns
        combined_pattern = '|'.join(f'({pattern})' for pattern in patterns)
        
        # Split by patterns
        sections = re.split(combined_pattern, text, flags=re.IGNORECASE)
        
        # Filter out empty sections
        return [section.strip() for section in sections if section and section.strip()]
    
    def _split_by_length(self, text: str, max_length: int = 800, overlap: int = 100) -> List[str]:
        """Split text by length with overlap"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_length
            
            # Try to break at sentence boundary
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + max_length // 2:
                    end = sentence_end + 1
                else:
                    # Try word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end > start + max_length // 2:
                        end = word_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(start + 1, end - overlap)
            
            if start >= len(text):
                break
        
        return chunks
    
    def _calculate_importance_score(self, content: str, document_type: str) -> float:
        """Calculate importance score for a chunk based on content and domain"""
        score = 0.0
        content_lower = content.lower()
        
        # Domain-specific keyword scoring
        domain_keywords = self.domain_keywords.get(document_type.lower(), [])
        for keyword in domain_keywords:
            if keyword in content_lower:
                score += 0.1
        
        # General importance indicators
        importance_indicators = [
            'important', 'note', 'attention', 'warning', 'required', 'mandatory',
            'shall', 'must', 'will', 'coverage', 'benefit', 'exclusion',
            'condition', 'term', 'definition', 'procedure', 'process'
        ]
        
        for indicator in importance_indicators:
            if indicator in content_lower:
                score += 0.05
        
        # Length-based scoring (moderate length chunks are often more informative)
        length_score = min(1.0, len(content) / 500)  # Normalize by 500 chars
        score += length_score * 0.2
        
        # Numerical information bonus (often contains specific details)
        import re
        numbers = re.findall(r'\d+', content)
        if numbers:
            score += min(0.3, len(numbers) * 0.1)
        
        return min(1.0, score)  # Cap at 1.0
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents in FAISS for semantic search"""
        logger.info(f"Indexing {len(documents)} documents...")
        
        all_chunks = []
        
        # Process each document
        for doc in documents:
            document_id = doc.get('id', f"doc_{len(all_chunks)}")
            document_type = doc.get('type', 'unknown')
            content = doc.get('content', '')
            
            # Smart chunking
            chunks = self.smart_chunk_document(content, document_id, document_type)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("No chunks to index")
            return
        
        # Generate embeddings in batches for memory efficiency
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]
            
            # Generate embeddings
            batch_embeddings = self.encoder.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,  # For cosine similarity
                show_progress_bar=False
            )
            
            embeddings.extend(batch_embeddings)
            
            # Memory cleanup
            if i % (batch_size * 4) == 0:
                gc.collect()
        
        # Convert to numpy array
        embeddings_matrix = np.array(embeddings, dtype=np.float32)
        
        # Add to FAISS index
        self.index.add(embeddings_matrix)
        
        # Store chunks and metadata
        self.document_chunks = all_chunks
        for i, chunk in enumerate(all_chunks):
            self.chunk_metadata[i] = chunk
        
        logger.info(f"Successfully indexed {len(all_chunks)} chunks")
        
        # Memory cleanup
        del embeddings, embeddings_matrix
        gc.collect()
    
    def semantic_search(self, query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[SearchResult]:
        """
        Perform semantic search with explainable results.
        Returns top-k most relevant chunks with similarity scores and explanations.
        """
        if self.index is None or len(self.document_chunks) == 0:
            logger.warning("No documents indexed for search")
            return []
        
        try:
            # Encode query
            query_embedding = self.encoder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Search in FAISS index
            similarities, indices = self.index.search(query_embedding, min(top_k * 2, len(self.document_chunks)))
            
            # Process results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity >= min_similarity and idx < len(self.document_chunks):
                    chunk = self.document_chunks[idx]
                    
                    # Generate explanation
                    explanation = self._generate_search_explanation(query, chunk, similarity)
                    
                    result = SearchResult(
                        chunk=chunk,
                        similarity_score=float(similarity),
                        rank=i + 1,
                        explanation=explanation
                    )
                    results.append(result)
            
            # Sort by combined score (similarity + importance)
            results.sort(key=lambda x: x.similarity_score * (1 + x.chunk.importance_score), reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    def _generate_search_explanation(self, query: str, chunk: DocumentChunk, similarity: float) -> str:
        """Generate explanation for why this chunk was retrieved"""
        explanation_parts = []
        
        # Similarity explanation
        if similarity > 0.8:
            explanation_parts.append("High semantic similarity")
        elif similarity > 0.6:
            explanation_parts.append("Good semantic similarity")
        else:
            explanation_parts.append("Moderate semantic similarity")
        
        # Importance explanation
        if chunk.importance_score > 0.7:
            explanation_parts.append("high importance content")
        elif chunk.importance_score > 0.4:
            explanation_parts.append("moderate importance content")
        
        # Content type explanation
        if chunk.metadata and 'section_type' in chunk.metadata:
            section_type = chunk.metadata['section_type']
            explanation_parts.append(f"from {section_type}")
        
        return f"Retrieved due to {', '.join(explanation_parts)} (similarity: {similarity:.3f})"
    
    def find_relevant_clauses(self, query: str, document_text: str) -> List[Dict[str, Any]]:
        """
        Find relevant clauses in a document for a specific query.
        Optimized for real-time processing of new documents.
        """
        # Quick indexing for single document
        temp_chunks = self.smart_chunk_document(document_text, "temp_doc", "insurance")
        
        if not temp_chunks:
            return []
        
        # Generate embeddings for chunks
        chunk_texts = [chunk.content for chunk in temp_chunks]
        chunk_embeddings = self.encoder.encode(
            chunk_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Encode query
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Calculate similarities
        similarities = np.dot(query_embedding, chunk_embeddings.T)[0]
        
        # Get top relevant chunks
        top_indices = np.argsort(similarities)[::-1][:3]  # Top 3 chunks
        
        relevant_clauses = []
        for i, idx in enumerate(top_indices):
            if similarities[idx] > 0.2:  # Minimum relevance threshold
                chunk = temp_chunks[idx]
                clause_info = {
                    'content': chunk.content,
                    'similarity_score': float(similarities[idx]),
                    'importance_score': chunk.importance_score,
                    'position': f"Position {chunk.start_pos}-{chunk.end_pos}",
                    'explanation': self._generate_search_explanation(query, chunk, similarities[idx])
                }
                relevant_clauses.append(clause_info)
        
        return relevant_clauses
    
    def save_index(self, filepath: str):
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save metadata
            metadata = {
                'document_chunks': self.document_chunks,
                'chunk_metadata': self.chunk_metadata,
                'model_name': self.model_name,
                'dimension': self.dimension
            }
            
            with open(f"{filepath}_metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"FAISS index saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
    
    def load_index(self, filepath: str) -> bool:
        """Load FAISS index and metadata from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load metadata
            with open(f"{filepath}_metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            self.document_chunks = metadata['document_chunks']
            self.chunk_metadata = metadata['chunk_metadata']
            
            logger.info(f"FAISS index loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            return False
