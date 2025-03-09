"""
Vector Store Module

This module provides vector storage capabilities for semantic search and retrieval.
It allows for efficient storage and querying of embeddings derived from text,
supporting similarity-based information retrieval across the agent system.
"""

import logging
import os
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import shutil

# Try to import different vector database libraries
try:
    from langchain_community.vectorstores import Chroma
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OPENAI_EMBEDDINGS_AVAILABLE = False

try:
    from langchain.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    HUGGINGFACE_EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector store for efficient storage and retrieval of embeddings.
    
    This class provides vector storage capabilities for semantic search
    and similarity-based retrieval of information. It supports multiple
    embedding models and vector database backends.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vector store.
        
        Args:
            config: Configuration dictionary with vector store settings
        """
        self.config = config
        self.data_dir = config.get("data_dir", "data")
        self.vector_dir = config.get("vector_dir", "vector_store")
        self.embedding_model = config.get("embedding_model", "text-embedding-3-small")
        self.embedding_dimension = config.get("embedding_dimension", 1536)  # Default for OpenAI embeddings
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.vector_db_type = config.get("vector_db_type", "chroma")
        
        # Ensure data directory exists
        self.vector_store_path = os.path.join(self.data_dir, self.vector_dir)
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Initialize embedding function and vector store
        self.embedding_function = self._initialize_embedding_function()
        self.vector_store = self._initialize_vector_store()
        
        logger.debug(f"Initialized VectorStore with model: {self.embedding_model}")
    
    def _initialize_embedding_function(self):
        """
        Initialize the embedding function based on configuration.
        
        Returns:
            Initialized embedding function
        """
        embedding_function = None
        
        # Try to initialize OpenAI embeddings
        if "openai" in self.embedding_model.lower() and OPENAI_EMBEDDINGS_AVAILABLE:
            try:
                embedding_function = OpenAIEmbeddings(model=self.embedding_model)
                logger.info(f"Initialized OpenAI embeddings: {self.embedding_model}")
            except Exception as e:
                logger.error(f"Error initializing OpenAI embeddings: {str(e)}")
        
        # Try to initialize HuggingFace embeddings if OpenAI embeddings are not available
        elif HUGGINGFACE_EMBEDDINGS_AVAILABLE:
            try:
                # Default to a common model if specific model not specified
                model_name = self.embedding_model
                if "openai" in model_name.lower():
                    model_name = "sentence-transformers/all-mpnet-base-v2"
                
                embedding_function = HuggingFaceEmbeddings(model_name=model_name)
                logger.info(f"Initialized HuggingFace embeddings: {model_name}")
            except Exception as e:
                logger.error(f"Error initializing HuggingFace embeddings: {str(e)}")
        
        # Fallback to a simple embedding function if all else fails
        if embedding_function is None:
            logger.warning("No embedding libraries available, using simple fallback embeddings")
            embedding_function = SimpleFallbackEmbeddings(dim=self.embedding_dimension)
        
        return embedding_function
    
    def _initialize_vector_store(self):
        """
        Initialize the vector store based on configuration.
        
        Returns:
            Initialized vector store
        """
        vector_store = None
        
        # Check if Chroma is available and configured
        if self.vector_db_type.lower() == "chroma" and CHROMA_AVAILABLE:
            try:
                # Check if there's an existing Chroma database
                chroma_dir = os.path.join(self.vector_store_path, "chroma")
                if os.path.exists(chroma_dir) and os.path.isdir(chroma_dir):
                    logger.info(f"Loading existing Chroma vector store from: {chroma_dir}")
                    vector_store = Chroma(
                        persist_directory=chroma_dir,
                        embedding_function=self.embedding_function
                    )
                else:
                    logger.info(f"Creating new Chroma vector store at: {chroma_dir}")
                    vector_store = Chroma(
                        persist_directory=chroma_dir,
                        embedding_function=self.embedding_function
                    )
            except Exception as e:
                logger.error(f"Error initializing Chroma vector store: {str(e)}")
        
        # Fallback to a simple vector store if needed
        if vector_store is None:
            logger.warning("No vector database libraries available, using simple fallback vector store")
            vector_store = SimpleFallbackVectorStore(
                data_dir=self.vector_store_path,
                embedding_function=self.embedding_function,
                similarity_threshold=self.similarity_threshold
            )
        
        return vector_store
    
    def add_texts(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dictionaries, one per text
            ids: Optional list of IDs for the texts
            
        Returns:
            List of IDs for the added texts
        """
        try:
            # Ensure metadatas is provided for each text
            if metadatas is None:
                metadatas = [{} for _ in texts]
            
            # Add texts to the vector store
            result_ids = self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            
            # Persist the vector store if it supports it
            if hasattr(self.vector_store, "persist"):
                self.vector_store.persist()
            
            logger.info(f"Added {len(texts)} texts to vector store")
            return result_ids
            
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {str(e)}")
            return []
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The query text
            k: Number of results to return
            filter: Optional filter criteria
            fetch_k: Optional number of documents to consider before filtering
            
        Returns:
            List of similar documents with content and metadata
        """
        try:
            # Perform similarity search
            if hasattr(self.vector_store, "similarity_search_with_score"):
                docs_and_scores = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter,
                    fetch_k=fetch_k
                )
                
                # Format results
                results = []
                for doc, score in docs_and_scores:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score
                    })
                
                return results
            else:
                # Fallback to regular similarity search
                docs = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter,
                    fetch_k=fetch_k
                )
                
                # Format results without scores
                results = []
                for doc in docs:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": None
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []
    
    def max_marginal_relevance_search(
        self, 
        query: str, 
        k: int = 4, 
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search with maximal marginal relevance to balance relevance and diversity.
        
        Args:
            query: The query text
            k: Number of results to return
            fetch_k: Number of documents to consider before filtering for diversity
            lambda_mult: Diversity vs relevance balance factor (0 to 1)
            filter: Optional filter criteria
            
        Returns:
            List of documents balancing relevance and diversity
        """
        try:
            # Check if the vector store supports MMR search
            if hasattr(self.vector_store, "max_marginal_relevance_search"):
                docs = self.vector_store.max_marginal_relevance_search(
                    query=query,
                    k=k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                    filter=filter
                )
                
                # Format results
                results = []
                for doc in docs:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": None
                    })
                
                return results
            else:
                # Fall back to regular similarity search
                logger.warning("Vector store does not support MMR search, falling back to regular search")
                return self.similarity_search(query=query, k=k, filter=filter)
                
        except Exception as e:
            logger.error(f"Error performing MMR search: {str(e)}")
            return []
    
    def delete(self, ids: List[str]) -> bool:
        """
        Delete documents from the vector store by ID.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if the vector store supports deletion
            if hasattr(self.vector_store, "delete"):
                self.vector_store.delete(ids)
                
                # Persist changes if supported
                if hasattr(self.vector_store, "persist"):
                    self.vector_store.persist()
                
                logger.info(f"Deleted {len(ids)} documents from vector store")
                return True
            else:
                logger.warning("Vector store does not support deletion")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting documents from vector store: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all documents from the vector store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Handle different vector store types
            if self.vector_db_type.lower() == "chroma" and CHROMA_AVAILABLE:
                # For Chroma, we can use the collection's delete method
                if hasattr(self.vector_store, "_collection"):
                    self.vector_store._collection.delete(where={})
                    
                    # Persist changes
                    if hasattr(self.vector_store, "persist"):
                        self.vector_store.persist()
                    
                    logger.info("Cleared all documents from Chroma vector store")
                    return True
                else:
                    # Try to recreate the vector store
                    chroma_dir = os.path.join(self.vector_store_path, "chroma")
                    if os.path.exists(chroma_dir):
                        shutil.rmtree(chroma_dir)
                    
                    self.vector_store = Chroma(
                        persist_directory=chroma_dir,
                        embedding_function=self.embedding_function
                    )
                    
                    logger.info("Recreated Chroma vector store")
                    return True
            
            # For the fallback vector store
            elif isinstance(self.vector_store, SimpleFallbackVectorStore):
                self.vector_store.clear()
                logger.info("Cleared all documents from fallback vector store")
                return True
            
            logger.warning("Vector store clearing not supported for this type")
            return False
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False
    
    def get_retriever(self, **kwargs):
        """
        Get a retriever interface to the vector store.
        
        Args:
            **kwargs: Additional parameters for the retriever
            
        Returns:
            Retriever object or None if not supported
        """
        try:
            # Check if the vector store supports creating a retriever
            if hasattr(self.vector_store, "as_retriever"):
                return self.vector_store.as_retriever(**kwargs)
            else:
                logger.warning("Vector store does not support retriever interface")
                return None
                
        except Exception as e:
            logger.error(f"Error creating retriever: {str(e)}")
            return None
    
    def count(self) -> int:
        """
        Count the number of documents in the vector store.
        
        Returns:
            Number of documents
        """
        try:
            # Try different methods to get the count
            if hasattr(self.vector_store, "count"):
                return self.vector_store.count()
            elif hasattr(self.vector_store, "_collection") and hasattr(self.vector_store._collection, "count"):
                return self.vector_store._collection.count()
            elif isinstance(self.vector_store, SimpleFallbackVectorStore):
                return self.vector_store.count()
            else:
                logger.warning("Unable to count documents in vector store")
                return -1
                
        except Exception as e:
            logger.error(f"Error counting documents in vector store: {str(e)}")
            return -1


class SimpleFallbackEmbeddings:
    """
    A simple fallback embedding function when no proper embedding libraries are available.
    This is not meant for production use and provides only basic functionality.
    """
    
    def __init__(self, dim: int = 1536):
        """
        Initialize the fallback embeddings.
        
        Args:
            dim: Dimension of the embeddings
        """
        self.dim = dim
        logger.warning(f"Using SimpleFallbackEmbeddings with dimension {dim}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings as float lists
        """
        embeddings = []
        for text in texts:
            # Create a deterministic but simple embedding based on the text
            # This is NOT a good embedding strategy but works as a fallback
            embedding = self._simple_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Create an embedding for a query string.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding as a list of floats
        """
        return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str) -> List[float]:
        """
        Create a simple deterministic embedding from text.
        
        Args:
            text: Text to embed
            
        Returns:
            Simple embedding vector
        """
        # Use a hash of the text to seed a random number generator
        import hashlib
        import random
        
        # Get deterministic seed from text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash, 16) % (2**32)
        
        # Create a pseudo-random embedding
        random.seed(seed)
        embedding = [random.uniform(-1, 1) for _ in range(self.dim)]
        
        # Normalize the embedding
        norm = sum(x**2 for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding


class SimpleFallbackVectorStore:
    """
    A simple fallback vector store when no proper vector database libraries are available.
    This is not meant for production use and provides only basic functionality.
    """
    
    def __init__(
        self, 
        data_dir: str, 
        embedding_function,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize the fallback vector store.
        
        Args:
            data_dir: Directory to store data
            embedding_function: Function to create embeddings
            similarity_threshold: Threshold for similarity searches
        """
        self.data_dir = data_dir
        self.embedding_function = embedding_function
        self.similarity_threshold = similarity_threshold
        self.vectors = []  # List of (id, embedding, text, metadata) tuples
        self.next_id = 1
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Try to load existing data
        self._load()
        
        logger.warning(f"Using SimpleFallbackVectorStore in {data_dir}")
    
    def add_texts(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs
            
        Returns:
            List of IDs for the added texts
        """
        # Ensure metadatas exists for each text
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Ensure IDs exist for each text
        if ids is None:
            ids = [str(self.next_id + i) for i in range(len(texts))]
            self.next_id += len(texts)
        
        # Get embeddings for texts
        embeddings = self.embedding_function.embed_documents(texts)
        
        # Add to vectors
        for i, (text, embedding, metadata, id) in enumerate(zip(texts, embeddings, metadatas, ids)):
            self.vectors.append((id, embedding, text, metadata))
        
        # Save data
        self._save()
        
        return ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The query text
            k: Number of results to return
            filter: Optional filter criteria
            fetch_k: Ignored in this implementation
            
        Returns:
            List of similar documents
        """
        from langchain_core.documents import Document
        
        # Get query embedding
        query_embedding = self.embedding_function.embed_query(query)
        
        # Calculate similarities
        similarities = []
        for id, embedding, text, metadata in self.vectors:
            # Apply filter if provided
            if filter and not self._matches_filter(metadata, filter):
                continue
                
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((id, similarity, text, metadata))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k results
        top_k = similarities[:k]
        
        # Convert to documents
        documents = []
        for id, similarity, text, metadata in top_k:
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        
        return documents
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: Optional[int] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for documents similar to the query, with similarity scores.
        
        Args:
            query: The query text
            k: Number of results to return
            filter: Optional filter criteria
            fetch_k: Ignored in this implementation
            
        Returns:
            List of (document, score) tuples
        """
        from langchain_core.documents import Document
        
        # Get query embedding
        query_embedding = self.embedding_function.embed_query(query)
        
        # Calculate similarities
        similarities = []
        for id, embedding, text, metadata in self.vectors:
            # Apply filter if provided
            if filter and not self._matches_filter(metadata, filter):
                continue
                
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((id, similarity, text, metadata))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k results
        top_k = similarities[:k]
        
        # Convert to documents with scores
        documents_with_scores = []
        for id, similarity, text, metadata in top_k:
            doc = Document(page_content=text, metadata=metadata)
            documents_with_scores.append((doc, similarity))
        
        return documents_with_scores
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity
        """
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0
            
        return dot_product / (norm_a * norm_b)
    
    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """
        Check if metadata matches filter criteria.
        
        Args:
            metadata: Metadata to check
            filter: Filter criteria
            
        Returns:
            True if metadata matches filter, False otherwise
        """
        for key, value in filter.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def count(self) -> int:
        """
        Count the number of documents in the vector store.
        
        Returns:
            Number of documents
        """
        return len(self.vectors)
    
    def clear(self) -> bool:
        """
        Clear all documents from the vector store.
        
        Returns:
            True if successful
        """
        self.vectors = []
        self._save()
        return True
    
    def _save(self):
        """Save the vector store data to disk."""
        data_path = os.path.join(self.data_dir, "fallback_vectors.json")
        
        # Convert embeddings to lists for JSON serialization
        serializable_vectors = []
        for id, embedding, text, metadata in self.vectors:
            serializable_vectors.append({
                "id": id,
                "embedding": list(embedding),
                "text": text,
                "metadata": metadata
            })
        
        # Save to file
        with open(data_path, 'w') as f:
            json.dump({
                "vectors": serializable_vectors,
                "next_id": self.next_id
            }, f)
    
    def _load(self):
        """Load the vector store data from disk."""
        data_path = os.path.join(self.data_dir, "fallback_vectors.json")
        
        if os.path.exists(data_path):
            try:
                with open(data_path, 'r') as f:
                    data = json.load(f)
                
                # Load vectors
                self.vectors = []
                for item in data.get("vectors", []):
                    self.vectors.append((
                        item["id"],
                        item["embedding"],
                        item["text"],
                        item["metadata"]
                    ))
                
                # Load next ID
                self.next_id = data.get("next_id", 1)
            except Exception as e:
                logger.error(f"Error loading vector store data: {str(e)}")
