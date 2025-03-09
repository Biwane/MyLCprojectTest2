"""
Knowledge Repository Module

This module is responsible for storing, retrieving, and managing shared knowledge
across agents. It provides a centralized repository for information that can be
accessed and updated by all agents in the team.
"""

import logging
import os
import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class KnowledgeRepository:
    """
    Repository for storing and retrieving knowledge shared across agents.
    Provides vector store capabilities for semantic search and structured
    storage for team compositions, execution results, and other data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the knowledge repository.
        
        Args:
            config: Configuration dictionary with repository settings
        """
        self.config = config
        self.data_dir = config.get("data_dir", "data")
        self.embedding_model = config.get("embedding_model", "text-embedding-3-small")
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 100)
        
        # Initialize structured storage
        self._structured_data = {
            "team_compositions": {},
            "execution_results": {},
            "agent_contributions": {},
            "task_schedules": {},
            "external_knowledge": {},
        }
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load any existing data
        self._load_structured_data()
        
        # Initialize vector store for semantic search
        self._init_vector_store()
        
        logger.debug(f"Initialized KnowledgeRepository with embedding model: {self.embedding_model}")
    
    def _init_vector_store(self):
        """Initialize the vector store for semantic search."""
        vector_store_dir = os.path.join(self.data_dir, "vector_store")
        os.makedirs(vector_store_dir, exist_ok=True)
        
        # Initialize embedding model
        try:
            # Try to use HuggingFaceEmbeddings instead of OpenAI
            from langchain.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            logger.info("Using HuggingFace embeddings")
            
            # Check if vector store exists
            if os.path.exists(os.path.join(vector_store_dir, "chroma.sqlite3")):
                logger.debug("Loading existing vector store")
                self.vector_store = Chroma(
                    persist_directory=vector_store_dir,
                    embedding_function=self.embeddings
                )
            else:
                logger.debug("Creating new vector store")
                self.vector_store = Chroma(
                    persist_directory=vector_store_dir,
                    embedding_function=self.embeddings
                )
                
            # Create text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            self.vector_store_initialized = True
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            self.vector_store_initialized = False
    
    def _load_structured_data(self):
        """Load structured data from disk if available."""
        structured_data_path = os.path.join(self.data_dir, "structured_data.json")
        
        if os.path.exists(structured_data_path):
            try:
                with open(structured_data_path, 'r') as f:
                    loaded_data = json.load(f)
                    
                    # Update storage with loaded data
                    for key, value in loaded_data.items():
                        if key in self._structured_data:
                            self._structured_data[key] = value
                
                logger.info("Loaded structured data from disk")
            except Exception as e:
                logger.error(f"Error loading structured data: {str(e)}")
    
    def _save_structured_data(self):
        """Save structured data to disk."""
        structured_data_path = os.path.join(self.data_dir, "structured_data.json")
        
        try:
            with open(structured_data_path, 'w') as f:
                json.dump(self._structured_data, f, indent=2)
                
            logger.debug("Saved structured data to disk")
        except Exception as e:
            logger.error(f"Error saving structured data: {str(e)}")
    
    def store_team_composition(self, task_description: str, team_composition: Dict[str, Any]):
        """
        Store team composition information in both structured_data and teams.json
        """
        # Generate a task ID based on timestamp and task description
        task_id = f"team_{int(time.time())}_{hash(task_description) % 10000}"
        
        # Add timestamp
        team_composition["timestamp"] = datetime.now().isoformat()
        team_composition["task_description"] = task_description
        
        # Store in structured data (existing functionality)
        self._structured_data["team_compositions"][task_id] = team_composition
        
        # Also store in teams.json
        self._save_team_to_teams_file(task_id, team_composition)
        
        # Save to disk
        self._save_structured_data()
        
        logger.info(f"Stored team composition for team ID: {task_id}")
        
        # Return the team ID for reference
        return task_id

    def _save_team_to_teams_file(self, team_id: str, team_composition: Dict[str, Any]):
        """
        Save a team composition to the dedicated teams.json file
        """
        teams_file = os.path.join(self.data_dir, "teams.json")
        
        # Load existing teams
        teams = {}
        if os.path.exists(teams_file):
            try:
                with open(teams_file, 'r') as f:
                    teams = json.load(f)
            except json.JSONDecodeError:
                # If file exists but is invalid, start with empty dict
                teams = {}
        
        # Add the new team
        teams[team_id] = {
            "id": team_id,
            "name": team_composition.get("team_name", "Unnamed Team"),
            "description": team_composition.get("team_goal", "No description"),
            "created_at": team_composition.get("timestamp", datetime.now().isoformat()),
            "task_description": team_composition.get("task_description", ""),
            "agent_specs": team_composition.get("agent_specs", []),
            "additional_context": team_composition.get("additional_context", "")
        }
        
        # Save to file
        with open(teams_file, 'w') as f:
            json.dump(teams, f, indent=2)

    def get_all_teams(self) -> Dict[str, Any]:
        """
        Get all teams from the teams.json file
        """
        teams_file = os.path.join(self.data_dir, "teams.json")
        
        if not os.path.exists(teams_file):
            return {}
        
        try:
            with open(teams_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error reading teams file: {teams_file}")
            return {}

    def get_team(self, team_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a team by ID from the teams.json file
        """
        teams = self.get_all_teams()
        return teams.get(team_id)

    def store_execution_results(
        self, 
        task_description: str, 
        execution_results: Dict[str, Any],
        conversation_history: Optional[List[Any]] = None
    ):
        """
        Store execution results from a task.
        
        Args:
            task_description: The task description
            execution_results: The execution results data
            conversation_history: Optional conversation history
        """
        # Generate a task ID based on timestamp and task description
        task_id = f"execution_{int(time.time())}_{hash(task_description) % 10000}"
        
        # Create storage object
        storage_obj = {
            "timestamp": datetime.now().isoformat(),
            "task_description": task_description,
            "execution_results": execution_results
        }
        
        # Add conversation history if provided
        if conversation_history:
            # Convert conversation history to serializable format
            serializable_history = []
            for message in conversation_history:
                if hasattr(message, "to_dict"):
                    serializable_history.append(message.to_dict())
                else:
                    serializable_history.append({
                        "type": type(message).__name__,
                        "content": str(message)
                    })
            
            storage_obj["conversation_history"] = serializable_history
        
        # Store in structured data
        self._structured_data["execution_results"][task_id] = storage_obj
        
        # Save to disk
        self._save_structured_data()
        
        logger.info(f"Stored execution results for task ID: {task_id}")
        
        # Also add to vector store for semantic search
        if self.vector_store_initialized:
            # Extract text content from execution results
            content_parts = [f"Execution Results for: {task_description}"]
            
            # Add each result output
            for subtask_id, result in execution_results.items():
                subtask_desc = result.get("subtask", {}).get("description", "Unknown subtask")
                agent_id = result.get("agent_id", "unknown")
                output = result.get("output", "No output")
                
                content_parts.append(f"Subtask: {subtask_desc}")
                content_parts.append(f"Agent: {agent_id}")
                content_parts.append(f"Output: {output}")
            
            self._add_to_vector_store(
                text="\n\n".join(content_parts),
                metadata={
                    "type": "execution_results",
                    "task_id": task_id,
                    "timestamp": storage_obj["timestamp"]
                }
            )
    
    def store_external_knowledge(
        self, 
        source: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store external knowledge such as web search results or documentation.
        
        Args:
            source: Source of the knowledge (e.g., URL, document name)
            content: The content to store
            metadata: Additional metadata about the content
        """
        # Generate an ID for this knowledge
        knowledge_id = f"knowledge_{int(time.time())}_{hash(source) % 10000}"
        
        # Create storage object
        metadata = metadata or {}
        storage_obj = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "content": content,
            "metadata": metadata
        }
        
        # Store in structured data
        self._structured_data["external_knowledge"][knowledge_id] = storage_obj
        
        # Save to disk
        self._save_structured_data()
        
        logger.info(f"Stored external knowledge with ID: {knowledge_id}")
        
        # Also add to vector store for semantic search
        if self.vector_store_initialized:
            self._add_to_vector_store(
                text=f"Knowledge from {source}:\n{content}",
                metadata={
                    "type": "external_knowledge",
                    "knowledge_id": knowledge_id,
                    "source": source,
                    "timestamp": storage_obj["timestamp"],
                    **metadata
                }
            )
    
    def _add_to_vector_store(self, text: str, metadata: Dict[str, Any]):
        """
        Add text to the vector store with metadata.
        
        Args:
            text: The text content to add
            metadata: Metadata about the content
        """
        if not self.vector_store_initialized:
            logger.warning("Vector store not initialized, skipping addition")
            return
        
        try:
            # Split text into chunks
            docs = self.text_splitter.create_documents([text], [metadata])
            
            # Add to vector store
            self.vector_store.add_documents(docs)
            
            # Persist the vector store
            if hasattr(self.vector_store, "persist"):
                self.vector_store.persist()
                
            logger.debug(f"Added content to vector store with metadata: {metadata.get('type')}")
            
        except Exception as e:
            logger.error(f"Error adding to vector store: {str(e)}")
    
    def search_knowledge(
        self, 
        query: str, 
        k: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge repository for relevant information.
        
        Args:
            query: The search query
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with content and metadata
        """
        if not self.vector_store_initialized:
            logger.warning("Vector store not initialized, returning empty results")
            return []
        
        try:
            # Search the vector store
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_metadata
            )
            
            # Format results
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
                
            logger.debug(f"Search for '{query}' returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def get_relevant_knowledge(
        self, 
        task_description: str, 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get knowledge relevant to a specific task.
        
        Args:
            task_description: Description of the task
            k: Number of results to return
            
        Returns:
            List of relevant knowledge items
        """
        return self.search_knowledge(query=task_description, k=k)
    
    def get_agent_knowledge(
        self, 
        agent_role: str, 
        specialization: Optional[str] = None, 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get knowledge relevant to a specific agent role and specialization.
        
        Args:
            agent_role: The agent's role
            specialization: Optional specialization
            k: Number of results to return
            
        Returns:
            List of relevant knowledge items
        """
        query = f"Knowledge for {agent_role}"
        if specialization:
            query += f" specialized in {specialization}"
            
        return self.search_knowledge(query=query, k=k)
    
    def get_team_composition(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific team composition by task ID.
        
        Args:
            task_id: The task ID
            
        Returns:
            Team composition dictionary or None if not found
        """
        return self._structured_data["team_compositions"].get(task_id)
    
    def get_execution_results(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get execution results for a specific task.
        
        Args:
            task_id: The task ID
            
        Returns:
            Execution results dictionary or None if not found
        """
        return self._structured_data["execution_results"].get(task_id)
    
    def get_external_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific external knowledge by ID.
        
        Args:
            knowledge_id: The knowledge ID
            
        Returns:
            Knowledge dictionary or None if not found
        """
        return self._structured_data["external_knowledge"].get(knowledge_id)
    
    def list_recent_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent tasks with their descriptions and IDs.
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            List of recent tasks with descriptions and IDs
        """
        # Collect tasks from team compositions and execution results
        tasks = []
        
        # Add tasks from team compositions
        for task_id, composition in self._structured_data["team_compositions"].items():
            tasks.append({
                "task_id": task_id,
                "description": composition.get("task_description", "Unknown"),
                "timestamp": composition.get("timestamp", ""),
                "type": "team_composition"
            })
        
        # Add tasks from execution results
        for task_id, results in self._structured_data["execution_results"].items():
            tasks.append({
                "task_id": task_id,
                "description": results.get("task_description", "Unknown"),
                "timestamp": results.get("timestamp", ""),
                "type": "execution_results"
            })
        
        # Sort by timestamp (recent first) and limit
        tasks.sort(key=lambda x: x["timestamp"], reverse=True)
        return tasks[:limit]
    
    def clear(self):
        """Clear all data in the repository."""
        # Clear structured data
        self._structured_data = {
            "team_compositions": {},
            "execution_results": {},
            "agent_contributions": {},
            "task_schedules": {},
            "external_knowledge": {},
        }
        
        # Save empty data to disk
        self._save_structured_data()
        
        # Clear vector store if initialized
        if self.vector_store_initialized:
            try:
                self.vector_store = Chroma(
                    persist_directory=os.path.join(self.data_dir, "vector_store"),
                    embedding_function=self.embeddings
                )
                self.vector_store.delete_collection()
                self.vector_store = Chroma(
                    persist_directory=os.path.join(self.data_dir, "vector_store"),
                    embedding_function=self.embeddings
                )
                logger.info("Vector store cleared")
            except Exception as e:
                logger.error(f"Error clearing vector store: {str(e)}")
        
        logger.info("Knowledge repository cleared")
