"""
Knowledge Retrieval Tool Module

This module provides tools for retrieving information from the knowledge repository.
It enables agents to access shared knowledge, documentation, and previously stored
information to support their decision making and task execution.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Union

from core.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)

class KnowledgeRetrievalTool:
    """
    Tool for retrieving information from the knowledge repository.
    
    This tool provides methods to search for and retrieve relevant knowledge
    from the shared knowledge repository, supporting various types of queries
    and filtering options.
    """
    
    def __init__(self, config: Dict[str, Any], knowledge_repository: KnowledgeRepository):
        """
        Initialize the knowledge retrieval tool.
        
        Args:
            config: Configuration dictionary with retrieval settings
            knowledge_repository: The knowledge repository to retrieve from
        """
        self.config = config
        self.knowledge_repository = knowledge_repository
        self.max_results = config.get("max_results", 5)
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        
        logger.debug("Initialized KnowledgeRetrievalTool")
    
    def search_knowledge(
        self, 
        query: str, 
        max_results: Optional[int] = None, 
        filter_by_type: Optional[str] = None,
        filter_by_source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge repository for relevant information.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return (overrides config)
            filter_by_type: Filter results by content type (e.g., "document", "execution_results")
            filter_by_source: Filter results by source (e.g., "web", "user", "agent")
            
        Returns:
            List of relevant knowledge items
        """
        max_results = max_results or self.max_results
        
        # Create filter metadata if needed
        filter_metadata = {}
        if filter_by_type:
            filter_metadata["type"] = filter_by_type
        if filter_by_source:
            filter_metadata["source"] = filter_by_source
        
        # Execute the search with the repository
        try:
            results = self.knowledge_repository.search_knowledge(
                query=query,
                k=max_results,
                filter_metadata=filter_metadata if filter_metadata else None
            )
            
            logger.debug(f"Knowledge search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error during knowledge search: {str(e)}")
            return [{
                "content": f"Error during knowledge search: {str(e)}",
                "metadata": {"type": "error", "source": "knowledge_retrieval_tool"}
            }]
    
    def get_relevant_knowledge(
        self, 
        task_description: str, 
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get knowledge relevant to a specific task.
        
        Args:
            task_description: Description of the task
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant knowledge items
        """
        max_results = max_results or self.max_results
        
        try:
            results = self.knowledge_repository.get_relevant_knowledge(
                task_description=task_description,
                k=max_results
            )
            
            logger.debug(f"Relevant knowledge search for task returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving relevant knowledge: {str(e)}")
            return [{
                "content": f"Error retrieving relevant knowledge: {str(e)}",
                "metadata": {"type": "error", "source": "knowledge_retrieval_tool"}
            }]
    
    def get_agent_knowledge(
        self, 
        agent_role: str, 
        specialization: Optional[str] = None, 
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get knowledge relevant to a specific agent role and specialization.
        
        Args:
            agent_role: The agent's role
            specialization: Optional specialization
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant knowledge items
        """
        max_results = max_results or self.max_results
        
        try:
            results = self.knowledge_repository.get_agent_knowledge(
                agent_role=agent_role,
                specialization=specialization,
                k=max_results
            )
            
            logger.debug(f"Agent knowledge search for {agent_role} returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving agent knowledge: {str(e)}")
            return [{
                "content": f"Error retrieving agent knowledge: {str(e)}",
                "metadata": {"type": "error", "source": "knowledge_retrieval_tool"}
            }]
    
    def retrieve_by_id(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific knowledge item by its ID.
        
        Args:
            knowledge_id: The ID of the knowledge item to retrieve
            
        Returns:
            Knowledge item if found, None otherwise
        """
        try:
            # Check if this is a team composition ID
            if knowledge_id.startswith("task_"):
                result = self.knowledge_repository.get_team_composition(knowledge_id)
                if result:
                    return {
                        "content": str(result),
                        "metadata": {
                            "type": "team_composition",
                            "task_id": knowledge_id
                        }
                    }
            
            # Check if this is an execution results ID
            if knowledge_id.startswith("execution_"):
                result = self.knowledge_repository.get_execution_results(knowledge_id)
                if result:
                    return {
                        "content": str(result),
                        "metadata": {
                            "type": "execution_results",
                            "task_id": knowledge_id
                        }
                    }
            
            # Check if this is an external knowledge ID
            if knowledge_id.startswith("knowledge_"):
                result = self.knowledge_repository.get_external_knowledge(knowledge_id)
                if result:
                    return {
                        "content": result.get("content", ""),
                        "metadata": {
                            "type": "external_knowledge",
                            "knowledge_id": knowledge_id,
                            "source": result.get("source", "unknown")
                        }
                    }
            
            logger.warning(f"Knowledge item with ID {knowledge_id} not found")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge item by ID: {str(e)}")
            return {
                "content": f"Error retrieving knowledge item: {str(e)}",
                "metadata": {"type": "error", "source": "knowledge_retrieval_tool"}
            }
    
    def get_recent_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get a list of recent tasks.
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            List of recent tasks with descriptions and IDs
        """
        try:
            recent_tasks = self.knowledge_repository.list_recent_tasks(limit=limit)
            
            logger.debug(f"Retrieved {len(recent_tasks)} recent tasks")
            return recent_tasks
            
        except Exception as e:
            logger.error(f"Error retrieving recent tasks: {str(e)}")
            return [{
                "task_id": "error",
                "description": f"Error retrieving recent tasks: {str(e)}",
                "type": "error"
            }]
    
    def format_knowledge_for_context(
        self, 
        knowledge_items: List[Dict[str, Any]], 
        include_metadata: bool = False
    ) -> str:
        """
        Format knowledge items into a string suitable for inclusion in a context.
        
        Args:
            knowledge_items: List of knowledge items to format
            include_metadata: Whether to include metadata in the formatted result
            
        Returns:
            Formatted knowledge string
        """
        if not knowledge_items:
            return "No relevant knowledge found."
        
        formatted_parts = ["Here is relevant information that might help:"]
        
        for i, item in enumerate(knowledge_items, 1):
            content = item.get("content", "")
            metadata = item.get("metadata", {})
            source = metadata.get("source", "Unknown source")
            knowledge_type = metadata.get("type", "information")
            
            formatted_parts.append(f"\n--- Relevant Information {i} (from {source}) ---")
            formatted_parts.append(content)
            
            if include_metadata:
                meta_str = "\nMetadata: "
                meta_items = []
                for key, value in metadata.items():
                    if key not in ["source", "content"]:
                        meta_items.append(f"{key}: {value}")
                
                if meta_items:
                    formatted_parts.append(meta_str + ", ".join(meta_items))
        
        return "\n".join(formatted_parts)
    
    def add_knowledge_to_repository(
        self, 
        content: str, 
        source: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add new knowledge to the repository.
        
        Args:
            content: The content to store
            source: Source of the knowledge (e.g., "web", "user", "agent")
            metadata: Additional metadata about the content
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.knowledge_repository.store_external_knowledge(
                source=source,
                content=content,
                metadata=metadata or {}
            )
            
            logger.debug(f"Added new knowledge from {source} to repository")
            return True
            
        except Exception as e:
            logger.error(f"Error adding knowledge to repository: {str(e)}")
            return False

    def combine_knowledge(
        self, 
        knowledge_items: List[Dict[str, Any]], 
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Combine knowledge from multiple items.
        
        Args:
            knowledge_items: List of knowledge items to combine
            query: Optional context for the combination
            
        Returns:
            Combined knowledge
        """
        if not knowledge_items:
            return {
                "content": "No knowledge items to combine.",
                "metadata": {"type": "combined", "item_count": 0}
            }
        
        # Simply concatenate the contents for now
        # In a more advanced implementation, this could use an LLM to synthesize information
        combined_content = []
        sources = set()
        types = set()
        
        for item in knowledge_items:
            content = item.get("content", "")
            metadata = item.get("metadata", {})
            source = metadata.get("source", "Unknown")
            item_type = metadata.get("type", "information")
            
            combined_content.append(f"From {source}:")
            combined_content.append(content)
            
            sources.add(source)
            types.add(item_type)
        
        combined_metadata = {
            "type": "combined",
            "item_count": len(knowledge_items),
            "sources": list(sources),
            "content_types": list(types)
        }
        
        # Add query context if provided
        if query:
            combined_content.insert(0, f"Combined knowledge related to: {query}")
            combined_metadata["query"] = query
        
        return {
            "content": "\n\n".join(combined_content),
            "metadata": combined_metadata
        }
