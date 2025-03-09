"""
Team Memory Module

This module provides a shared memory system for the agent team, allowing agents to
store and retrieve information throughout the execution of tasks. It enables
persistent context and knowledge sharing between different agents.
"""

import logging
import time
import json
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class TeamMemory:
    """
    Shared memory system for the agent team.
    
    TeamMemory provides a central repository for shared information,
    allowing agents to store and retrieve data across multiple interactions.
    It supports different memory types, persistence, and efficient retrieval.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the team memory system.
        
        Args:
            config: Configuration dictionary with memory settings
        """
        self.config = config
        self.data_dir = config.get("data_dir", "data")
        self.memory_file = config.get("memory_file", "team_memory.json")
        self.max_entries = config.get("max_entries", 1000)
        self.enable_persistence = config.get("enable_persistence", True)
        
        # Initialize memory storage
        self.working_memory = {}  # Short-term memory for current task
        self.long_term_memory = {}  # Persistent memory across tasks
        self.agent_memories = {}  # Agent-specific memories
        self.task_memories = {}  # Task-specific memories
        
        # Create data directory if it doesn't exist
        if self.enable_persistence:
            os.makedirs(self.data_dir, exist_ok=True)
            self._load_from_disk()
        
        logger.debug("Initialized TeamMemory")
    
    def store(self, key: str, value: Any, memory_type: str = "working", metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a value in memory.
        
        Args:
            key: Key to store the value under
            value: Value to store
            memory_type: Type of memory ("working", "long_term", "agent", "task")
            metadata: Optional metadata about the value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare the memory entry
            entry = {
                "value": value,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Store in the appropriate memory
            if memory_type == "working":
                self.working_memory[key] = entry
            elif memory_type == "long_term":
                self.long_term_memory[key] = entry
            elif memory_type == "agent":
                agent_id = metadata.get("agent_id")
                if not agent_id:
                    logger.error("Agent ID required for agent memory")
                    return False
                
                if agent_id not in self.agent_memories:
                    self.agent_memories[agent_id] = {}
                
                self.agent_memories[agent_id][key] = entry
            elif memory_type == "task":
                task_id = metadata.get("task_id")
                if not task_id:
                    logger.error("Task ID required for task memory")
                    return False
                
                if task_id not in self.task_memories:
                    self.task_memories[task_id] = {}
                
                self.task_memories[task_id][key] = entry
            else:
                logger.error(f"Unknown memory type: {memory_type}")
                return False
            
            # Enforce maximum entries limit
            self._enforce_limits()
            
            # Persist memory if enabled
            if self.enable_persistence and memory_type != "working":
                self._save_to_disk()
            
            logger.debug(f"Stored value with key '{key}' in {memory_type} memory")
            return True
            
        except Exception as e:
            logger.error(f"Error storing value in memory: {str(e)}")
            return False
    
    def retrieve(self, key: str, memory_type: str = "working", metadata: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Retrieve a value from memory.
        
        Args:
            key: Key to retrieve
            memory_type: Type of memory to retrieve from
            metadata: Optional metadata for specific memory types
            
        Returns:
            The stored value or None if not found
        """
        try:
            # Retrieve from the appropriate memory
            if memory_type == "working":
                entry = self.working_memory.get(key)
            elif memory_type == "long_term":
                entry = self.long_term_memory.get(key)
            elif memory_type == "agent":
                agent_id = metadata.get("agent_id")
                if not agent_id:
                    logger.error("Agent ID required for agent memory")
                    return None
                
                if agent_id not in self.agent_memories:
                    return None
                
                entry = self.agent_memories[agent_id].get(key)
            elif memory_type == "task":
                task_id = metadata.get("task_id")
                if not task_id:
                    logger.error("Task ID required for task memory")
                    return None
                
                if task_id not in self.task_memories:
                    return None
                
                entry = self.task_memories[task_id].get(key)
            else:
                logger.error(f"Unknown memory type: {memory_type}")
                return None
            
            # Return the value if found
            if entry:
                return entry["value"]
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving value from memory: {str(e)}")
            return None
    
    def update(self, key: str, value: Any, memory_type: str = "working", metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing value in memory.
        
        Args:
            key: Key to update
            value: New value
            memory_type: Type of memory to update
            metadata: Optional metadata
            
        Returns:
            True if successful, False otherwise
        """
        # Check if the key exists
        exists = self.retrieve(key, memory_type, metadata) is not None
        
        # If it exists, store the new value
        if exists:
            return self.store(key, value, memory_type, metadata)
        
        logger.warning(f"Key '{key}' not found in {memory_type} memory, cannot update")
        return False
    
    def delete(self, key: str, memory_type: str = "working", metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Delete a value from memory.
        
        Args:
            key: Key to delete
            memory_type: Type of memory to delete from
            metadata: Optional metadata for specific memory types
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete from the appropriate memory
            if memory_type == "working":
                if key in self.working_memory:
                    del self.working_memory[key]
                    return True
            elif memory_type == "long_term":
                if key in self.long_term_memory:
                    del self.long_term_memory[key]
                    if self.enable_persistence:
                        self._save_to_disk()
                    return True
            elif memory_type == "agent":
                agent_id = metadata.get("agent_id")
                if not agent_id:
                    logger.error("Agent ID required for agent memory")
                    return False
                
                if agent_id in self.agent_memories and key in self.agent_memories[agent_id]:
                    del self.agent_memories[agent_id][key]
                    if self.enable_persistence:
                        self._save_to_disk()
                    return True
            elif memory_type == "task":
                task_id = metadata.get("task_id")
                if not task_id:
                    logger.error("Task ID required for task memory")
                    return False
                
                if task_id in self.task_memories and key in self.task_memories[task_id]:
                    del self.task_memories[task_id][key]
                    if self.enable_persistence:
                        self._save_to_disk()
                    return True
            else:
                logger.error(f"Unknown memory type: {memory_type}")
                return False
            
            logger.warning(f"Key '{key}' not found in {memory_type} memory, nothing to delete")
            return False
            
        except Exception as e:
            logger.error(f"Error deleting value from memory: {str(e)}")
            return False
    
    def search_memory(self, query: str, memory_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for values in memory that match the query.
        This is a simple string matching search.
        
        Args:
            query: String to search for
            memory_types: List of memory types to search in (default all)
            
        Returns:
            List of matching memory entries
        """
        if memory_types is None:
            memory_types = ["working", "long_term", "agent", "task"]
            
        results = []
        
        # Helper function to search in a memory dictionary
        def search_dict(memory_dict, memory_type, extra_meta=None):
            for key, entry in memory_dict.items():
                value = entry["value"]
                value_str = str(value)
                
                if query.lower() in key.lower() or query.lower() in value_str.lower():
                    result = {
                        "key": key,
                        "value": value,
                        "memory_type": memory_type,
                        "timestamp": entry["timestamp"],
                        "metadata": entry["metadata"].copy()
                    }
                    
                    # Add extra metadata if provided
                    if extra_meta:
                        result["metadata"].update(extra_meta)
                    
                    results.append(result)
        
        # Search in each requested memory type
        if "working" in memory_types:
            search_dict(self.working_memory, "working")
        
        if "long_term" in memory_types:
            search_dict(self.long_term_memory, "long_term")
        
        if "agent" in memory_types:
            for agent_id, agent_memory in self.agent_memories.items():
                search_dict(agent_memory, "agent", {"agent_id": agent_id})
        
        if "task" in memory_types:
            for task_id, task_memory in self.task_memories.items():
                search_dict(task_memory, "task", {"task_id": task_id})
        
        return results
    
    def list_keys(self, memory_type: str = "working", metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        List all keys in a specific memory type.
        
        Args:
            memory_type: Type of memory to list keys from
            metadata: Optional metadata for specific memory types
            
        Returns:
            List of keys
        """
        try:
            # Get keys from the appropriate memory
            if memory_type == "working":
                return list(self.working_memory.keys())
            elif memory_type == "long_term":
                return list(self.long_term_memory.keys())
            elif memory_type == "agent":
                agent_id = metadata.get("agent_id")
                if not agent_id:
                    logger.error("Agent ID required for agent memory")
                    return []
                
                if agent_id not in self.agent_memories:
                    return []
                
                return list(self.agent_memories[agent_id].keys())
            elif memory_type == "task":
                task_id = metadata.get("task_id")
                if not task_id:
                    logger.error("Task ID required for task memory")
                    return []
                
                if task_id not in self.task_memories:
                    return []
                
                return list(self.task_memories[task_id].keys())
            else:
                logger.error(f"Unknown memory type: {memory_type}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing keys: {str(e)}")
            return []
    
    def clear_memory(self, memory_type: str = "working", metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Clear all entries from a specific memory type.
        
        Args:
            memory_type: Type of memory to clear
            metadata: Optional metadata for specific memory types
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear the appropriate memory
            if memory_type == "working":
                self.working_memory = {}
                logger.info("Cleared working memory")
                return True
            elif memory_type == "long_term":
                self.long_term_memory = {}
                if self.enable_persistence:
                    self._save_to_disk()
                logger.info("Cleared long-term memory")
                return True
            elif memory_type == "agent":
                agent_id = metadata.get("agent_id")
                if not agent_id:
                    logger.error("Agent ID required for agent memory")
                    return False
                
                if agent_id in self.agent_memories:
                    self.agent_memories[agent_id] = {}
                    if self.enable_persistence:
                        self._save_to_disk()
                    logger.info(f"Cleared memory for agent {agent_id}")
                    return True
                
                logger.warning(f"Agent {agent_id} not found in memory")
                return False
            elif memory_type == "task":
                task_id = metadata.get("task_id")
                if not task_id:
                    logger.error("Task ID required for task memory")
                    return False
                
                if task_id in self.task_memories:
                    self.task_memories[task_id] = {}
                    if self.enable_persistence:
                        self._save_to_disk()
                    logger.info(f"Cleared memory for task {task_id}")
                    return True
                
                logger.warning(f"Task {task_id} not found in memory")
                return False
            elif memory_type == "all":
                self.working_memory = {}
                self.long_term_memory = {}
                self.agent_memories = {}
                self.task_memories = {}
                if self.enable_persistence:
                    self._save_to_disk()
                logger.info("Cleared all memory")
                return True
            else:
                logger.error(f"Unknown memory type: {memory_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
            return False
    
    def _enforce_limits(self):
        """Enforce memory size limits by removing oldest entries if needed."""
        # Check working memory
        if len(self.working_memory) > self.max_entries:
            # Sort by timestamp and keep only the most recent entries
            sorted_entries = sorted(
                self.working_memory.items(),
                key=lambda x: x[1]["timestamp"],
                reverse=True
            )
            self.working_memory = dict(sorted_entries[:self.max_entries])
        
        # Check long-term memory
        if len(self.long_term_memory) > self.max_entries:
            sorted_entries = sorted(
                self.long_term_memory.items(),
                key=lambda x: x[1]["timestamp"],
                reverse=True
            )
            self.long_term_memory = dict(sorted_entries[:self.max_entries])
    
    def _save_to_disk(self):
        """Save memory to disk for persistence."""
        if not self.enable_persistence:
            return
            
        try:
            # Create the memory data structure
            memory_data = {
                "long_term_memory": self.long_term_memory,
                "agent_memories": self.agent_memories,
                "task_memories": self.task_memories,
                "last_saved": datetime.now().isoformat()
            }
            
            # Save to file
            file_path = os.path.join(self.data_dir, self.memory_file)
            with open(file_path, 'w') as f:
                json.dump(memory_data, f, indent=2)
                
            logger.debug(f"Saved memory to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving memory to disk: {str(e)}")
    
    def _load_from_disk(self):
        """Load memory from disk."""
        if not self.enable_persistence:
            return
            
        try:
            file_path = os.path.join(self.data_dir, self.memory_file)
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    memory_data = json.load(f)
                    
                self.long_term_memory = memory_data.get("long_term_memory", {})
                self.agent_memories = memory_data.get("agent_memories", {})
                self.task_memories = memory_data.get("task_memories", {})
                
                logger.info(f"Loaded memory from {file_path}")
            else:
                logger.info(f"No memory file found at {file_path}, starting with empty memory")
                
        except Exception as e:
            logger.error(f"Error loading memory from disk: {str(e)}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {
            "working_memory_size": len(self.working_memory),
            "long_term_memory_size": len(self.long_term_memory),
            "agent_memories_count": len(self.agent_memories),
            "task_memories_count": len(self.task_memories),
            "total_agent_memory_entries": sum(len(mem) for mem in self.agent_memories.values()),
            "total_task_memory_entries": sum(len(mem) for mem in self.task_memories.values())
        }
        
        return stats
    
    def export_memory(self, memory_type: str = "all") -> Dict[str, Any]:
        """
        Export memory data for the specified memory type.
        
        Args:
            memory_type: Type of memory to export
            
        Returns:
            Dictionary with exported memory data
        """
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "memory_type": memory_type
        }
        
        if memory_type == "working" or memory_type == "all":
            export_data["working_memory"] = self.working_memory
            
        if memory_type == "long_term" or memory_type == "all":
            export_data["long_term_memory"] = self.long_term_memory
            
        if memory_type == "agent" or memory_type == "all":
            export_data["agent_memories"] = self.agent_memories
            
        if memory_type == "task" or memory_type == "all":
            export_data["task_memories"] = self.task_memories
        
        return export_data
    
    def import_memory(self, import_data: Dict[str, Any], overwrite: bool = False) -> bool:
        """
        Import memory data.
        
        Args:
            import_data: Dictionary with memory data to import
            overwrite: Whether to overwrite existing memory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            memory_type = import_data.get("memory_type", "unknown")
            
            if memory_type == "working" or memory_type == "all":
                if overwrite:
                    self.working_memory = import_data.get("working_memory", {})
                else:
                    self.working_memory.update(import_data.get("working_memory", {}))
                    
            if memory_type == "long_term" or memory_type == "all":
                if overwrite:
                    self.long_term_memory = import_data.get("long_term_memory", {})
                else:
                    self.long_term_memory.update(import_data.get("long_term_memory", {}))
                    
            if memory_type == "agent" or memory_type == "all":
                agent_memories = import_data.get("agent_memories", {})
                if overwrite:
                    self.agent_memories = agent_memories
                else:
                    for agent_id, memory in agent_memories.items():
                        if agent_id not in self.agent_memories:
                            self.agent_memories[agent_id] = {}
                        self.agent_memories[agent_id].update(memory)
                    
            if memory_type == "task" or memory_type == "all":
                task_memories = import_data.get("task_memories", {})
                if overwrite:
                    self.task_memories = task_memories
                else:
                    for task_id, memory in task_memories.items():
                        if task_id not in self.task_memories:
                            self.task_memories[task_id] = {}
                        self.task_memories[task_id].update(memory)
            
            # Enforce limits after import
            self._enforce_limits()
            
            # Save to disk if enabled
            if self.enable_persistence:
                self._save_to_disk()
                
            logger.info(f"Successfully imported {memory_type} memory")
            return True
            
        except Exception as e:
            logger.error(f"Error importing memory: {str(e)}")
            return False
