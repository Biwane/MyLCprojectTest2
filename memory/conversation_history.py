"""
Conversation History Module

This module provides functionality for storing, retrieving, and managing conversation
history between agents and users. It maintains context across interactions and
supports persistent storage of conversation data.
"""

import logging
import os
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import uuid

# Try importing LangChain message types
try:
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        FunctionMessage,
        ToolMessage,
        BaseMessage
    )
    LANGCHAIN_MESSAGES_AVAILABLE = True
except ImportError:
    LANGCHAIN_MESSAGES_AVAILABLE = False

logger = logging.getLogger(__name__)

class ConversationHistory:
    """
    Manages conversation history between agents and users.
    
    This class stores and retrieves conversation messages, maintains context
    across interactions, and supports persistence to disk for long-running
    conversations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the conversation history manager.
        
        Args:
            config: Configuration dictionary with history settings
        """
        self.config = config
        self.data_dir = config.get("data_dir", "data")
        self.history_dir = config.get("history_dir", "conversation_history")
        self.max_history_length = config.get("max_history_length", 100)
        self.enable_persistence = config.get("enable_persistence", True)
        self.auto_save = config.get("auto_save", True)
        
        # Create history storage
        self.conversations = {}  # Map of conversation_id to list of messages
        self.metadata = {}  # Map of conversation_id to metadata
        
        # Create data directory if it doesn't exist
        self.history_path = os.path.join(self.data_dir, self.history_dir)
        os.makedirs(self.history_path, exist_ok=True)
        
        # Load existing conversations if persistence is enabled
        if self.enable_persistence:
            self._load_conversations()
        
        logger.debug(f"Initialized ConversationHistory with max_length: {self.max_history_length}")
    
    def add_message(
        self, 
        message: Union[Dict, Any], 
        conversation_id: Optional[str] = None,
        role: Optional[str] = None,
        content: Optional[str] = None
    ) -> str:
        """
        Add a message to the conversation history.
        
        Args:
            message: Message to add (either a dict, BaseMessage object, or will be created from role/content)
            conversation_id: ID of the conversation to add to (created if None)
            role: Role of the message sender (used if message is not a dict or BaseMessage)
            content: Content of the message (used if message is not a dict or BaseMessage)
            
        Returns:
            The conversation ID
        """
        # Generate conversation ID if not provided
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        
        # Initialize conversation if it doesn't exist
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
            self.metadata[conversation_id] = {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "message_count": 0
            }
        
        # Process the message
        processed_message = self._process_message(message, role, content)
        
        # Add message to conversation
        self.conversations[conversation_id].append(processed_message)
        
        # Update metadata
        self.metadata[conversation_id]["updated_at"] = datetime.now().isoformat()
        self.metadata[conversation_id]["message_count"] += 1
        
        # Enforce maximum length
        if len(self.conversations[conversation_id]) > self.max_history_length:
            # Remove oldest messages, keeping the most recent ones
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_history_length:]
        
        # Auto-save if enabled
        if self.enable_persistence and self.auto_save:
            self._save_conversation(conversation_id)
        
        return conversation_id
    
    def _process_message(
        self,
        message: Union[Dict, Any],
        role: Optional[str] = None,
        content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a message to ensure it's in a standard format.
        
        Args:
            message: Message to process
            role: Role of the message sender
            content: Content of the message
            
        Returns:
            Processed message dictionary
        """
        # Check message type
        if LANGCHAIN_MESSAGES_AVAILABLE and isinstance(message, BaseMessage):
            # Handle LangChain message types
            processed = {
                "role": self._get_role_from_langchain_message(message),
                "content": message.content,
                "type": message.type,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add additional fields
            if hasattr(message, "additional_kwargs") and message.additional_kwargs:
                processed["additional_kwargs"] = message.additional_kwargs
                
            if hasattr(message, "id") and message.id:
                processed["message_id"] = message.id
                
        elif isinstance(message, dict):
            # Handle dictionary message
            processed = message.copy()
            
            # Ensure required fields
            if "role" not in processed:
                processed["role"] = role or "unknown"
            if "content" not in processed:
                processed["content"] = content or ""
            if "timestamp" not in processed:
                processed["timestamp"] = datetime.now().isoformat()
                
        else:
            # Create new message from role and content
            processed = {
                "role": role or "unknown",
                "content": content or str(message),
                "timestamp": datetime.now().isoformat()
            }
        
        # Add message ID if not present
        if "message_id" not in processed:
            processed["message_id"] = str(uuid.uuid4())
            
        return processed
    
    def _get_role_from_langchain_message(self, message: Any) -> str:
        """
        Get the role from a LangChain message.
        
        Args:
            message: LangChain message
            
        Returns:
            Role string
        """
        if isinstance(message, AIMessage):
            return "assistant"
        elif isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, SystemMessage):
            return "system"
        elif isinstance(message, FunctionMessage):
            return "function"
        elif isinstance(message, ToolMessage):
            return "tool"
        else:
            return "unknown"
    
    def get_history(
        self, 
        conversation_id: str, 
        limit: Optional[int] = None, 
        as_langchain_messages: bool = False
    ) -> List[Any]:
        """
        Get the conversation history.
        
        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of messages to return (default: all)
            as_langchain_messages: Whether to return LangChain message objects
            
        Returns:
            List of messages
        """
        # Check if conversation exists
        if conversation_id not in self.conversations:
            logger.warning(f"Conversation {conversation_id} not found")
            return []
        
        # Get messages
        messages = self.conversations[conversation_id]
        
        # Apply limit if specified
        if limit is not None:
            messages = messages[-limit:]
        
        # Convert to LangChain messages if requested
        if as_langchain_messages and LANGCHAIN_MESSAGES_AVAILABLE:
            return self._convert_to_langchain_messages(messages)
        
        return messages
    
    def _convert_to_langchain_messages(self, messages: List[Dict[str, Any]]) -> List[Any]:
        """
        Convert message dictionaries to LangChain message objects.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            List of LangChain message objects
        """
        langchain_messages = []
        
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            additional_kwargs = message.get("additional_kwargs", {})
            
            # Create appropriate message type
            if role == "assistant":
                langchain_messages.append(AIMessage(content=content, additional_kwargs=additional_kwargs))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content, additional_kwargs=additional_kwargs))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content, additional_kwargs=additional_kwargs))
            elif role == "function":
                langchain_messages.append(FunctionMessage(
                    content=content,
                    name=additional_kwargs.get("name", "unknown_function"),
                    additional_kwargs=additional_kwargs
                ))
            elif role == "tool":
                langchain_messages.append(ToolMessage(
                    content=content,
                    tool_call_id=additional_kwargs.get("tool_call_id", "unknown_tool"),
                    additional_kwargs=additional_kwargs
                ))
            else:
                # Default to human message for unknown types
                langchain_messages.append(HumanMessage(content=content, additional_kwargs=additional_kwargs))
        
        return langchain_messages
    
    def create_conversation(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new conversation.
        
        Args:
            metadata: Optional metadata for the conversation
            
        Returns:
            New conversation ID
        """
        conversation_id = str(uuid.uuid4())
        
        # Initialize conversation
        self.conversations[conversation_id] = []
        self.metadata[conversation_id] = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": 0
        }
        
        # Add additional metadata if provided
        if metadata:
            self.metadata[conversation_id].update(metadata)
        
        # Save if persistence is enabled
        if self.enable_persistence and self.auto_save:
            self._save_conversation(conversation_id)
        
        logger.debug(f"Created new conversation with ID: {conversation_id}")
        return conversation_id
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            conversation_id: ID of the conversation to delete
            
        Returns:
            True if successful, False otherwise
        """
        # Check if conversation exists
        if conversation_id not in self.conversations:
            logger.warning(f"Conversation {conversation_id} not found, cannot delete")
            return False
        
        # Delete from memory
        del self.conversations[conversation_id]
        del self.metadata[conversation_id]
        
        # Delete from disk if persistence is enabled
        if self.enable_persistence:
            conversation_file = os.path.join(self.history_path, f"{conversation_id}.json")
            if os.path.exists(conversation_file):
                try:
                    os.remove(conversation_file)
                except Exception as e:
                    logger.error(f"Error deleting conversation file: {str(e)}")
        
        logger.debug(f"Deleted conversation with ID: {conversation_id}")
        return True
    
    def update_metadata(self, conversation_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update the metadata for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            metadata: Metadata to update
            
        Returns:
            True if successful, False otherwise
        """
        # Check if conversation exists
        if conversation_id not in self.metadata:
            logger.warning(f"Conversation {conversation_id} not found, cannot update metadata")
            return False
        
        # Update metadata
        self.metadata[conversation_id].update(metadata)
        
        # Save if persistence is enabled
        if self.enable_persistence and self.auto_save:
            self._save_conversation(conversation_id)
        
        logger.debug(f"Updated metadata for conversation: {conversation_id}")
        return True
    
    def get_metadata(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the metadata for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Metadata dictionary or None if not found
        """
        if conversation_id not in self.metadata:
            logger.warning(f"Conversation {conversation_id} not found, cannot get metadata")
            return None
            
        return self.metadata[conversation_id]
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List all conversations.
        
        Returns:
            List of conversation summaries
        """
        conversations = []
        
        for conversation_id, metadata in self.metadata.items():
            # Create summary
            conversation_summary = {
                "conversation_id": conversation_id,
                "created_at": metadata.get("created_at"),
                "updated_at": metadata.get("updated_at"),
                "message_count": metadata.get("message_count", 0)
            }
            
            # Add custom metadata fields
            for key, value in metadata.items():
                if key not in ["created_at", "updated_at", "message_count"]:
                    conversation_summary[key] = value
            
            conversations.append(conversation_summary)
        
        # Sort by updated_at (newest first)
        conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return conversations
    
    def clear_history(self, conversation_id: str) -> bool:
        """
        Clear the history for a conversation while keeping the metadata.
        
        Args:
            conversation_id: ID of the conversation to clear
            
        Returns:
            True if successful, False otherwise
        """
        # Check if conversation exists
        if conversation_id not in self.conversations:
            logger.warning(f"Conversation {conversation_id} not found, cannot clear")
            return False
        
        # Clear messages but keep metadata
        self.conversations[conversation_id] = []
        
        # Update metadata
        self.metadata[conversation_id]["updated_at"] = datetime.now().isoformat()
        self.metadata[conversation_id]["message_count"] = 0
        
        # Save if persistence is enabled
        if self.enable_persistence and self.auto_save:
            self._save_conversation(conversation_id)
        
        logger.debug(f"Cleared conversation history for: {conversation_id}")
        return True
    
    def save_all(self) -> bool:
        """
        Save all conversations to disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_persistence:
            logger.warning("Persistence is disabled, not saving conversations")
            return False
        
        success = True
        
        for conversation_id in self.conversations:
            if not self._save_conversation(conversation_id):
                success = False
        
        return success
    
    def _save_conversation(self, conversation_id: str) -> bool:
        """
        Save a conversation to disk.
        
        Args:
            conversation_id: ID of the conversation to save
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_persistence:
            return False
            
        try:
            # Prepare data for saving
            data = {
                "conversation_id": conversation_id,
                "metadata": self.metadata.get(conversation_id, {}),
                "messages": self.conversations.get(conversation_id, [])
            }
            
            # Save to file
            file_path = os.path.join(self.history_path, f"{conversation_id}.json")
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Saved conversation {conversation_id} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving conversation {conversation_id}: {str(e)}")
            return False
    
    def _load_conversations(self):
        """Load all conversations from disk."""
        try:
            # Get all JSON files in the history directory
            for filename in os.listdir(self.history_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(self.history_path, filename)
                    
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            
                        # Extract data
                        conversation_id = data.get("conversation_id")
                        metadata = data.get("metadata", {})
                        messages = data.get("messages", [])
                        
                        # Store in memory
                        if conversation_id:
                            self.conversations[conversation_id] = messages
                            self.metadata[conversation_id] = metadata
                            
                    except Exception as e:
                        logger.error(f"Error loading conversation from {file_path}: {str(e)}")
            
            logger.info(f"Loaded {len(self.conversations)} conversations from disk")
            
        except Exception as e:
            logger.error(f"Error loading conversations: {str(e)}")
    
    def export_conversation(self, conversation_id: str, format: str = "json") -> Optional[str]:
        """
        Export a conversation to a specific format.
        
        Args:
            conversation_id: ID of the conversation to export
            format: Export format (json, text)
            
        Returns:
            Exported conversation string or None if failed
        """
        # Check if conversation exists
        if conversation_id not in self.conversations:
            logger.warning(f"Conversation {conversation_id} not found, cannot export")
            return None
        
        try:
            messages = self.conversations[conversation_id]
            metadata = self.metadata[conversation_id]
            
            if format.lower() == "json":
                # Export as JSON
                data = {
                    "conversation_id": conversation_id,
                    "metadata": metadata,
                    "messages": messages
                }
                
                return json.dumps(data, indent=2)
                
            elif format.lower() == "text":
                # Export as plain text
                lines = [f"Conversation: {conversation_id}"]
                lines.append("-" * 50)
                
                # Add metadata
                lines.append("Metadata:")
                for key, value in metadata.items():
                    lines.append(f"  {key}: {value}")
                
                lines.append("-" * 50)
                lines.append("Messages:")
                
                # Add messages
                for message in messages:
                    role = message.get("role", "unknown")
                    content = message.get("content", "")
                    timestamp = message.get("timestamp", "")
                    
                    lines.append(f"[{timestamp}] {role.upper()}: {content}")
                
                return "\n".join(lines)
                
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
                
        except Exception as e:
            logger.error(f"Error exporting conversation {conversation_id}: {str(e)}")
            return None
    
    def import_conversation(self, data: str, format: str = "json") -> Optional[str]:
        """
        Import a conversation from a specific format.
        
        Args:
            data: Conversation data to import
            format: Import format (json)
            
        Returns:
            Imported conversation ID or None if failed
        """
        try:
            if format.lower() == "json":
                # Import from JSON
                json_data = json.loads(data)
                
                conversation_id = json_data.get("conversation_id", str(uuid.uuid4()))
                metadata = json_data.get("metadata", {})
                messages = json_data.get("messages", [])
                
                # Store in memory
                self.conversations[conversation_id] = messages
                self.metadata[conversation_id] = metadata
                
                # Save if persistence is enabled
                if self.enable_persistence and self.auto_save:
                    self._save_conversation(conversation_id)
                
                logger.info(f"Imported conversation with ID: {conversation_id}")
                return conversation_id
                
            else:
                logger.error(f"Unsupported import format: {format}")
                return None
                
        except Exception as e:
            logger.error(f"Error importing conversation: {str(e)}")
            return None
    
    def get_last_message(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the last message from a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Last message or None if conversation is empty or not found
        """
        # Check if conversation exists
        if conversation_id not in self.conversations:
            logger.warning(f"Conversation {conversation_id} not found")
            return None
            
        # Check if conversation has messages
        messages = self.conversations[conversation_id]
        if not messages:
            logger.warning(f"Conversation {conversation_id} has no messages")
            return None
            
        return messages[-1]
    
    def get_last_n_messages(
        self, 
        conversation_id: str, 
        n: int, 
        as_langchain_messages: bool = False
    ) -> List[Any]:
        """
        Get the last N messages from a conversation.
        
        Args:
            conversation_id: ID of the conversation
            n: Number of messages to get
            as_langchain_messages: Whether to return LangChain message objects
            
        Returns:
            List of messages
        """
        return self.get_history(
            conversation_id=conversation_id,
            limit=n,
            as_langchain_messages=as_langchain_messages
        )
