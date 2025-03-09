"""
Communication Tool Module

This module provides tools for communication between agents, enabling
information sharing, message passing, and collaborative workflows.
It supports structured communication patterns and maintains conversation history.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class CommunicationTool:
    """
    Tool for facilitating communication between agents.
    
    This tool provides methods for structured communication between agents,
    enabling information sharing, message passing, and collaborative discussions.
    It maintains conversation history and provides mechanisms for context sharing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the communication tool.
        
        Args:
            config: Configuration dictionary with communication settings
        """
        self.config = config
        self.max_message_history = config.get("max_message_history", 100)
        self.enable_agent_tagging = config.get("enable_agent_tagging", True)
        self.enable_timestamps = config.get("enable_timestamps", True)
        self.structured_messages = config.get("structured_messages", True)
        
        # Initialize conversation history
        self.conversation_history = []
        self.agent_states = {}
        self.shared_context = {}
        
        logger.debug("Initialized CommunicationTool")
    
    def send_message(
        self, 
        sender_id: str, 
        receiver_id: Optional[str], 
        message_content: Union[str, Dict[str, Any]],
        message_type: str = "text"
    ) -> Dict[str, Any]:
        """
        Send a message from one agent to another, or broadcast to all agents.
        
        Args:
            sender_id: ID of the sending agent
            receiver_id: ID of the receiving agent, or None for broadcast
            message_content: The content of the message
            message_type: Type of message (text, data, request, response)
            
        Returns:
            Dictionary with message details and status
        """
        # Create the message structure
        message = {
            "message_id": f"msg_{int(time.time())}_{hash(str(message_content)) % 10000}",
            "sender_id": sender_id,
            "receiver_id": receiver_id if receiver_id else "broadcast",
            "content": message_content,
            "type": message_type,
            "status": "sent"
        }
        
        # Add timestamp if enabled
        if self.enable_timestamps:
            message["timestamp"] = datetime.now().isoformat()
        
        # Add to conversation history
        self.conversation_history.append(message)
        
        # Trim conversation history if it exceeds the maximum
        if len(self.conversation_history) > self.max_message_history:
            self.conversation_history = self.conversation_history[-self.max_message_history:]
        
        logger.debug(f"Message sent from {sender_id} to {receiver_id if receiver_id else 'broadcast'}")
        
        return {
            "message": message,
            "success": True
        }
    
    def receive_messages(
        self, 
        receiver_id: str, 
        sender_id: Optional[str] = None,
        message_type: Optional[str] = None,
        unread_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve messages intended for a specific agent.
        
        Args:
            receiver_id: ID of the receiving agent
            sender_id: Optional filter for messages from a specific sender
            message_type: Optional filter for message type
            unread_only: Whether to return only unread messages
            
        Returns:
            List of messages for the receiving agent
        """
        messages = []
        
        for message in self.conversation_history:
            # Check if the message is intended for this receiver
            is_recipient = (
                message["receiver_id"] == receiver_id or 
                message["receiver_id"] == "broadcast"
            )
            
            # Apply filters
            sender_match = not sender_id or message["sender_id"] == sender_id
            type_match = not message_type or message["type"] == message_type
            status_match = not unread_only or message.get("status") != "read"
            
            if is_recipient and sender_match and type_match and status_match:
                # Create a copy of the message
                msg_copy = message.copy()
                
                # Mark as read if it wasn't before
                if msg_copy.get("status") != "read":
                    # Update the original message status
                    message["status"] = "read"
                    # Update the copy as well
                    msg_copy["status"] = "read"
                
                messages.append(msg_copy)
        
        logger.debug(f"Retrieved {len(messages)} messages for {receiver_id}")
        return messages
    
    def update_agent_state(self, agent_id: str, state_update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the shared state of an agent.
        
        Args:
            agent_id: ID of the agent
            state_update: Dictionary with state updates
            
        Returns:
            Dictionary with updated state
        """
        # Initialize agent state if it doesn't exist
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = {}
        
        # Update the agent state
        self.agent_states[agent_id].update(state_update)
        
        # Add timestamp of last update
        self.agent_states[agent_id]["last_updated"] = datetime.now().isoformat()
        
        logger.debug(f"Updated state for agent {agent_id}")
        
        return {
            "agent_id": agent_id,
            "state": self.agent_states[agent_id],
            "success": True
        }
    
    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """
        Get the current state of an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary with agent state
        """
        # Return empty state if agent doesn't exist
        if agent_id not in self.agent_states:
            return {
                "agent_id": agent_id,
                "state": {},
                "success": False,
                "error": "Agent state not found"
            }
        
        return {
            "agent_id": agent_id,
            "state": self.agent_states[agent_id],
            "success": True
        }
    
    def share_context(
        self, 
        context_id: str, 
        content: Any, 
        access_scope: Union[str, List[str]] = "all"
    ) -> Dict[str, Any]:
        """
        Share context information with other agents.
        
        Args:
            context_id: Identifier for this context
            content: The context content to share
            access_scope: "all" for all agents, or list of specific agent IDs
            
        Returns:
            Dictionary with context details
        """
        # Create the context structure
        context = {
            "context_id": context_id,
            "content": content,
            "access_scope": access_scope,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to shared context
        self.shared_context[context_id] = context
        
        logger.debug(f"Shared context '{context_id}' with scope {access_scope}")
        
        return {
            "context": context,
            "success": True
        }
    
    def get_context(self, context_id: str, agent_id: str) -> Dict[str, Any]:
        """
        Retrieve shared context by ID if the agent has access.
        
        Args:
            context_id: ID of the context to retrieve
            agent_id: ID of the agent requesting context
            
        Returns:
            Dictionary with context content if available
        """
        # Check if context exists
        if context_id not in self.shared_context:
            return {
                "context_id": context_id,
                "content": None,
                "success": False,
                "error": "Context not found"
            }
        
        context = self.shared_context[context_id]
        access_scope = context["access_scope"]
        
        # Check if agent has access
        has_access = (
            access_scope == "all" or 
            (isinstance(access_scope, list) and agent_id in access_scope)
        )
        
        if not has_access:
            return {
                "context_id": context_id,
                "content": None,
                "success": False,
                "error": "Access denied"
            }
        
        return {
            "context_id": context_id,
            "content": context["content"],
            "timestamp": context["timestamp"],
            "success": True
        }
    
    def get_all_accessible_contexts(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get all contexts accessible to a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of accessible contexts
        """
        accessible_contexts = []
        
        for context_id, context in self.shared_context.items():
            access_scope = context["access_scope"]
            
            # Check if agent has access
            has_access = (
                access_scope == "all" or 
                (isinstance(access_scope, list) and agent_id in access_scope)
            )
            
            if has_access:
                accessible_contexts.append({
                    "context_id": context_id,
                    "content": context["content"],
                    "timestamp": context["timestamp"],
                    "success": True
                })
        
        return accessible_contexts
    
    def create_structured_message(
        self, 
        sender_id: str, 
        action: str, 
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a structured message for standardized agent communication.
        
        Args:
            sender_id: ID of the sending agent
            action: The action or intent of the message
            data: The main payload of the message
            metadata: Optional additional metadata
            
        Returns:
            Structured message dictionary
        """
        if not self.structured_messages:
            logger.warning("Structured messages are disabled in configuration")
        
        # Create structured message
        message = {
            "action": action,
            "data": data,
            "metadata": metadata or {}
        }
        
        # Add sender ID
        if self.enable_agent_tagging:
            message["sender_id"] = sender_id
        
        # Add timestamp
        if self.enable_timestamps:
            message["timestamp"] = datetime.now().isoformat()
        
        return message
    
    def get_conversation_summary(
        self, 
        max_messages: int = 10, 
        participants: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get a summary of recent conversation history.
        
        Args:
            max_messages: Maximum number of messages to include
            participants: Optional filter for specific participants
            
        Returns:
            Dictionary with conversation summary
        """
        # Filter messages by participants if specified
        filtered_history = self.conversation_history
        if participants:
            filtered_history = [
                msg for msg in self.conversation_history
                if msg["sender_id"] in participants or msg["receiver_id"] in participants
            ]
        
        # Get the most recent messages
        recent_messages = filtered_history[-max_messages:] if filtered_history else []
        
        # Create summary statistics
        message_count = len(self.conversation_history)
        agent_participation = {}
        
        for message in self.conversation_history:
            sender = message["sender_id"]
            if sender not in agent_participation:
                agent_participation[sender] = 0
            agent_participation[sender] += 1
        
        # Sort agents by participation
        sorted_participation = sorted(
            agent_participation.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Format the summary
        summary = {
            "total_messages": message_count,
            "agent_participation": dict(sorted_participation),
            "recent_messages": recent_messages
        }
        
        return summary
    
    def clear_conversation_history(self) -> Dict[str, Any]:
        """
        Clear the conversation history.
        
        Returns:
            Status dictionary
        """
        old_count = len(self.conversation_history)
        self.conversation_history = []
        
        logger.info(f"Cleared conversation history ({old_count} messages)")
        
        return {
            "success": True,
            "cleared_messages": old_count
        }
    
    def export_conversation_history(self, format: str = "json") -> Dict[str, Any]:
        """
        Export the conversation history in various formats.
        
        Args:
            format: Export format (json, text, html)
            
        Returns:
            Dictionary with exported content
        """
        if format.lower() == "json":
            # Export as JSON
            export_data = json.dumps(self.conversation_history, indent=2)
            
        elif format.lower() == "text":
            # Export as plain text
            lines = []
            for msg in self.conversation_history:
                sender = msg["sender_id"]
                receiver = msg["receiver_id"]
                timestamp = msg.get("timestamp", "")
                content = msg["content"]
                
                if isinstance(content, dict):
                    # Format dictionary content
                    content_str = json.dumps(content)
                else:
                    content_str = str(content)
                
                line = f"[{timestamp}] {sender} -> {receiver}: {content_str}"
                lines.append(line)
            
            export_data = "\n".join(lines)
            
        elif format.lower() == "html":
            # Export as HTML
            html_lines = [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                "  <title>Conversation History</title>",
                "  <style>",
                "    body { font-family: Arial, sans-serif; margin: 20px; }",
                "    .message { margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; }",
                "    .sender { font-weight: bold; }",
                "    .timestamp { color: #888; font-size: 0.8em; }",
                "    .content { margin-top: 5px; white-space: pre-wrap; }",
                "  </style>",
                "</head>",
                "<body>",
                "  <h1>Conversation History</h1>"
            ]
            
            for msg in self.conversation_history:
                sender = msg["sender_id"]
                receiver = msg["receiver_id"]
                timestamp = msg.get("timestamp", "")
                content = msg["content"]
                
                if isinstance(content, dict):
                    # Format dictionary content
                    content_str = json.dumps(content, indent=2)
                else:
                    content_str = str(content)
                
                html_lines.append("  <div class='message'>")
                html_lines.append(f"    <div class='sender'>{sender} -> {receiver}</div>")
                html_lines.append(f"    <div class='timestamp'>{timestamp}</div>")
                html_lines.append(f"    <div class='content'>{content_str}</div>")
                html_lines.append("  </div>")
            
            html_lines.append("</body>")
            html_lines.append("</html>")
            
            export_data = "\n".join(html_lines)
            
        else:
            return {
                "success": False,
                "error": f"Unsupported format: {format}",
                "supported_formats": ["json", "text", "html"]
            }
        
        return {
            "success": True,
            "format": format,
            "data": export_data,
            "message_count": len(self.conversation_history)
        }
