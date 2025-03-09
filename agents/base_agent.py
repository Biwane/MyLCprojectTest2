"""
Base Agent Module

This module defines the BaseAgent class, which provides the foundation for all
specialized agents in the system. It encapsulates common functionality and interfaces
that all agents should implement.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod

from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from core.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Base class for all agent implementations in the system.
    
    This abstract class defines the interface and common functionality
    that all specialized agents should implement.
    """
    
    def __init__(
        self, 
        agent_executor: AgentExecutor,
        role: str,
        config: Dict[str, Any],
        knowledge_repository: Optional[KnowledgeRepository] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_executor: The LangChain agent executor
            role: The role of this agent (e.g., "researcher", "planner")
            config: Configuration dictionary with agent settings
            knowledge_repository: Optional knowledge repository for accessing shared information
        """
        self.agent_executor = agent_executor
        self.role = role
        self.config = config
        self.knowledge_repository = knowledge_repository
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2)
        self.execution_history = []
        
        logger.debug(f"Initialized BaseAgent with role: {role}")
    
    def execute_task(self, task_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a task with this agent.
        
        Args:
            task_input: The task input, either a string or a dictionary
            
        Returns:
            Dictionary containing the execution results
        """
        # Convert string input to dictionary
        if isinstance(task_input, str):
            task_input = {"input": task_input}
        
        # Record start time
        start_time = time.time()
        
        # Get relevant knowledge if available
        context = self._get_relevant_context(task_input)
        if context:
            # Incorporate context into the input
            if "input" in task_input:
                task_input["input"] = self._format_input_with_context(task_input["input"], context)
        
        # Execute with retries
        result = self._execute_with_retries(task_input)
        
        # Record execution time
        execution_time = time.time() - start_time
        
        # Process the result
        processed_result = self._process_result(result)
        
        # Log execution
        self._log_execution(task_input, processed_result, execution_time)
        
        return processed_result
    
    def _execute_with_retries(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task with retries in case of failures.
        
        Args:
            task_input: The task input
            
        Returns:
            Execution result
        """
        attempts = 0
        last_error = None
        
        while attempts < self.max_retries:
            try:
                # Execute task
                result = self.agent_executor.invoke(task_input)
                return result
            
            except Exception as e:
                attempts += 1
                last_error = e
                logger.warning(f"Error executing task (attempt {attempts}/{self.max_retries}): {str(e)}")
                
                # Wait before retrying
                if attempts < self.max_retries:
                    time.sleep(self.retry_delay)
        
        # All retries failed
        logger.error(f"Task execution failed after {self.max_retries} attempts: {str(last_error)}")
        
        # Return error result
        return {
            "output": f"Error: {str(last_error)}",
            "error": str(last_error)
        }
    
    def _get_relevant_context(self, task_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get relevant context for the task from the knowledge repository.
        
        Args:
            task_input: The task input
            
        Returns:
            List of relevant context items
        """
        if not self.knowledge_repository:
            return []
        
        try:
            # Extract the main input text
            input_text = task_input.get("input", "")
            
            # Get relevant knowledge
            relevant_items = self.knowledge_repository.get_relevant_knowledge(input_text, k=3)
            
            # Also get agent-specific knowledge
            agent_knowledge = self.knowledge_repository.get_agent_knowledge(self.role, k=2)
            
            # Combine and return
            return relevant_items + agent_knowledge
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def _format_input_with_context(self, input_text: str, context: List[Dict[str, Any]]) -> str:
        """
        Format the input text with the relevant context.
        
        Args:
            input_text: The original input text
            context: List of context items
            
        Returns:
            Formatted input text with context
        """
        if not context:
            return input_text
        
        # Format context items
        context_text = []
        for item in context:
            content = item.get("content", "")
            metadata = item.get("metadata", {})
            source = metadata.get("source", "Unknown source")
            
            context_text.append(f"--- From {source} ---")
            context_text.append(content)
        
        # Combine with input
        formatted_input = [
            "Here is some relevant information that might help with this task:",
            "\n".join(context_text),
            "\nYour task:",
            input_text
        ]
        
        return "\n\n".join(formatted_input)
    
    def _process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the execution result.
        
        This method can be overridden by subclasses to perform specialized processing.
        
        Args:
            result: The raw execution result
            
        Returns:
            Processed result
        """
        # This base implementation just ensures standard fields are present
        processed = result.copy()
        
        # Ensure output field exists
        if "output" not in processed:
            if "return_values" in processed:
                processed["output"] = processed["return_values"]
            elif "response" in processed:
                processed["output"] = processed["response"]
            else:
                # Fallback to string representation
                processed["output"] = str(processed)
        
        # Add metadata
        if "metadata" not in processed:
            processed["metadata"] = {}
        
        processed["metadata"]["agent_role"] = self.role
        
        return processed
    
    def _log_execution(
        self, 
        task_input: Dict[str, Any], 
        result: Dict[str, Any], 
        execution_time: float
    ):
        """
        Log the execution details for record keeping.
        
        Args:
            task_input: The task input
            result: The execution result
            execution_time: Execution time in seconds
        """
        # Create execution record
        execution_record = {
            "timestamp": time.time(),
            "agent_role": self.role,
            "task_input": task_input,
            "result": result,
            "execution_time": execution_time
        }
        
        # Add to history
        self.execution_history.append(execution_record)
        
        # Limit history size
        max_history = self.config.get("max_history", 10)
        if len(self.execution_history) > max_history:
            self.execution_history = self.execution_history[-max_history:]
        
        logger.info(f"Agent {self.role} executed task in {execution_time:.2f} seconds")
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        Get the execution history for this agent.
        
        Returns:
            List of execution records
        """
        return self.execution_history
    
    def clear_history(self):
        """Clear the execution history."""
        self.execution_history = []
        logger.debug(f"Cleared execution history for agent {self.role}")
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get the list of capabilities this agent has.
        
        This abstract method must be implemented by all subclasses.
        
        Returns:
            List of capability descriptions
        """
        pass
    
    def get_role_description(self) -> str:
        """
        Get a description of this agent's role.
        
        Returns:
            Description of the agent's role
        """
        # This can be overridden by subclasses for more specific descriptions
        return f"I am a {self.role} agent that can help with tasks related to this domain."
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.role} Agent"
    
    def __repr__(self) -> str:
        """Representation of the agent."""
        return f"<{self.__class__.__name__} role={self.role}>"
