"""
Agent Factory Module

This module is responsible for creating specialized AI agents with different capabilities
based on the required roles and expertise. It serves as a factory that can instantiate
various types of agents with appropriate configurations.
"""

import logging
from typing import Dict, Any, List, Optional, Type

from langchain.agents import AgentExecutor, create_react_agent, create_tool_calling_agent
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from agents.base_agent import BaseAgent
from agents.research_agent import ResearchAgent
from agents.specialist_agent import SpecialistAgent
from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
from agents.reviewer_agent import ReviewerAgent
from utils.prompt_templates import get_prompt_template_for_role
from core.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)

class AgentFactory:
    """
    Factory class for creating different types of AI agents.
    """
    
    def __init__(self, config: Dict[str, Any], knowledge_repository: Optional[KnowledgeRepository] = None):
        """
        Initialize the agent factory.
        
        Args:
            config: Configuration dictionary with agent settings
            knowledge_repository: Knowledge repository for agents to access shared information
        """
        self.config = config
        self.knowledge_repository = knowledge_repository
        self.default_model = config.get("default_model", "gpt-4o")
        self.research_model = config.get("research_model", "gpt-4o")
        self.specialist_model = config.get("specialist_model", "gpt-4o")
        self.planner_model = config.get("planner_model", "gpt-4o")
        self.executor_model = config.get("executor_model", "gpt-4o")
        self.reviewer_model = config.get("reviewer_model", "gpt-4o")
        
        logger.debug(f"Initialized AgentFactory with models: default={self.default_model}")
    
    def _get_agent_class(self, role: str) -> Type[BaseAgent]:
        """
        Map role to agent class.
        
        Args:
            role: The role name for the agent
            
        Returns:
            The appropriate agent class
        """
        role_to_class = {
            "research": ResearchAgent,
            "specialist": SpecialistAgent,
            "planner": PlannerAgent,
            "executor": ExecutorAgent,
            "reviewer": ReviewerAgent,
        }
        
        # Get the base role (before any specialization)
        base_role = role.split("_")[0] if "_" in role else role
        
        return role_to_class.get(base_role.lower(), SpecialistAgent)
    
    def _get_model_for_role(self, role: str) -> str:
        """
        Get the appropriate model for a given role.
        
        Args:
            role: The role name for the agent
            
        Returns:
            Model name to use for this agent
        """
        role_to_model = {
            "research": self.research_model,
            "specialist": self.specialist_model,
            "planner": self.planner_model,
            "executor": self.executor_model,
            "reviewer": self.reviewer_model,
        }
        
        # Get the base role (before any specialization)
        base_role = role.split("_")[0] if "_" in role else role
        
        return role_to_model.get(base_role.lower(), self.default_model)
    
    def _create_llm(self, model_name: str, temperature: float = 0.1) -> ChatOpenAI:
        """
        Create a language model instance.
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature setting for generation (0.0 to 1.0)
            
        Returns:
            Initialized language model
        """
        return ChatOpenAI(
            model=model_name,
            temperature=temperature
        )
    
    def _create_agent_executor(
        self, 
        role: str, 
        tools: List[BaseTool], 
        system_prompt: str,
        model_name: Optional[str] = None,
        memory: Optional[Any] = None
    ) -> AgentExecutor:
        """
        Create an agent executor with the appropriate configuration.
        
        Args:
            role: The role of the agent
            tools: List of tools available to the agent
            system_prompt: System prompt for the agent
            model_name: Optional model name override
            memory: Optional memory for the agent
            
        Returns:
            Configured AgentExecutor
        """
        # Determine which model to use
        model_name = model_name or self._get_model_for_role(role)
        
        # Create the language model
        llm = self._create_llm(model_name)
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent based on the tools provided
        if tools:
            agent = create_tool_calling_agent(llm, tools, prompt)
        else:
            # Create a basic agent without tools if none provided
            agent = create_react_agent(llm, [], prompt)
        
        # Create and return the agent executor
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=self.config.get("verbose", True),
            handle_parsing_errors=True,
            max_iterations=self.config.get("max_iterations", 15),
        )
    
    def create_agent(
        self, 
        role: str, 
        specialization: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
        background_info: Optional[str] = None,
        memory: Optional[Any] = None
    ) -> BaseAgent:
        """
        Create an agent with the specified role and specialization.
        
        Args:
            role: The basic role of the agent (research, specialist, planner, etc.)
            specialization: Optional specialization within the role (e.g., "salesforce_developer")
            tools: Optional list of tools for the agent to use
            background_info: Optional background information to include in the agent's context
            memory: Optional memory component
            
        Returns:
            An initialized agent instance
        """
        # Ensure tools are always provided
        if tools is None or len(tools) == 0:
            # Create a default tool if none provided
            from langchain.tools.base import Tool
            tools = [
                Tool(
                    name="empty_tool",
                    description="A placeholder tool that does nothing",
                    func=lambda x: "This tool does nothing"
                )
            ]
        
        # Combine role and specialization if provided
        full_role = f"{role}_{specialization}" if specialization else role
        
        # Get appropriate agent class
        agent_class = self._get_agent_class(role)
        
        # Get appropriate model name
        model_name = self._get_model_for_role(role)
        
        # Get role-specific prompt template
        system_prompt = get_prompt_template_for_role(full_role)
        
        # Inject background information if provided
        if background_info:
            system_prompt = f"{system_prompt}\n\nBackground Information:\n{background_info}"
        
        # Create tools list if not provided
        tools = tools or []
        
        # Create the agent executor
        agent_executor = self._create_agent_executor(
            role=full_role,
            tools=tools,
            system_prompt=system_prompt,
            model_name=model_name,
            memory=memory
        )
        
        # Initialize and return the agent
        agent = agent_class(
            agent_executor=agent_executor,
            role=full_role,
            config=self.config,
            knowledge_repository=self.knowledge_repository
        )
        
        logger.info(f"Created agent with role: {full_role}")
        return agent

    def create_specialized_agent(
        self,
        agent_spec: Dict[str, Any],
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[Any] = None
    ) -> BaseAgent:
        """
        Create an agent based on a specification dictionary.
        
        Args:
            agent_spec: Dictionary with agent specifications including role, specialization, etc.
            tools: Optional tools for the agent
            memory: Optional memory for the agent
            
        Returns:
            An initialized agent instance
        """
        role = agent_spec.get("role", "specialist")
        specialization = agent_spec.get("specialization")
        background_info = agent_spec.get("background_info")
        
        # Merge any tools provided in the spec with those passed to the method
        agent_tools = agent_spec.get("tools", [])
        if tools:
            agent_tools.extend(tools)
        
        return self.create_agent(
            role=role,
            specialization=specialization,
            tools=agent_tools,
            background_info=background_info,
            memory=memory
        )