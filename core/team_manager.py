"""
Team Manager Module

This module is responsible for analyzing tasks, determining the required team composition,
and creating teams of specialized agents to accomplish the given tasks.
"""

import logging
from typing import Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from core.agent_factory import AgentFactory
from core.knowledge_repository import KnowledgeRepository
from agents.base_agent import BaseAgent
from utils.prompt_templates import TEAM_COMPOSITION_PROMPT

logger = logging.getLogger(__name__)

class AgentSpec(BaseModel):
    """Specification for an agent to be created."""
    role: str = Field(description="The primary role of the agent (research, specialist, planner, executor, reviewer)")
    specialization: str = Field(description="The specific domain expertise of the agent")
    importance: int = Field(description="Importance level from 1-10, with 10 being most essential", default=5)
    description: str = Field(description="Brief description of the agent's responsibilities")
    required_skills: List[str] = Field(description="List of specific skills this agent needs to have")
    background_info: Optional[str] = Field(description="Additional context for this agent's initialization", default=None)

class TeamComposition(BaseModel):
    """The composition of an agent team for a specific task."""
    team_name: str = Field(description="A descriptive name for the team")
    team_goal: str = Field(description="The primary goal of this team")
    required_agents: List[AgentSpec] = Field(description="List of agent specifications")
    additional_context: Optional[str] = Field(description="Additional context for the entire team", default=None)

class TeamManager:
    """
    Manager class for analyzing tasks and creating appropriate teams of agents.
    """
    
    def __init__(self, config: Dict[str, Any], knowledge_repository: Optional[KnowledgeRepository] = None):
        """
        Initialize the team manager.
        
        Args:
            config: Configuration dictionary with team manager settings
            knowledge_repository: Knowledge repository for accessing shared information
        """
        self.config = config
        self.knowledge_repository = knowledge_repository
        self.agent_factory = AgentFactory(config.get("agent_factory", {}), knowledge_repository)
        self.analysis_model = config.get("analysis_model", "gpt-4o")
        
        logger.debug(f"Initialized TeamManager with analysis model: {self.analysis_model}")
    
    def analyze_task(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Analyze a task description and determine the required team composition.
        
        Args:
            task_description: The description of the task to be performed
            
        Returns:
            List of agent specifications for the required team members
        """
        logger.info(f"Analyzing task: {task_description}")
        
        # Create the prompt with the task description
        prompt = ChatPromptTemplate.from_template(TEAM_COMPOSITION_PROMPT)
        
        # Create a parser for the team composition
        parser = PydanticOutputParser(pydantic_object=TeamComposition)
        
        # Create the language model
        llm = ChatOpenAI(model=self.analysis_model, temperature=0.2)
        
        # Format the prompt with the task description and format instructions
        formatted_prompt = prompt.format(
            task_description=task_description,
            format_instructions=parser.get_format_instructions()
        )
        
        # Get the response from the LLM
        response = llm.invoke(formatted_prompt)
        
        try:
            # Parse the response into a TeamComposition object
            team_composition = parser.parse(response.content)
            logger.debug(f"Successfully parsed team composition: {team_composition.team_name}")
            
            # Convert the TeamComposition to a list of agent specifications
            agent_specs = [
                {
                    "role": agent.role,
                    "specialization": agent.specialization,
                    "importance": agent.importance,
                    "description": agent.description,
                    "required_skills": agent.required_skills,
                    "background_info": agent.background_info
                }
                for agent in team_composition.required_agents
            ]
            
            # Store the team composition in the knowledge repository if available
            if self.knowledge_repository:
                self.knowledge_repository.store_team_composition(
                    task_description, 
                    {
                        "team_name": team_composition.team_name,
                        "team_goal": team_composition.team_goal,
                        "additional_context": team_composition.additional_context,
                        "agent_specs": agent_specs
                    }
                )
            
            return agent_specs
            
        except Exception as e:
            logger.error(f"Error parsing team composition: {str(e)}")
            logger.debug(f"Raw LLM response: {response.content}")
            
            # Fallback to a default team composition
            return self._get_default_team_composition(task_description)
    
    def _get_default_team_composition(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Get a default team composition when analysis fails.
        
        Args:
            task_description: Original task description
            
        Returns:
            Default list of agent specifications
        """
        logger.info("Using default team composition")
        
        return [
            {
                "role": "planner",
                "specialization": "project_manager",
                "importance": 10,
                "description": "Coordinates the team and plans the approach",
                "required_skills": ["project management", "task decomposition", "coordination"],
                "background_info": None
            },
            {
                "role": "research",
                "specialization": "information_retrieval",
                "importance": 8,
                "description": "Gathers information related to the task",
                "required_skills": ["web search", "information synthesis", "knowledge retrieval"],
                "background_info": None
            },
            {
                "role": "specialist",
                "specialization": "domain_expert",
                "importance": 9,
                "description": "Provides domain expertise for the task",
                "required_skills": ["domain knowledge", "problem solving", "technical expertise"],
                "background_info": None
            },
            {
                "role": "executor",
                "specialization": "implementation",
                "importance": 7,
                "description": "Implements solutions and executes plans",
                "required_skills": ["coding", "implementation", "technical execution"],
                "background_info": None
            },
            {
                "role": "reviewer",
                "specialization": "quality_assurance",
                "importance": 6,
                "description": "Reviews work and ensures quality",
                "required_skills": ["quality assurance", "testing", "review"],
                "background_info": None
            }
        ]
    
    def create_team(self, agent_specs: List[Dict[str, Any]]) -> Dict[str, BaseAgent]:
        """
        Create a team of agents based on the provided specifications.
        
        Args:
            agent_specs: List of agent specifications
            
        Returns:
            Dictionary mapping agent roles to agent instances
        """
        logger.info(f"Creating team with {len(agent_specs)} agents")
        
        team = {}
        
        # Process agents in order of importance (if specified)
        sorted_specs = sorted(
            agent_specs, 
            key=lambda x: x.get("importance", 5),
            reverse=True
        )
        
        for spec in sorted_specs:
            role = spec.get("role")
            specialization = spec.get("specialization")
            
            # Create a unique identifier for this agent
            agent_id = f"{role}_{specialization}" if specialization else role
            
            # Check if we already have this agent type in the team
            if agent_id in team:
                logger.warning(f"Agent with ID {agent_id} already exists in the team, skipping")
                continue
            
            try:
                # Create the agent using the agent factory
                agent = self.agent_factory.create_specialized_agent(spec)
                
                # Add the agent to the team
                team[agent_id] = agent
                logger.debug(f"Added agent {agent_id} to the team")
                
            except Exception as e:
                logger.error(f"Error creating agent {agent_id}: {str(e)}")
        
        logger.info(f"Team created successfully with {len(team)} agents")
        return team
    
    def get_team_roles(self, team: Dict[str, BaseAgent]) -> List[str]:
        """
        Get the list of roles present in a team.
        
        Args:
            team: Dictionary mapping agent IDs to agent instances
            
        Returns:
            List of roles in the team
        """
        return [agent.role for agent in team.values()]
    
    def update_team(
        self, 
        team: Dict[str, BaseAgent], 
        additional_specs: List[Dict[str, Any]]
    ) -> Dict[str, BaseAgent]:
        """
        Update an existing team with additional agents.
        
        Args:
            team: Existing team of agents
            additional_specs: Specifications for agents to add
            
        Returns:
            Updated team dictionary
        """
        logger.info(f"Updating team with {len(additional_specs)} additional agents")
        
        # Create agents for the additional specifications
        for spec in additional_specs:
            role = spec.get("role")
            specialization = spec.get("specialization")
            
            # Create a unique identifier for this agent
            agent_id = f"{role}_{specialization}" if specialization else role
            
            # Check if we already have this agent type in the team
            if agent_id in team:
                logger.warning(f"Agent with ID {agent_id} already exists in the team, skipping")
                continue
            
            try:
                # Create the agent using the agent factory
                agent = self.agent_factory.create_specialized_agent(spec)
                
                # Add the agent to the team
                team[agent_id] = agent
                logger.debug(f"Added agent {agent_id} to the team")
                
            except Exception as e:
                logger.error(f"Error creating agent {agent_id}: {str(e)}")
        
        return team

    def get_team(self, team_id: str) -> Dict[str, BaseAgent]:
        """
        Get a previously created team by its ID.
        
        Args:
            team_id: ID of the team to retrieve
            
        Returns:
            Dictionary mapping agent IDs to agent instances
        """
        # Récupérer la composition de l'équipe
        team_data = self.knowledge_repository.get_team(team_id)
        
        if not team_data:
            logger.warning(f"Team with ID {team_id} not found")
            return {}
        
        # Créer l'équipe d'agents à partir des spécifications
        agent_specs = team_data.get("agent_specs", [])
        logger.info(f"Recreating team '{team_data.get('name')}' with {len(agent_specs)} agents")
        
        return self.create_team(agent_specs)
