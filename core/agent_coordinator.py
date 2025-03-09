"""
Agent Coordinator Module

This module is responsible for coordinating the workflow between multiple agents,
managing the execution of tasks, and facilitating communication between team members.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agents.base_agent import BaseAgent
from core.knowledge_repository import KnowledgeRepository
from core.task_scheduler import TaskScheduler
from utils.prompt_templates import COORDINATION_PROMPT, TASK_BREAKDOWN_PROMPT, RESULT_SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)

class AgentCoordinator:
    """
    Coordinates the execution of tasks across a team of agents, managing
    the workflow and facilitating communication between agents.
    """
    
    def __init__(self, config: Dict[str, Any], knowledge_repository: Optional[KnowledgeRepository] = None):
        """
        Initialize the agent coordinator.
        
        Args:
            config: Configuration dictionary with coordinator settings
            knowledge_repository: Knowledge repository for accessing shared information
        """
        self.config = config
        self.knowledge_repository = knowledge_repository
        self.coordination_model = config.get("coordination_model", "gpt-4o")
        self.max_coordination_retries = config.get("max_coordination_retries", 3)
        self.task_scheduler = TaskScheduler(config.get("task_scheduler", {}))
        
        # Initialize the coordination LLM
        self.coordination_llm = ChatOpenAI(
            model=self.coordination_model,
            temperature=0.2
        )
        
        logger.debug(f"Initialized AgentCoordinator with model: {self.coordination_model}")
    
    def execute_task(self, task_description: str, agent_team: Dict[str, BaseAgent]) -> Dict[str, Any]:
        """
        Execute a task with a team of agents.
        
        Args:
            task_description: Description of the task to execute
            agent_team: Dictionary mapping agent IDs to agent instances
            
        Returns:
            Dictionary containing the results and metadata
        """
        logger.info(f"Executing task: {task_description}")
        logger.info(f"Team composition: {', '.join(agent_team.keys())}")
        
        # Step 1: Break down the task into subtasks
        subtasks = self._break_down_task(task_description, agent_team)
        logger.info(f"Task broken down into {len(subtasks)} subtasks")
        
        # Step 2: Schedule the subtasks
        task_schedule = self.task_scheduler.create_schedule(subtasks, list(agent_team.keys()))
        logger.info(f"Created task schedule with {len(task_schedule)} steps")
        
        # Step 3: Execute the scheduled tasks
        execution_results = self._execute_scheduled_tasks(task_schedule, agent_team, task_description)
        logger.info("Task execution completed")
        
        # Step 4: Synthesize the results
        final_results = self._synthesize_results(task_description, execution_results, agent_team)
        logger.info("Results synthesized")
        
        return final_results
    
    def _break_down_task(
        self, 
        task_description: str, 
        agent_team: Dict[str, BaseAgent]
    ) -> List[Dict[str, Any]]:
        logger.debug(f"Available agents for task breakdown: {list(agent_team.keys())}")
        logger.debug(f"Agent roles: {[(agent_id, agent.role) for agent_id, agent in agent_team.items()]}")
        """
        Break down a task into subtasks that can be assigned to agents.
        
        Args:
            task_description: The main task description
            agent_team: Dictionary of available agents
            
        Returns:
            List of subtask specifications
        """
        # Check if we have a planner agent in the team
        planner_agent = None
        for agent_id, agent in agent_team.items():
            if agent.role.startswith("planner"):
                planner_agent = agent
                break
        
        subtasks = []
        
        # If we have a planner, use it to break down the task
        if planner_agent:
            logger.debug("Using planner agent to break down task")
            
            # Create a prompt for the planner
            prompt = f"""
            Task Description: {task_description}
            
            As the planning agent, break down this task into subtasks that can be assigned to team members.
            For each subtask, specify:
            1. A clear description
            2. The assigned_agent who should complete it (use one of: {', '.join(agent_team.keys())})
            3. Estimated complexity (low, medium, high)
            4. Any dependencies on other subtasks
            
            Available team members and their roles:
            {', '.join([f"{agent_id} ({agent.role})" for agent_id, agent in agent_team.items()])}
            
            Format your response as a list of JSON objects, one per subtask.
            """
            
            # Ask the planner to break down the task
            planner_response = planner_agent.execute_task(prompt)
            
            try:
                # Try to parse the planner's response as a list of subtasks
                import json
                parsed_response = planner_response.get("output", "")
                
                # Find JSON content in the response
                import re
                json_match = re.search(r'\[[\s\S]*\]', parsed_response)
                if json_match:
                    json_content = json_match.group(0)
                    subtasks = json.loads(json_content)
                    logger.debug(f"Successfully parsed {len(subtasks)} subtasks from planner")
                else:
                    raise ValueError("No JSON list found in planner response")
                
            except Exception as e:
                logger.error(f"Error parsing planner response: {str(e)}")
                logger.debug(f"Planner response: {planner_response}")
                # Fall back to LLM-based task breakdown
                subtasks = self._llm_task_breakdown(task_description, agent_team)
        else:
            # No planner, use LLM to break down the task
            logger.debug("No planner agent available, using LLM for task breakdown")
            subtasks = self._llm_task_breakdown(task_description, agent_team)
        
        return subtasks
    
    def _llm_task_breakdown(
        self, 
        task_description: str, 
        agent_team: Dict[str, BaseAgent]
    ) -> List[Dict[str, Any]]:
        """
        Use an LLM to break down a task into subtasks.
        
        Args:
            task_description: The main task description
            agent_team: Dictionary of available agents
            
        Returns:
            List of subtask specifications
        """
        # Create the prompt for task breakdown
        prompt = ChatPromptTemplate.from_template(TASK_BREAKDOWN_PROMPT)
        
        # Format the prompt with task description and team info
        formatted_prompt = prompt.format(
            task_description=task_description,
            available_agents=", ".join([f"{agent_id} ({agent.role})" for agent_id, agent in agent_team.items()])
        )
        
        # Get response from the LLM
        response = self.coordination_llm.invoke(formatted_prompt)
        
        try:
            # Try to parse the response as a list of subtasks
            import json
            import re
            
            # Find JSON content in the response
            json_match = re.search(r'\[[\s\S]*\]', response.content)
            if json_match:
                json_content = json_match.group(0)
                subtasks = json.loads(json_content)
                logger.debug(f"Successfully parsed {len(subtasks)} subtasks from LLM")
                return subtasks
            else:
                raise ValueError("No JSON list found in LLM response")
            
        except Exception as e:
            logger.error(f"Error parsing LLM task breakdown: {str(e)}")
            logger.debug(f"LLM response: {response.content}")
            
            # Return a simplified default task breakdown
            return self._create_default_subtasks(task_description, agent_team)
    
    def _create_default_subtasks(
        self, 
        task_description: str, 
        agent_team: Dict[str, BaseAgent]
    ) -> List[Dict[str, Any]]:
        """
        Create a default set of subtasks when breakdown fails.
        
        Args:
            task_description: The main task description
            agent_team: Dictionary of available agents
            
        Returns:
            List of default subtask specifications
        """
        logger.info("Creating default subtasks")
        
        # Map of roles to default subtasks
        role_subtasks = {
            "research": {
                "description": "Research and gather information related to the task",
                "assigned_agent": "",
                "complexity": "medium",
                "dependencies": []
            },
            "planner": {
                "description": "Create a detailed plan for completing the task",
                "assigned_agent": "",
                "complexity": "medium",
                "dependencies": ["research"]
            },
            "specialist": {
                "description": "Apply domain expertise to solve core problems",
                "assigned_agent": "",
                "complexity": "high",
                "dependencies": ["planner"]
            },
            "executor": {
                "description": "Implement the solution based on the plan",
                "assigned_agent": "",
                "complexity": "high",
                "dependencies": ["specialist"]
            },
            "reviewer": {
                "description": "Review and validate the implemented solution",
                "assigned_agent": "",
                "complexity": "medium",
                "dependencies": ["executor"]
            }
        }
        
        # Create subtasks based on available agent roles
        subtasks = []
        for agent_id, agent in agent_team.items():
            # Get the base role (before any specialization)
            base_role = agent.role.split("_")[0] if "_" in agent.role else agent.role
            
            if base_role in role_subtasks:
                subtask = role_subtasks[base_role].copy()
                subtask["assigned_agent"] = agent_id
                subtasks.append(subtask)
        
        # Sort subtasks based on dependencies
        return subtasks
    
    def _execute_scheduled_tasks(
        self, 
        task_schedule: List[Dict[str, Any]], 
        agent_team: Dict[str, BaseAgent],
        task_description: str
    ) -> Dict[str, Any]:
        """
        Execute the scheduled tasks with the agent team.
        
        Args:
            task_schedule: List of scheduled tasks to execute
            agent_team: Dictionary of available agents
            task_description: Original task description
            
        Returns:
            Dictionary mapping subtask IDs to execution results
        """
        execution_results = {}
        conversation_history = []
        
        # Add the initial task description to the conversation history
        conversation_history.append(
            HumanMessage(content=f"Main task: {task_description}")
        )
        
        # Add debug logging for available agents and their roles
        logger.debug(f"Available agents: {list(agent_team.keys())}")
        logger.debug(f"Agent roles: {[(agent_id, agent.role) for agent_id, agent in agent_team.items()]}")

        # Execute each task in the schedule
        for task_step in task_schedule:
            step_id = task_step.get("step_id", "unknown")
            subtasks = task_step.get("subtasks", [])
            
            logger.info(f"Executing step {step_id} with {len(subtasks)} subtasks")
            
            # Process each subtask in this step (these can be executed in parallel)
            for subtask in subtasks:
                subtask_id = subtask.get("id", "unknown")
                
                # Add debug logging for processing subtask
                logger.debug(f"Processing subtask {subtask_id}: {subtask}")
                
                # MODIFICATION: Vérifier plusieurs champs pour trouver l'agent assigné
                agent_id = None
                
                # Champs possibles pour l'assignation d'agent
                possible_fields = ["assigned_agent", "required_skills_or_role", "role"]
                
                # Vérifier chaque champ possible
                for field in possible_fields:
                    if field in subtask and subtask[field]:
                        potential_id = subtask[field]
                        
                        # Vérifier si c'est directement un ID d'agent
                        if potential_id in agent_team:
                            agent_id = potential_id
                            break
                        
                        # Sinon, chercher un agent par son rôle
                        for ag_id, agent in agent_team.items():
                            if agent.role == potential_id:
                                agent_id = ag_id
                                break
                        
                        # Si on a trouvé un agent, sortir de la boucle
                        if agent_id:
                            break
                
                description = subtask.get("description", "No description provided")
                
                # Skip if no agent is assigned
                if not agent_id or agent_id not in agent_team:
                    logger.warning(f"No valid agent assigned for subtask {subtask_id}, skipping")
                    continue
                
                # Get the assigned agent
                agent = agent_team[agent_id]
                
                # Prepare the context for this subtask
                context = self._prepare_subtask_context(
                    subtask, 
                    execution_results, 
                    conversation_history,
                    task_description
                )
                
                logger.info(f"Executing subtask {subtask_id} with agent {agent_id}")
                
                # Execute the subtask with the agent
                try:
                    result = agent.execute_task(context)
                    
                    # Store the result
                    execution_results[subtask_id] = {
                        "subtask": subtask,
                        "agent_id": agent_id,
                        "output": result.get("output", ""),
                        "status": "completed",
                        "metadata": result.get("metadata", {})
                    }
                    
                    # Add to conversation history
                    conversation_history.append(
                        SystemMessage(content=f"Agent {agent_id} completed subtask: {description}")
                    )
                    conversation_history.append(
                        AIMessage(content=result.get("output", ""))
                    )
                    
                    logger.debug(f"Subtask {subtask_id} completed successfully")
                    
                except Exception as e:
                    logger.error(f"Error executing subtask {subtask_id}: {str(e)}")
                    
                    # Store the error result
                    execution_results[subtask_id] = {
                        "subtask": subtask,
                        "agent_id": agent_id,
                        "output": f"Error: {str(e)}",
                        "status": "failed",
                        "metadata": {"error": str(e)}
                    }
                    
                    # Add to conversation history
                    conversation_history.append(
                        SystemMessage(content=f"Agent {agent_id} failed subtask: {description}")
                    )
                    conversation_history.append(
                        AIMessage(content=f"Error: {str(e)}")
                    )
            
            # Brief pause between steps to avoid rate limiting
            time.sleep(0.5)
        
        # Store the execution results in the knowledge repository if available
        if self.knowledge_repository:
            self.knowledge_repository.store_execution_results(
                task_description, 
                execution_results,
                conversation_history
            )
        
        return {
            "execution_results": execution_results,
            "conversation_history": conversation_history
        }
    
    def _prepare_subtask_context(
        self, 
        subtask: Dict[str, Any], 
        results_so_far: Dict[str, Any],
        conversation_history: List[Any],
        task_description: str
    ) -> str:
        """
        Prepare the context for a subtask execution.
        
        Args:
            subtask: The subtask specification
            results_so_far: Results from previously executed subtasks
            conversation_history: History of the conversation so far
            task_description: Original task description
            
        Returns:
            Context string for the agent
        """
        # Start with the subtask description
        context_parts = [
            f"Main task: {task_description}",
            f"Your subtask: {subtask.get('description', 'No description provided')}"
        ]
        
        # Add dependency results if any
        dependencies = subtask.get("dependencies", [])
        if dependencies:
            context_parts.append("\nRelevant information from dependent tasks:")
            
            for dep_id in dependencies:
                if dep_id in results_so_far:
                    dep_result = results_so_far[dep_id]
                    agent_id = dep_result.get("agent_id", "unknown")
                    output = dep_result.get("output", "No output")
                    
                    context_parts.append(f"\nFrom {agent_id}:")
                    context_parts.append(output)
        
        # Add a request for specific output
        context_parts.append("\nPlease complete this subtask and provide your results.")
        
        return "\n\n".join(context_parts)
    
    def _synthesize_results(
        self, 
        task_description: str, 
        execution_data: Dict[str, Any], 
        agent_team: Dict[str, BaseAgent]
    ) -> Dict[str, Any]:
        """
        Synthesize the execution results into a coherent final result.
        
        Args:
            task_description: Original task description
            execution_data: Data from the task execution
            agent_team: Dictionary of available agents
            
        Returns:
            Synthesized results
        """
        # Extract execution results and conversation history
        execution_results = execution_data.get("execution_results", {})
        conversation_history = execution_data.get("conversation_history", [])
        
        # Check if we have a reviewer agent in the team
        reviewer_agent = None
        for agent_id, agent in agent_team.items():
            if agent.role.startswith("reviewer"):
                reviewer_agent = agent
                break
        
        # If we have a reviewer, use it to synthesize results
        if reviewer_agent:
            logger.debug("Using reviewer agent to synthesize results")
            
            # Create a summary of all results
            results_summary = []
            for subtask_id, result in execution_results.items():
                agent_id = result.get("agent_id", "unknown")
                subtask_desc = result.get("subtask", {}).get("description", "Unknown subtask")
                status = result.get("status", "unknown")
                output = result.get("output", "No output")
                
                results_summary.append(f"Subtask: {subtask_desc}")
                results_summary.append(f"Executed by: {agent_id}")
                results_summary.append(f"Status: {status}")
                results_summary.append(f"Output: {output}\n")
            
            # Create a prompt for the reviewer
            review_prompt = f"""
            Task Description: {task_description}
            
            Below are the results from all team members who worked on this task.
            Please review these results and create:
            1. A comprehensive summary of the work done
            2. An assessment of the quality and completeness
            3. A final deliverable that combines the best parts of everyone's work
            
            Results:
            {''.join(results_summary)}
            
            Your synthesis should be well-structured and ready for delivery to the user.
            """
            
            # Ask the reviewer to synthesize the results
            review_result = reviewer_agent.execute_task(review_prompt)
            synthesis = review_result.get("output", "")
            
        else:
            # No reviewer, use LLM to synthesize results
            logger.debug("No reviewer agent available, using LLM for synthesis")
            
            # Create the prompt for result synthesis
            prompt = ChatPromptTemplate.from_template(RESULT_SYNTHESIS_PROMPT)
            
            # Format the prompt with task description and results
            results_text = ""
            for subtask_id, result in execution_results.items():
                agent_id = result.get("agent_id", "unknown")
                subtask_desc = result.get("subtask", {}).get("description", "Unknown subtask")
                status = result.get("status", "unknown")
                output = result.get("output", "No output")
                
                results_text += f"Subtask: {subtask_desc}\n"
                results_text += f"Executed by: {agent_id}\n"
                results_text += f"Status: {status}\n"
                results_text += f"Output: {output}\n\n"
            
            formatted_prompt = prompt.format(
                task_description=task_description,
                execution_results=results_text
            )
            
            # Get response from the LLM
            response = self.coordination_llm.invoke(formatted_prompt)
            synthesis = response.content
        
        # Extract individual agent contributions
        agent_contributions = {}
        for subtask_id, result in execution_results.items():
            agent_id = result.get("agent_id", "unknown")
            if agent_id not in agent_contributions:
                agent_contributions[agent_id] = []
            
            subtask_desc = result.get("subtask", {}).get("description", "Unknown subtask")
            output = result.get("output", "No output")
            
            agent_contributions[agent_id].append(f"Subtask: {subtask_desc}\nOutput: {output}")
        
        # Combine contributions for each agent
        for agent_id, contributions in agent_contributions.items():
            agent_contributions[agent_id] = "\n\n".join(contributions)
        
        # Check for any output files
        output_files = []
        for result in execution_results.values():
            metadata = result.get("metadata", {})
            if "output_files" in metadata and metadata["output_files"]:
                output_files.extend(metadata["output_files"])
        
        # Create the final result structure
        final_results = {
            "summary": synthesis,
            "agent_contributions": agent_contributions,
            "execution_results": execution_results,
            "output_files": output_files
        }
        
        return final_results
    
    def get_agent_contributions(self, execution_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract contributions from each agent from the execution results.
        
        Args:
            execution_results: Results from task execution
            
        Returns:
            Dictionary mapping agent IDs to their contributions
        """
        agent_contributions = {}
        
        for subtask_id, result in execution_results.items():
            agent_id = result.get("agent_id", "unknown")
            if agent_id not in agent_contributions:
                agent_contributions[agent_id] = []
            
            subtask_desc = result.get("subtask", {}).get("description", "Unknown subtask")
            output = result.get("output", "No output")
            
            agent_contributions[agent_id].append(f"Subtask: {subtask_desc}\nOutput: {output}")
        
        # Combine contributions for each agent
        for agent_id, contributions in agent_contributions.items():
            agent_contributions[agent_id] = "\n\n".join(contributions)
        
        return agent_contributions
