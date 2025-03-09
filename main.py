#!/usr/bin/env python3
"""
Main entry point for the Team Agents application.
This script handles user input and orchestrates the creation and execution of agent teams.
"""

import os
import argparse
import logging
from typing import Dict, Any, List, Optional
import patch_agents

from dotenv import load_dotenv

# Core components
from core.team_manager import TeamManager
from core.agent_coordinator import AgentCoordinator
from core.knowledge_repository import KnowledgeRepository

# Utils
from utils.config import Config
from utils.logging_utils import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create and manage dynamic teams of AI agents.")
    
    # Ajouter un sous-parseur pour les différentes commandes
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Commande 'task' pour traiter une tâche générale (comportement par défaut)
    task_parser = subparsers.add_parser("task", help="Process a general task with agent team")
    task_parser.add_argument("description", type=str, help="The task description for the agent team")
    
    # Commande 'evolve' pour le workflow d'évolution du code
    evolve_parser = subparsers.add_parser("evolve", help="Evolve the application code")
    evolve_parser.add_argument("description", type=str, help="Description of the evolution request")
    
    # Arguments communs
    for subparser in [task_parser, evolve_parser]:
        subparser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
        subparser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
        subparser.add_argument("--output", "-o", type=str, default="output", help="Output directory for generated files")
        subparser.add_argument("--use-team", type=str, help="ID of an existing team to use")
    
    # Autres arguments généraux
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--list-teams", action="store_true", help="List all available teams")
    
    # Pour maintenir la compatibilité avec l'ancienne interface
    parser.add_argument("task", type=str, nargs="?", help="The task description for the agent team")
    
    return parser.parse_args()


def initialize_system(config_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Initialize the system components based on configuration.
    
    Args:
        config_path: Path to the configuration file
        verbose: Whether to enable verbose logging
    
    Returns:
        Dictionary containing initialized system components
    """
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    logger.info("Initializing Team Agents system...")
    
    # Load configuration
    config = Config(config_path)
    logger.debug(f"Loaded configuration from {config_path}")
    
    # Create tools manually
    from langchain.tools.base import Tool
    tools = [
        Tool(
            name="empty_tool",
            description="A placeholder tool that does nothing",
            func=lambda x: "This tool does nothing"
        )
    ]
    
    # Initialize core components
    knowledge_repo = KnowledgeRepository(config.get("knowledge_repository", {}))
    
    # Modify the team_manager config to include tools
    team_manager_config = config.get("team_manager", {})
    if "agent_factory" not in team_manager_config:
        team_manager_config["agent_factory"] = {}
    
    # Add tools directly to agent_factory config
    team_manager_config["agent_factory"]["tools"] = tools
    
    team_manager = TeamManager(team_manager_config, knowledge_repo)
    agent_coordinator = AgentCoordinator(config.get("agent_coordinator", {}), knowledge_repo)
    
    return {
        "config": config,
        "knowledge_repository": knowledge_repo,
        "team_manager": team_manager,
        "agent_coordinator": agent_coordinator,
        "logger": logger
    }


def process_task(task: str, system_components: Dict[str, Any], team_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a task by creating and executing an agent team.
    
    Args:
        task: The task description
        system_components: Dictionary containing system components
        team_id: Optional ID of an existing team to use
    
    Returns:
        Dictionary containing the results
    """
    logger = system_components["logger"]
    team_manager = system_components["team_manager"]
    agent_coordinator = system_components["agent_coordinator"]
    
    logger.info(f"Processing task: {task}")
    
    # Utiliser une équipe existante ou en créer une nouvelle
    if team_id:
        agent_team = team_manager.get_team(team_id)
        if not agent_team:
            logger.error(f"Team with ID {team_id} not found")
            return {"error": f"Team with ID {team_id} not found"}
        logger.info(f"Using existing team with ID {team_id}")
    else:
        # Analyze task and determine required team composition
        team_composition = team_manager.analyze_task(task)
        logger.info(f"Determined team composition: {', '.join([agent['role'] for agent in team_composition])}")
        
        # Create the team of agents
        agent_team = team_manager.create_team(team_composition)
        logger.info(f"Created agent team with {len(agent_team)} members")
    
    # Execute the task with the team
    results = agent_coordinator.execute_task(task, agent_team)
    logger.info("Task execution completed")
    
    return results

def process_evolution(description: str, system_components: Dict[str, Any], team_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process an evolution request by analyzing and modifying the application code.
    
    Args:
        description: The evolution request description
        system_components: Dictionary containing system components
        team_id: Optional ID of an existing team to use
    
    Returns:
        Dictionary containing the results
    """
    logger = system_components["logger"]
    
    # Importer le workflow d'évolution
    from core.evolution_workflow import EvolutionWorkflow
    
    # Créer l'instance du workflow d'évolution
    evolution_workflow = EvolutionWorkflow(
        config=system_components["config"].to_dict(),
        team_manager=system_components["team_manager"],
        agent_coordinator=system_components["agent_coordinator"],
        knowledge_repository=system_components["knowledge_repository"]
    )
    
    logger.info(f"Processing evolution request: {description}")
    
    # Exécuter le workflow d'évolution
    results = evolution_workflow.execute_evolution(description, team_id)
    
    logger.info("Evolution process completed")
    
    return results


def interactive_mode(system_components: Dict[str, Any]) -> None:
    """
    Run the system in interactive mode, accepting user input continuously.
    
    Args:
        system_components: Dictionary containing system components
    """
    logger = system_components["logger"]
    logger.info("Starting interactive mode. Type 'exit' to quit.")
    
    while True:
        task = input("\nEnter your task (or 'exit' to quit): ")
        if task.lower() == 'exit':
            logger.info("Exiting interactive mode")
            break
            
        try:
            results = process_task(task, system_components)
            print("\n--- Results ---")
            print(results.get("summary", "No summary available"))
            
            # Display agent contributions if available
            if "agent_contributions" in results:
                print("\n--- Agent Contributions ---")
                for agent, contribution in results["agent_contributions"].items():
                    print(f"\n{agent}:")
                    print(contribution)
                    
            # Handle any output files
            if "output_files" in results and results["output_files"]:
                print("\n--- Generated Files ---")
                for file_path in results["output_files"]:
                    print(f"- {file_path}")
        
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            print(f"An error occurred: {str(e)}")


def list_available_teams(system_components: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    List all available teams in the knowledge repository.
    
    Args:
        system_components: Dictionary containing system components
    
    Returns:
        List of team information dictionaries
    """
    knowledge_repository = system_components["knowledge_repository"]
    
    # Filtrer uniquement les team_compositions
    teams = []
    recent_tasks = knowledge_repository.list_recent_tasks(limit=100)
    
    for task in recent_tasks:
        if task["type"] == "team_composition":
            team_id = task["task_id"]
            team_data = knowledge_repository.get_team_composition(team_id)
            
            if team_data:
                teams.append({
                    "id": team_id,
                    "name": team_data.get("team_name", "Unnamed Team"),
                    "description": team_data.get("team_goal", "No description"),
                    "created_at": team_data.get("timestamp", "Unknown"),
                    "agent_specs": team_data.get("agent_specs", [])
                })
    
    return teams


def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Initialize system components
    system_components = initialize_system(args.config, args.verbose)
    logger = system_components["logger"]
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        logger.debug(f"Created output directory: {args.output}")
    
    # Set output directory in system components
    system_components["output_dir"] = args.output
    
    try:
        if args.list_teams:
            # Afficher toutes les équipes disponibles
            knowledge_repo = system_components["knowledge_repository"]
            teams = knowledge_repo.get_all_teams()
            
            if not teams:
                print("No teams found.")
                return
            
            print("\n--- Available Teams ---")
            for team_id, team in teams.items():
                print(f"ID: {team_id}")
                print(f"Name: {team['name']}")
                print(f"Description: {team['description']}")
                print(f"Created: {team['created_at']}")
                print(f"Agents: {len(team['agent_specs'])}")
                print("---")
            return

        if args.interactive:
            # Run in interactive mode
            interactive_mode(system_components)
        elif args.command == "evolve":
            # Process evolution request
            results = process_evolution(args.description, system_components, args.use_team)
            
            # Display results
            print("\n--- Evolution Results ---")
            print(results.get("summary", "No summary available"))
            
            if "output_files" in results and results["output_files"]:
                print("\n--- Generated Files ---")
                for file_path in results["output_files"]:
                    print(f"- {file_path}")
        elif args.command == "task" or args.task:
            # Process a single task (maintain backward compatibility)
            task_description = args.description if args.command == "task" else args.task
            
            if args.use_team:
                results = process_task(task_description, system_components, args.use_team)
            else:
                results = process_task(task_description, system_components)
            
            # Display results
            print("\n--- Results ---")
            print(results.get("summary", "No summary available"))
            
            if "output_files" in results and results["output_files"]:
                print("\n--- Generated Files ---")
                for file_path in results["output_files"]:
                    print(f"- {file_path}")
        else:
            # No command or task provided, show help
            print("No task provided. Use --interactive mode or provide a task description.")
            print("Example: python main.py task 'Create a team of Salesforce developers'")
            print("Example: python main.py evolve 'Add support for MySQL database'")
    
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        print(f"An error occurred: {str(e)}")
    
    logger.info("Application shutting down")


if __name__ == "__main__":
    main()