"""
Evolution Workflow Module

This module implements a workflow for code evolution, helping developers make 
systematic and incremental changes to a codebase with the assistance of AI agents.
"""

import logging
import os
import time
import json
import sys  # Pour l'interaction avec stdin/stdout
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from core.team_manager import TeamManager
from core.agent_coordinator import AgentCoordinator
from core.knowledge_repository import KnowledgeRepository
from tools.code_indexer_tool import CodeIndexerTool
from tools.file_manager_tool import FileManagerTool

logger = logging.getLogger(__name__)

class EvolutionWorkflow:
    """
    Workflow for evolving code with the assistance of AI agents.
    
    This class manages the process of analyzing existing code,
    planning changes, implementing them, and validating the results.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        team_manager: TeamManager,
        agent_coordinator: AgentCoordinator,
        knowledge_repository: KnowledgeRepository
    ):
        """
        Initialize the evolution workflow.
        
        Args:
            config: Configuration dictionary
            team_manager: Instance of TeamManager to manage agent teams
            agent_coordinator: Instance of AgentCoordinator to coordinate agent tasks
            knowledge_repository: Instance of KnowledgeRepository to store knowledge
        """
        self.config = config
        self.team_manager = team_manager
        self.agent_coordinator = agent_coordinator
        self.knowledge_repository = knowledge_repository
        self.code_base_dir = config.get("code_base_dir", ".")
        self.output_dir = config.get("output_dir", "output")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize tools
        self.code_indexer = CodeIndexerTool(
            config.get("code_indexer", {}),
            knowledge_repository
        )
        self.file_manager = FileManagerTool(
            config.get("file_manager", {})
        )
        
        logger.debug("Initialized EvolutionWorkflow")
    
    def execute_evolution(self, evolution_request: str, team_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute code evolution based on an evolution request.
        
        Args:
            evolution_request: Description of the requested evolution
            team_id: Optional ID of a pre-configured team to use
            
        Returns:
            Dictionary with evolution results
        """
        return self.evolve_code(evolution_request, team_id)
    
    def evolve_code(self, evolution_request: str, team_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Evolve code based on an evolution request.
        
        Args:
            evolution_request: Description of the requested evolution
            team_id: Optional ID of a pre-configured team to use
            
        Returns:
            Dictionary with evolution results
        """
        # Start by indexing the code base for context
        self._ensure_code_indexed()
        
        # Format the task description for the team
        task_description = self._format_evolution_task(evolution_request)
        
        # Create or get the agent team
        if team_id:
            agent_team = self.team_manager.get_team(team_id)
            if not agent_team:
                logger.error(f"Team with ID {team_id} not found")
                return {"error": f"Team with ID {team_id} not found"}
        else:
            # Analyze task and determine required team composition
            team_composition = self.team_manager.analyze_task(task_description)
            
            # Create the team of agents
            agent_team = self.team_manager.create_team(team_composition)
        
        # Execute the evolution task with the team
        results = self.agent_coordinator.execute_task(task_description, agent_team)
        
        # Extract and process generated files
        generated_files = self._extract_and_save_files(results)
        
        # Return the results
        return {
            "task_description": task_description,
            "summary": results.get("summary", "No summary available"),
            "generated_files": generated_files
        }
    
    def _ensure_code_indexed(self):
        """Ensure the code base is indexed for context."""
        # Check if the code is already indexed in the knowledge repository
        indexed = False
        
        try:
            # Arbitrary search to check if code is indexed
            results = self.knowledge_repository.search_knowledge(
                "code_file",
                k=1,
                filter_metadata={"type": "code_file"}
            )
            
            indexed = len(results) > 0
        except Exception:
            indexed = False
        
        if indexed:
            logger.info("Base de code déjà indexée, utilisation de l'index existant")
        else:
            logger.info("Indexation de la base de code...")
            try:
                self.code_indexer.index_codebase(self.code_base_dir)
                logger.info("Indexation terminée")
            except Exception as e:
                logger.error(f"Erreur lors de l'indexation: {str(e)}")
    
    def _format_evolution_task(self, evolution_request: str) -> str:
        """
        Format the evolution request into a structured task description.
        
        Args:
            evolution_request: The raw evolution request
            
        Returns:
            Formatted task description
        """
        return f"""
        Analyser et implémenter l'évolution suivante du code:

        {evolution_request}

        Suivre ces étapes:
        1. Rechercher et comprendre le contexte de cette demande d'évolution
        2. Analyser la base de code pour identifier les fichiers à modifier
        3. Planifier les modifications nécessaires en détail
        4. Implémenter les modifications de code requises
        5. Vérifier et valider les modifications
        6. Mettre à jour la documentation si nécessaire
        """
    
    def _extract_and_save_files(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract and save files generated during evolution.
        
        Args:
            results: Results from agent execution
            
        Returns:
            List of information about generated files
        """
        generated_files = []
        output_dir = self.output_dir
        
        # Extract from agent contributions
        extracted_files = []
        agent_contributions = results.get("agent_contributions", {})
        
        for agent_id, contribution in agent_contributions.items():
            files_info = self._extract_files_from_contribution(contribution)
            for file_info in files_info:
                file_info["agent"] = agent_id
                
                # Transform the destination path
                file_info["app_path"] = self._transform_destination_path(file_info["relative_path"])
                extracted_files.append(file_info)
        
        # Also check any files from output_files that may have been generated by tools
        output_files = results.get("output_files", [])
        for file_path in output_files:
            if os.path.exists(file_path):
                file_type = os.path.splitext(file_path)[1][1:]  # Get extension without the dot
                relative_path = os.path.relpath(file_path, self.output_dir)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                app_path = self._transform_destination_path(relative_path)
                extracted_files.append({
                    "relative_path": relative_path,
                    "app_path": app_path,
                    "content": content,
                    "type": file_type,
                    "agent": "tools"
                })
        
        # Get user approval for each file
        approved_files = self._get_user_approval(extracted_files)
        
        # Write approved files to disk
        for file_info in approved_files:
            # Use the transformed path for the application
            file_path = file_info["app_path"]
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write the file
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(file_info["content"])
                generated_files.append({
                    "path": file_path,
                    "type": file_info["type"],
                    "agent": file_info["agent"],
                    "status": "created"
                })
                logger.debug(f"Fichier généré: {file_path}")
            except Exception as e:
                logger.error(f"Erreur lors de l'écriture du fichier {file_path}: {str(e)}")
        
        if not generated_files:
            logger.info("Aucun fichier n'a été généré pendant l'évolution")
        else:
            logger.info(f"Nombre de fichiers générés: {len(generated_files)}")
            
        return generated_files

    def _transform_destination_path(self, relative_path: str) -> str:
        """
        Transform the destination path based on file type and purpose.
        
        Args:
            relative_path: Original relative path
            
        Returns:
            Transformed path appropriate for the application
        """
        # Analyze the original path
        path_parts = relative_path.split('/')
        filename = path_parts[-1]
        extension = os.path.splitext(filename)[1].lower()
        
        # Determine the file type and its usage
        if extension == '.py':
            # Python files: check if it's a tool
            if 'tools/' in relative_path or path_parts[0] == 'tools':
                # If it's a file in tools/*, place it in the real tools/ folder
                return os.path.join('tools', filename)
            elif 'agents/' in relative_path or path_parts[0] == 'agents':
                # If it's a file in agents/*, place it in the real agents/ folder
                return os.path.join('agents', filename)
            elif 'core/' in relative_path or path_parts[0] == 'core':
                # If it's a file in core/*, place it in the real core/ folder
                return os.path.join('core', filename)
            else:
                # Other Python scripts at the root
                return filename
        
        # Documentation and other files (markdown, text, etc.)
        elif extension in ['.md', '.txt', '.json', '.yaml', '.yml']:
            # Keep these files in the output folder
            if path_parts[0] == 'output':
                return relative_path
            else:
                return os.path.join(self.output_dir, filename)
        
        # Default: keep the original path
        return relative_path

    def _get_user_approval(self, extracted_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get user approval for each extracted file.
        
        Args:
            extracted_files: List of extracted file information
            
        Returns:
            List of approved file information
        """
        if not extracted_files:
            print("Aucun fichier n'a été proposé par les agents.")
            return []
        
        approved_files = []
        
        print("\n--- Modifications proposées ---")
        for i, file_info in enumerate(extracted_files, 1):
            relative_path = file_info["relative_path"]
            app_path = file_info["app_path"]
            file_type = file_info["type"]
            content_size = len(file_info["content"])
            agent = file_info["agent"]
            
            # Determine if it's a documentation or code file
            is_doc = app_path.startswith(self.output_dir)
            file_purpose = "Documentation" if is_doc else "Code fonctionnel"
            
            print(f"{i}. [NOUVEAU] {app_path} ({content_size / 1024:.1f}KB)")
            print(f"   Type: {file_type}, Catégorie: {file_purpose}, Proposé par: {agent}")
            
            while True:
                choice = input(f"   Approuver cette modification? (o/n) > ").lower()
                if choice in ['o', 'oui', 'y', 'yes']:
                    approved_files.append(file_info)
                    print(f"   ✓ Modification approuvée")
                    break
                elif choice in ['n', 'non', 'no']:
                    print(f"   ✗ Modification rejetée")
                    break
                else:
                    print("   Réponse non reconnue. Veuillez répondre par 'o' (oui) ou 'n' (non).")
        
        # Display a summary
        total_approved = len(approved_files)
        total_rejected = len(extracted_files) - total_approved
        
        print("\n--- Résumé des actions ---")
        print(f"{total_approved} modifications approuvées, {total_rejected} rejetées")
        
        if approved_files:
            print("Modifications à appliquer:")
            for file_info in approved_files:
                print(f"- {file_info['app_path']}")
        
        return approved_files
    
    def _extract_files_from_contribution(self, contribution: str) -> List[Dict[str, Any]]:
        """
        Extract file information from agent contribution text.
        
        Args:
            contribution: Text contribution from an agent
            
        Returns:
            List of dictionaries with file information
        """
        files_info = []
        
        # Patterns to detect Markdown file declaration
        markdown_patterns = [
            r"# ([a-zA-Z0-9_\-\.]+\.md)",  # Markdown headers that look like filenames
            r"```markdown\s*\n(.*?)\n```",  # Markdown code blocks
            r"(\btop_10_langchain_tools\.md\b)",  # Specific file mentioned
            r"(\bREADME\.md\b)"  # README.md file
        ]
        
        # Patterns to detect Python file declaration
        python_patterns = [
            r"```python\s*\n# (output/tools/[a-zA-Z0-9_]+\.py)",  # Python file header comments
            r"# (output/tools/[a-zA-Z0-9_]+\.py)",  # Python file header outside code blocks
            r"output/tools/([a-zA-Z0-9_]+)\.py"  # Python file paths
        ]
        
        # Check for our specific target files first
        if "top_10_langchain_tools.md" in contribution:
            # Try to extract the content between markdown blocks
            import re
            markdown_content = ""
            
            # Look for markdown blocks
            md_matches = re.findall(r"```markdown\s*\n(.*?)\n```", contribution, re.DOTALL)
            if md_matches:
                for match in md_matches:
                    if "# Top 10 Langchain Tools" in match:
                        markdown_content = match
                        break
            
            if not markdown_content:
                # Try another approach - find sections between specified strings
                sections = contribution.split("\n\n")
                for i, section in enumerate(sections):
                    if "top_10_langchain_tools.md" in section:
                        # Try to find the content in the next sections
                        for j in range(i+1, min(i+5, len(sections))):
                            if j < len(sections) and "# Top 10 Langchain Tools" in sections[j]:
                                markdown_content = sections[j]
                                break
            
            if markdown_content:
                files_info.append({
                    "relative_path": "top_10_langchain_tools.md",
                    "content": markdown_content,
                    "type": "markdown"
                })
        
        # Look for README.md
        if "README.md" in contribution:
            import re
            readme_content = ""
            
            # Look for markdown blocks that contain README content
            md_matches = re.findall(r"```markdown\s*\n(.*?)\n```", contribution, re.DOTALL)
            if md_matches:
                for match in md_matches:
                    if "# Integrating Langchain Tools" in match:
                        readme_content = match
                        break
            
            if not readme_content:
                # Try another approach - find sections between specified strings
                start_marker = "Here's the content for the `README.md`:"
                if start_marker in contribution:
                    start_idx = contribution.find(start_marker) + len(start_marker)
                    end_idx = contribution.find("```", start_idx + 1)
                    if end_idx > start_idx:
                        content_between = contribution[start_idx:end_idx].strip()
                        if content_between:
                            readme_content = content_between
            
            if readme_content:
                files_info.append({
                    "relative_path": "README.md",
                    "content": readme_content,
                    "type": "markdown"
                })
        
        # Look for Python tool implementations
        python_files = {}
        import re
        
        # Find Python code blocks
        py_blocks = re.findall(r"```python\s*\n(.*?)\n```", contribution, re.DOTALL)
        for block in py_blocks:
            # Look for class definitions or file comments
            for prefix in ["class ", "# output/tools/"]:
                if prefix in block:
                    lines = block.split("\n")
                    for line in lines:
                        if prefix == "class " and line.strip().startswith(prefix):
                            # Extract class name
                            class_name = line.strip()[len(prefix):].split("(")[0].split(":")[0].strip()
                            python_files[class_name] = block
                        elif prefix == "# output/tools/" and line.strip().startswith(prefix):
                            # Extract filename
                            filename = line.strip()[len(prefix):].strip()
                            class_name = os.path.splitext(filename)[0]
                            python_files[class_name] = block
        
        # Create file info for each Python file
        for class_name, content in python_files.items():
            os.makedirs(os.path.join(self.output_dir, "tools"), exist_ok=True)
            files_info.append({
                "relative_path": f"tools/{class_name}.py",
                "content": content,
                "type": "python"
            })
        
        return files_info