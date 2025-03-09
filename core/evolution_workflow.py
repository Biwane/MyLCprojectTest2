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
import datetime

from core.team_manager import TeamManager
from core.agent_coordinator import AgentCoordinator
from core.knowledge_repository import KnowledgeRepository
from tools.code_indexer_tool import CodeIndexerTool
from tools.file_manager_tool import FileManagerTool
from services.indexing_service import IndexingService

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
        indexing_service = IndexingService.get_instance()
        if indexing_service:
            # Ne fait rien, car l'indexation a déjà été effectuée dans main.py
            pass
        else:
            logger.warning("Service d'indexation non initialisé")
    
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
            
            # Ensure file_path is not empty and has a valid directory
            if not file_path:
                logger.warning(f"Empty file path detected, using default path")
                file_path = os.path.join(self.output_dir, f"generated_{int(time.time())}.{file_info.get('type', 'txt')}")
            
            # Create directory if it doesn't exist
            dir_name = os.path.dirname(file_path)
            if not dir_name:
                # If there's no directory part, use the output directory
                file_path = os.path.join(self.output_dir, file_path)
                dir_name = self.output_dir
            
            os.makedirs(dir_name, exist_ok=True)
            
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
        
        # Log pour debug
        logger.debug(f"Analysing contribution of length: {len(contribution)}")
        logger.debug(f"First 200 chars: {contribution[:200]}")
        
        # Patterns pour détecter divers formats de déclaration de fichiers
        file_patterns = [
            # Format standard: ```language\n content \n```
            r"```(\w+)\s*\n(.*?)\n```",
            
            # Format avec nom de fichier spécifié dans un commentaire ou un header
            r"# File: ([^\n]+)\n```(\w+)\s*\n(.*?)\n```",
            r"# FICHIER À CRÉER/MODIFIER: ([^\n]+)\n```(\w+)?\s*\n(.*?)\n```",
            r"Fichier: ([^\n]+)\n```(\w+)?\s*\n(.*?)\n```",
            
            # Formats spécifiques pour différents types de fichiers
            r"# ([a-zA-Z0-9_\-\.\/]+\.py)\n```python\s*\n(.*?)\n```",
            r"# ([a-zA-Z0-9_\-\.\/]+\.js)\n```javascript\s*\n(.*?)\n```",
            r"# ([a-zA-Z0-9_\-\.\/]+\.css)\n```css\s*\n(.*?)\n```",
            r"# ([a-zA-Z0-9_\-\.\/]+\.html)\n```html\s*\n(.*?)\n```",
            r"# ([a-zA-Z0-9_\-\.\/]+\.md)\n```markdown\s*\n(.*?)\n```"
        ]
        
        # Analyser la contribution avec les différents patterns
        import re
        
        # Chercher d'abord les formats avec nom de fichier explicite
        for pattern in file_patterns[1:]:  # Tous sauf le premier pattern
            matches = re.findall(pattern, contribution, re.DOTALL)
            if matches:
                for match in matches:
                    if len(match) == 3:  # Formats avec nom de fichier, langage et contenu
                        file_path, language, content = match
                        ext = os.path.splitext(file_path)[1][1:] if '.' in file_path else language
                    elif len(match) == 2:  # Formats avec nom de fichier et contenu
                        file_path, content = match
                        ext = os.path.splitext(file_path)[1][1:] if '.' in file_path else "txt"
                    else:
                        continue
                    
                    # Nettoyer le chemin du fichier
                    file_path = file_path.strip()
                    
                    # Ajouter l'information du fichier
                    files_info.append({
                        "relative_path": file_path,
                        "content": content,
                        "type": ext
                    })
                    logger.debug(f"Extracted file: {file_path} ({len(content)} chars)")
        
        # Si aucun fichier n'a été trouvé avec les patterns explicites, utiliser le pattern simple
        if not files_info:
            code_blocks = re.findall(file_patterns[0], contribution, re.DOTALL)
            for i, (language, content) in enumerate(code_blocks):
                # Générer un nom de fichier basé sur le langage
                ext = language.lower() if language else "txt"
                file_name = f"generated_file_{i+1}.{ext}"
                
                files_info.append({
                    "relative_path": file_name,
                    "content": content,
                    "type": ext
                })
                logger.debug(f"Extracted unnamed file: {file_name} ({len(content)} chars)")
        
        # Si toujours pas de fichiers, chercher de manière plus agressive
        if not files_info:
            # Chercher des blocs de texte qui ressemblent à du code
            sections = contribution.split("\n\n")
            for section in sections:
                if len(section) > 100 and (
                    "def " in section or 
                    "class " in section or 
                    "function " in section or 
                    "import " in section or
                    "<html>" in section or
                    "const " in section or
                    "@media" in section
                ):
                    # Deviner le type de fichier
                    file_type = "py" if "def " in section or "class " in section else "js"
                    file_type = "html" if "<html>" in section else file_type
                    file_type = "css" if "@media" in section else file_type
                    
                    file_name = f"extracted_code_{len(files_info)+1}.{file_type}"
                    files_info.append({
                        "relative_path": file_name,
                        "content": section,
                        "type": file_type
                    })
                    logger.debug(f"Aggressively extracted file: {file_name} ({len(section)} chars)")
        
        logger.info(f"Total files extracted from contribution: {len(files_info)}")
        return files_info