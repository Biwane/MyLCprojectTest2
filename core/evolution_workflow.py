"""
Evolution Workflow Module

Ce module implémente le workflow d'évolution du code, permettant aux agents
d'analyser et d'améliorer leur propre codebase en réponse à des demandes d'évolution.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Union

from core.agent_coordinator import AgentCoordinator
from core.team_manager import TeamManager
from core.knowledge_repository import KnowledgeRepository
from tools.code_indexer_tool import CodeIndexerTool
from tools.code_diff_tool import CodeDiffTool

logger = logging.getLogger(__name__)

class EvolutionWorkflow:
    """
    Gère le workflow d'évolution du code, coordonnant les différentes étapes
    nécessaires pour analyser, planifier et implémenter des modifications de code.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        team_manager: TeamManager,
        agent_coordinator: AgentCoordinator,
        knowledge_repository: KnowledgeRepository
    ):
        """
        Initialise le workflow d'évolution.
        
        Args:
            config: Configuration du workflow
            team_manager: Gestionnaire d'équipe pour créer ou récupérer des équipes
            agent_coordinator: Coordinateur d'agents pour exécuter des tâches
            knowledge_repository: Dépôt de connaissances partagées
        """
        self.config = config
        self.team_manager = team_manager
        self.agent_coordinator = agent_coordinator
        self.knowledge_repository = knowledge_repository
        
        # Initialiser les outils spécifiques à l'évolution
        self.code_indexer = CodeIndexerTool(config.get("tools", {}).get("code_indexer", {}), knowledge_repository)
        self.code_diff_tool = CodeDiffTool(config.get("tools", {}).get("code_diff", {}))
        
        # Répertoire racine du code à analyser
        self.code_root_dir = config.get("code_root_dir", ".")
        
        logger.debug("Workflow d'évolution initialisé")
    
    def execute_evolution(
        self, 
        evolution_request: str, 
        team_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Exécute le workflow d'évolution complet pour une demande d'évolution.
        
        Args:
            evolution_request: Description de l'évolution demandée
            team_id: Identifiant d'une équipe existante (en crée une nouvelle si None)
            
        Returns:
            Résultats de l'évolution
        """
        logger.info(f"Démarrage du workflow d'évolution pour: {evolution_request}")
        
        # Étape 1: Obtenir ou créer l'équipe d'agents
        agent_team = self._get_or_create_team(evolution_request, team_id)
        
        # Étape 2: Indexer la base de code si ce n'est pas déjà fait
        self._ensure_code_indexed()
        
        # Étape 3: Créer les tâches pour le workflow d'évolution
        evolution_task = f"""
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
        
        # Étape 4: Exécuter les tâches avec l'équipe d'agents
        results = self.agent_coordinator.execute_task(evolution_task, agent_team)
        
        # Étape 5: Récupérer et traiter les résultats
        self._process_evolution_results(results, evolution_request)
        
        return results
    
    def _get_or_create_team(self, evolution_request: str, team_id: Optional[str]) -> Dict[str, Any]:
        """
        Obtient une équipe existante ou en crée une nouvelle pour l'évolution.
        
        Args:
            evolution_request: Description de l'évolution
            team_id: ID d'une équipe existante (optionnel)
            
        Returns:
            L'équipe d'agents
        """
        if team_id:
            # Utiliser une équipe existante
            agent_team = self.team_manager.get_team(team_id)
            if not agent_team:
                logger.warning(f"Équipe {team_id} non trouvée, création d'une nouvelle équipe")
                return self._create_evolution_team(evolution_request)
            return agent_team
        else:
            # Créer une nouvelle équipe spécialisée
            return self._create_evolution_team(evolution_request)
    
    def _create_evolution_team(self, evolution_request: str) -> Dict[str, Any]:
        """
        Crée une équipe spécialisée pour l'évolution du code.
        
        Args:
            evolution_request: Description de l'évolution
            
        Returns:
            L'équipe d'agents créée
        """
        # Analyser la tâche pour déterminer la composition de l'équipe
        agent_specs = self.team_manager.analyze_task(
            f"Améliorer cette application en implémentant: {evolution_request}"
        )
        
        # Assurer qu'un agent d'analyse de code est présent
        has_code_analyst = any(spec.get("role") == "code_analyst" for spec in agent_specs)
        if not has_code_analyst:
            agent_specs.append({
                "role": "code_analyst",
                "specialization": "code_analysis",
                "importance": 9,
                "description": "Analyse le code pour comprendre sa structure et proposer des modifications",
                "required_skills": ["code analysis", "refactoring", "software architecture"]
            })
        
        # Créer l'équipe avec les spécifications
        return self.team_manager.create_team(agent_specs)
    
    def _ensure_code_indexed(self) -> None:
        """
        S'assure que la base de code est indexée dans le référentiel de connaissances.
        """
        # Vérifier si la base de code a déjà été indexée
        code_indexed = False
        
        # Chercher dans le référentiel de connaissances
        search_results = self.knowledge_repository.search_knowledge(
            query="code_file",
            k=1,
            filter_metadata={"type": "code_file"}
        )
        
        code_indexed = len(search_results) > 0
        
        if not code_indexed:
            logger.info("Indexation de la base de code...")
            # Exclure certains répertoires de l'indexation
            exclude_dirs = ["__pycache__", "venv", ".git", "node_modules", "data/vector_store"]
            
            # Indexer la base de code
            result = self.code_indexer.index_codebase(
                self.code_root_dir, 
                exclude_patterns=[f".*{d}.*" for d in exclude_dirs]
            )
            
            if result["success"]:
                logger.info(f"Base de code indexée: {result['stats']['files_indexed']} fichiers")
            else:
                logger.error(f"Échec de l'indexation: {result.get('error', 'Unknown error')}")
        else:
            logger.info("Base de code déjà indexée, utilisation de l'index existant")
    
    def _process_evolution_results(self, results: Dict[str, Any], evolution_request: str) -> None:
        """
        Traite les résultats de l'évolution, applique les modifications si nécessaire.
        
        Args:
            results: Résultats de l'exécution des tâches
            evolution_request: Description de l'évolution demandée
        """
        # Récupérer les fichiers de sortie générés pendant l'évolution
        output_files = results.get("output_files", [])
        
        if output_files:
            logger.info(f"Fichiers générés pendant l'évolution: {len(output_files)}")
            for file_path in output_files:
                logger.info(f"- {file_path}")
            
            # Ici, on pourrait ajouter une logique pour appliquer automatiquement 
            # les modifications (par exemple en utilisant les fichiers diff générés)
        else:
            logger.info("Aucun fichier n'a été généré pendant l'évolution")