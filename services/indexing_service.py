"""
Service d'indexation pour gérer l'indexation du code une seule fois par session.
"""
import logging
from typing import Dict, Any, Optional
from core.knowledge_repository import KnowledgeRepository
from tools.code_indexer_tool import CodeIndexerTool

logger = logging.getLogger(__name__)

class IndexingService:
    """Service pour gérer l'indexation du code."""
    
    _instance = None  # Instance singleton
    _indexed = False  # Drapeau indiquant si l'indexation a été effectuée
    
    @classmethod
    def get_instance(cls, config: Dict[str, Any] = None, knowledge_repo: Optional[KnowledgeRepository] = None):
        """Obtenir l'instance singleton du service."""
        if cls._instance is None and config and knowledge_repo:
            cls._instance = cls(config, knowledge_repo)
        return cls._instance
    
    def __init__(self, config: Dict[str, Any], knowledge_repo: KnowledgeRepository):
        """Initialiser le service."""
        self.config = config
        self.knowledge_repo = knowledge_repo
        self.code_indexer = CodeIndexerTool(config.get("code_indexer", {}), knowledge_repo)
    
    def ensure_code_indexed(self, force: bool = False):
        """
        S'assurer que le code est indexé.
        Ne fait l'indexation qu'une seule fois par session sauf si force=True.
        """
        if IndexingService._indexed and not force:
            logger.info("Code déjà indexé dans cette session, utilisation de l'index existant")
            return
        
        # Vérifier si le code est déjà indexé
        indexed = False
        try:
            results = self.knowledge_repo.search_knowledge("code_file", k=1, filter_metadata={"type": "code_file"})
            indexed = len(results) > 0
        except Exception:
            indexed = False
        
        if indexed:
            logger.info("Base de code déjà indexée, vérification des modifications...")
            try:
                result = self.code_indexer.index_codebase_incrementally(".", force_update=False)
                logger.info(f"Mise à jour incrémentielle terminée: {result['stats']['files_indexed']} fichiers indexés")
            except Exception as e:
                logger.error(f"Erreur lors de la mise à jour de l'index: {str(e)}")
        else:
            logger.info("Indexation complète de la base de code...")
            try:
                self.code_indexer.index_codebase(".")
                logger.info("Indexation terminée")
            except Exception as e:
                logger.error(f"Erreur lors de l'indexation: {str(e)}")
        
        # Marquer comme indexé
        IndexingService._indexed = True