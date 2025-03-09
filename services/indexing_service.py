"""
Service d'indexation pour gérer l'indexation du code une seule fois par session.
"""
import logging
from typing import Dict, Any, Optional
from core.knowledge_repository import KnowledgeRepository
from tools.code_indexer_tool import CodeIndexerTool
import time
import datetime

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
        # Vérifier si déjà indexé en mémoire lors de cette session
        if IndexingService._indexed and not force:
            logger.info("Code déjà indexé dans cette session, utilisation de l'index existant")
            return
        
        # Vérifier si le dépôt de connaissances contient des infos d'indexation
        try:
            indexing_info = self.knowledge_repo.get_external_knowledge("code_indexing_info")
            
            # Vérifier si l'info d'indexation existe et contient un timestamp valide
            indexed = False
            last_indexing_time = 0
            
            if indexing_info and isinstance(indexing_info, dict) and "content" in indexing_info:
                content = indexing_info.get("content", {})
                if isinstance(content, dict) and "last_indexing_time" in content:
                    last_indexing_time = content.get("last_indexing_time", 0)
                    
                    # Si indexé récemment (moins de 5 minutes), considérer comme déjà indexé
                    current_time = time.time()
                    if current_time - last_indexing_time < 300:  # 5 minutes
                        logger.info(f"Indexation récente effectuée il y a {current_time - last_indexing_time:.1f} secondes")
                        IndexingService._indexed = True
                        return
                    
                    indexed = True
                    logger.info(f"Dernière indexation: {datetime.datetime.fromtimestamp(last_indexing_time).isoformat()}")
            
            # Vérifier aussi si des fichiers sont déjà indexés dans le dépôt
            if not indexed:
                results = self.knowledge_repo.search_knowledge("code_file", k=1, filter_metadata={"type": "code_file"})
                indexed = len(results) > 0
                if indexed:
                    logger.info("Des fichiers de code sont déjà présents dans le dépôt")
            
            # Si indexé, faire une mise à jour incrémentielle
            if indexed:
                logger.info("Base de code déjà indexée, vérification des modifications...")
                try:
                    # Utiliser l'horodatage pour l'indexation incrémentielle
                    result = self.code_indexer.index_codebase_incrementally(
                        ".", 
                        force_update=False, 
                        last_index_time=last_indexing_time
                    )
                    
                    # Enregistrer l'horodatage actuel pour la prochaine indexation
                    self._store_indexing_timestamp()
                    
                    # Marquer comme indexé cette session
                    IndexingService._indexed = True
                    
                    logger.info(f"Mise à jour incrémentielle terminée: {result.get('stats', {}).get('files_indexed', 0)} fichiers indexés")
                except Exception as e:
                    logger.error(f"Erreur lors de la mise à jour de l'index: {str(e)}")
            else:
                # Indexation complète requise
                logger.info("Indexation complète de la base de code...")
                try:
                    self.code_indexer.index_codebase(".")
                    
                    # Enregistrer l'horodatage après indexation
                    self._store_indexing_timestamp()
                    
                    # Marquer comme indexé cette session
                    IndexingService._indexed = True
                    
                    logger.info("Indexation terminée")
                except Exception as e:
                    logger.error(f"Erreur lors de l'indexation: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Erreur générale lors de l'indexation: {str(e)}")
            # Tentons l'indexation complète en cas d'erreur dans la logique
            try:
                self.code_indexer.index_codebase(".")
                self._store_indexing_timestamp()
                IndexingService._indexed = True
                logger.info("Indexation complète de secours terminée")
            except Exception as e2:
                logger.error(f"Échec de l'indexation de secours: {str(e2)}")
    
    def _store_indexing_timestamp(self):
        """Enregistrer l'horodatage actuel de l'indexation."""
        try:
            current_time = time.time()
            self.knowledge_repo.store_external_knowledge(
                source="code_indexing_info",
                content={"last_indexing_time": current_time},
                metadata={"type": "system_info"}
            )
            logger.info(f"Horodatage d'indexation mis à jour: {datetime.datetime.fromtimestamp(current_time).isoformat()}")
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de l'horodatage d'indexation: {str(e)}")