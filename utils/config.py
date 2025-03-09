"""
Configuration Module

This module handles configuration loading, validation, and access throughout the
system. It supports loading from YAML files, environment variables, and provides
default values for missing configurations.
"""

import os
import logging
import yaml
from typing import Dict, Any, Optional, List, Union
import json

logger = logging.getLogger(__name__)

class Config:
    """
    Configuration manager for the system.
    
    Handles loading configuration from various sources, provides access
    to configuration values, and validates configuration settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to a YAML configuration file
        """
        self.config_data = {}
        self.config_path = config_path
        
        # Load default configuration
        self._load_defaults()
        
        # Load from config file if specified
        if config_path:
            self._load_from_file(config_path)
        
        # Override with environment variables
        self._load_from_env()
        
        # Validate the configuration
        self._validate_config()
        
        logger.debug(f"Configuration initialized. Path: {config_path if config_path else 'default'}")
    
    def _load_defaults(self):
        """Load default configuration values."""
        self.config_data = {
            # General settings
            "general": {
                "data_dir": "data",
                "output_dir": "output",
                "log_level": "INFO",
                "verbose": False
            },
            
            # LLM settings
            "llm": {
                "default_model": "gpt-4o",
                "research_model": "gpt-4o",
                "planning_model": "gpt-4o",
                "coordination_model": "gpt-4o",
                "specialist_model": "gpt-4o",
                "execution_model": "gpt-3.5-turbo",
                "review_model": "gpt-4o",
                "default_temperature": 0.2,
                "api_request_timeout": 30
            },
            
            # Agent settings
            "agent_factory": {
                "default_model": "gpt-4o",
                "max_iterations": 10,
                "verbose": True
            },
            
            # Team manager settings
            "team_manager": {
                "analysis_model": "gpt-4o",
                "agent_factory": {
                    "default_model": "gpt-4o"
                }
            },
            
            # Agent coordinator settings
            "agent_coordinator": {
                "coordination_model": "gpt-4o",
                "max_coordination_retries": 3,
                "task_scheduler": {
                    "max_parallel_tasks": 3,
                    "prioritize_by_complexity": True
                }
            },
            
            # Knowledge repository settings
            "knowledge_repository": {
                "data_dir": "data",
                "embedding_model": "text-embedding-3-small",
                "chunk_size": 1000,
                "chunk_overlap": 100
            },
            
            # Task scheduler settings
            "task_scheduler": {
                "max_parallel_tasks": 3,
                "prioritize_by_complexity": True
            },
            
            # Agents settings
            "agents": {
                "base_agent": {
                    "max_retries": 3,
                    "retry_delay": 2,
                    "max_history": 10
                },
                "research_agent": {
                    "auto_save_results": True,
                    "max_search_results": 5,
                    "include_sources": True
                },
                "specialist_agent": {
                    "domain_knowledge": {},
                    "best_practices": []
                },
                "planner_agent": {
                    "planning_depth": "medium",
                    "include_contingencies": True
                },
                "executor_agent": {
                    "execution_timeout": 120,
                    "validate_results": True
                },
                "reviewer_agent": {
                    "review_criteria": [
                        "correctness",
                        "completeness",
                        "efficiency",
                        "maintainability"
                    ]
                }
            },
            
            # Tools settings
            "tools": {
                "web_search": {
                    "search_provider": "tavily",
                    "max_results": 5
                },
                "code_generation": {
                    "language_support": [
                        "python",
                        "javascript",
                        "java",
                        "csharp",
                        "apex"
                    ]
                },
                "knowledge_retrieval": {
                    "max_results": 5,
                    "similarity_threshold": 0.7
                }
            }
        }
    
    def _load_from_file(self, config_path: str):
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    
                    if file_config:
                        # Recursively update the config with values from the file
                        self._update_nested_dict(self.config_data, file_config)
                        logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.warning(f"Configuration file not found: {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from file: {str(e)}")
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        try:
            # Look for environment variables with the prefix TEAM_AGENTS_
            prefix = "TEAM_AGENTS_"
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    # Remove the prefix and convert to lowercase
                    config_key = key[len(prefix):].lower()
                    
                    # Split by double underscore to represent nested keys
                    path = config_key.split("__")
                    
                    # Try to parse as JSON for complex values
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        parsed_value = value
                    
                    # Update the config at the specified path
                    self._set_nested_value(self.config_data, path, parsed_value)
                    
                    logger.debug(f"Loaded configuration from environment: {key}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from environment: {str(e)}")
    
    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]):
        """
        Recursively update a nested dictionary.
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def _set_nested_value(self, d: Dict[str, Any], path: List[str], value: Any):
        """
        Set a value in a nested dictionary given a path.
        
        Args:
            d: Dictionary to update
            path: List of keys forming the path
            value: Value to set
        """
        if len(path) == 1:
            d[path[0]] = value
            return
            
        if path[0] not in d:
            d[path[0]] = {}
        elif not isinstance(d[path[0]], dict):
            d[path[0]] = {}
            
        self._set_nested_value(d[path[0]], path[1:], value)
    
    def _validate_config(self):
        """Validate the configuration and ensure required values are present."""
        # Check for required LLM models
        if not self.get("llm.default_model"):
            logger.warning("No default LLM model specified, using gpt-4o")
            self._set_nested_value(self.config_data, ["llm", "default_model"], "gpt-4o")
        
        # Check for data directory
        data_dir = self.get("general.data_dir")
        if not data_dir:
            data_dir = "data"
            self._set_nested_value(self.config_data, ["general", "data_dir"], data_dir)
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir, exist_ok=True)
                logger.info(f"Created data directory: {data_dir}")
            except Exception as e:
                logger.error(f"Failed to create data directory: {str(e)}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by its key path.
        
        Args:
            key_path: Dot-separated path to the configuration value
            default: Default value to return if key not found
            
        Returns:
            The configuration value or default if not found
        """
        path = key_path.split('.')
        value = self.config_data
        
        try:
            for key in path:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        Set a configuration value by its key path.
        
        Args:
            key_path: Dot-separated path to the configuration value
            value: Value to set
        """
        path = key_path.split('.')
        self._set_nested_value(self.config_data, path, value)
    
    def save(self, file_path: Optional[str] = None):
        """
        Save the configuration to a YAML file.
        
        Args:
            file_path: Path to save the configuration to (defaults to original path)
        """
        save_path = file_path or self.config_path
        
        if not save_path:
            logger.warning("No file path specified for saving configuration")
            return
            
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config_data, f, default_flow_style=False)
                
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to file: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the entire configuration as a dictionary.
        
        Returns:
            The configuration dictionary
        """
        return self.config_data.copy()
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return yaml.dump(self.config_data, default_flow_style=False)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from the specified path or default locations.
    
    Args:
        config_path: Optional explicit path to configuration file
        
    Returns:
        Config instance
    """
    # If no path specified, try standard locations
    if not config_path:
        potential_paths = [
            "config.yaml",
            "config.yml",
            os.path.join("config", "config.yaml"),
            os.path.expanduser("~/.team_agents/config.yaml")
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    # Load the configuration
    return Config(config_path)
