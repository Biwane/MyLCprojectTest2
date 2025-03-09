"""
File Manager Tool Module

This module provides tools for managing files, including reading, writing, creating,
and organizing files and directories to support the agents' operations.
"""

import os
import logging
import json
import yaml
import csv
import shutil
from typing import Dict, Any, List, Optional, Union, BinaryIO
from pathlib import Path
import datetime

logger = logging.getLogger(__name__)

class FileManagerTool:
    """
    Tool for managing files and directories.
    
    This tool provides methods for reading, writing, creating, and organizing files
    and directories to support the agents' operations and store their outputs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the file manager tool.
        
        Args:
            config: Configuration dictionary with file manager settings
        """
        self.config = config
        self.base_dir = config.get("base_dir", ".")
        self.output_dir = config.get("output_dir", "output")
        self.allowed_extensions = config.get("allowed_extensions", [
            "txt", "json", "yaml", "yml", "csv", "md", "py", "js", "html", "css", 
            "java", "cs", "cls", "xml", "log", "ini", "conf"
        ])
        self.max_file_size = config.get("max_file_size", 10 * 1024 * 1024)  # 10 MB default
        
        # Create output directory if it doesn't exist
        output_path = Path(self.base_dir) / self.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Initialized FileManagerTool with base_dir: {self.base_dir}, output_dir: {self.output_dir}")
    
    def read_file(self, file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """
        Read a file and return its contents.
        
        Args:
            file_path: Path to the file to read
            encoding: Encoding to use when reading the file
            
        Returns:
            Dictionary with file contents and metadata
        """
        # Normalize path and check if it exists
        full_path = self._get_full_path(file_path)
        
        try:
            # Check if file exists
            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "content": "",
                    "error": f"File not found: {file_path}",
                    "metadata": {}
                }
            
            # Check if path is a directory
            if os.path.isdir(full_path):
                return {
                    "success": False,
                    "content": "",
                    "error": f"Path is a directory, not a file: {file_path}",
                    "metadata": {}
                }
            
            # Check file size
            file_size = os.path.getsize(full_path)
            if file_size > self.max_file_size:
                return {
                    "success": False,
                    "content": "",
                    "error": f"File size ({file_size} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)",
                    "metadata": {"size": file_size}
                }
            
            # Determine file type based on extension
            file_extension = self._get_file_extension(full_path)
            
            # Read file based on its type
            content = ""
            metadata = {
                "path": file_path,
                "size": file_size,
                "extension": file_extension,
                "last_modified": datetime.datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat()
            }
            
            if file_extension == "json":
                with open(full_path, "r", encoding=encoding) as f:
                    content = json.load(f)
                    metadata["content_type"] = "json"
            elif file_extension in ["yaml", "yml"]:
                with open(full_path, "r", encoding=encoding) as f:
                    content = yaml.safe_load(f)
                    metadata["content_type"] = "yaml"
            elif file_extension == "csv":
                with open(full_path, "r", encoding=encoding, newline="") as f:
                    reader = csv.reader(f)
                    content = list(reader)
                    metadata["content_type"] = "csv"
                    metadata["rows"] = len(content)
                    metadata["columns"] = len(content[0]) if content else 0
            else:
                # Default to text
                with open(full_path, "r", encoding=encoding) as f:
                    content = f.read()
                    metadata["content_type"] = "text"
            
            return {
                "success": True,
                "content": content,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return {
                "success": False,
                "content": "",
                "error": f"Error reading file: {str(e)}",
                "metadata": {}
            }
    
    def write_file(
        self, 
        file_path: str, 
        content: Union[str, Dict, List], 
        mode: str = "w", 
        encoding: str = "utf-8",
        create_dirs: bool = True
    ) -> Dict[str, Any]:
        """
        Write content to a file.
        
        Args:
            file_path: Path to write the file to
            content: Content to write to the file
            mode: File mode ('w' for write, 'a' for append)
            encoding: Encoding to use when writing the file
            create_dirs: Whether to create parent directories if they don't exist
            
        Returns:
            Dictionary with status and metadata
        """
        # DEBUG - Chemin demandé
        print(f"DEBUG - Chemin demandé: {file_path}")
        
        # Normalize path
        full_path = self._get_full_path(file_path)
        
        try:
            # Ensure the file extension is allowed
            file_extension = self._get_file_extension(full_path)
            if file_extension not in self.allowed_extensions:
                return {
                    "success": False,
                    "error": f"File extension '{file_extension}' not allowed",
                    "metadata": {}
                }
            
            # Create parent directories if needed
            if create_dirs:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Write content based on its type and file extension
            if isinstance(content, (dict, list)) and file_extension == "json":
                with open(full_path, mode, encoding=encoding) as f:
                    json.dump(content, f, indent=2)
            elif isinstance(content, (dict, list)) and file_extension in ["yaml", "yml"]:
                with open(full_path, mode, encoding=encoding) as f:
                    yaml.dump(content, f)
            elif isinstance(content, list) and file_extension == "csv":
                with open(full_path, mode, encoding=encoding, newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(content)
            else:
                # Default to text
                with open(full_path, mode, encoding=encoding) as f:
                    f.write(str(content))
            
            # Get file metadata
            file_size = os.path.getsize(full_path)
            metadata = {
                "path": file_path,
                "size": file_size,
                "extension": file_extension,
                "last_modified": datetime.datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat()
            }
            
            logger.debug(f"Successfully wrote to file: {file_path}")
            return {
                "success": True,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error writing to file {file_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Error writing to file: {str(e)}",
                "metadata": {}
            }
    
    def write_app_file(
        self, 
        file_path: str, 
        content: Union[str, Dict, List], 
        is_application_file: bool = False,
        create_dirs: bool = True
    ) -> Dict[str, Any]:
        """
        Write content to a file with special handling for application files vs output files.
        
        Args:
            file_path: Path to write the file to
            content: Content to write to the file
            is_application_file: Whether this is a core application file (not in output directory)
            create_dirs: Whether to create parent directories if they don't exist
            
        Returns:
            Dictionary with status and metadata
        """
        # DEBUG - Chemin demandé
        print(f"DEBUG - Chemin demandé: {file_path}")
        
        # Déterminer le chemin approprié
        if is_application_file:
            # Si c'est un fichier d'application, utiliser le chemin directement (relatif à la racine)
            full_path = os.path.abspath(file_path)
        else:
            # Sinon, utiliser le chemin relatif au dossier output
            full_path = self._get_full_path(file_path)
        
        # DEBUG - Chemin avant transformation
        print(f"DEBUG - Chemin avant transformation: {file_path}")
        
        transformed_path = self._transform_destination_path(file_path)
        
        # DEBUG - Chemin après transformation
        print(f"DEBUG - Chemin après transformation: {transformed_path}")
        
        try:
            # Ensure the file extension is allowed
            file_extension = self._get_file_extension(full_path)
            if file_extension not in self.allowed_extensions:
                return {
                    "success": False,
                    "error": f"File extension '{file_extension}' not allowed",
                    "metadata": {}
                }
            
            # Create parent directories if needed
            if create_dirs:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Détermine le mode d'écriture selon le type
            if isinstance(content, (dict, list)) and file_extension == "json":
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, indent=2)
            elif isinstance(content, (dict, list)) and file_extension in ["yaml", "yml"]:
                with open(full_path, 'w', encoding='utf-8') as f:
                    yaml.dump(content, f)
            elif isinstance(content, list) and file_extension == "csv":
                with open(full_path, 'w', encoding='utf-8', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(content)
            else:
                # Default to text
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(str(content))
            
            # Get file metadata
            file_size = os.path.getsize(full_path)
            metadata = {
                "path": file_path,
                "size": file_size,
                "extension": file_extension,
                "is_application_file": is_application_file,
                "last_modified": datetime.datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat()
            }
            
            logger.debug(f"Successfully wrote to file: {file_path}")
            return {
                "success": True,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error writing to file {file_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Error writing to file: {str(e)}",
                "metadata": {"is_application_file": is_application_file}
            }

    def create_directory(self, dir_path: str) -> Dict[str, Any]:
        """
        Create a directory.
        
        Args:
            dir_path: Path to the directory to create
            
        Returns:
            Dictionary with status and metadata
        """
        # Normalize path
        full_path = self._get_full_path(dir_path)
        
        try:
            # Create directory and parent directories
            os.makedirs(full_path, exist_ok=True)
            
            metadata = {
                "path": dir_path,
                "created": datetime.datetime.now().isoformat()
            }
            
            logger.debug(f"Successfully created directory: {dir_path}")
            return {
                "success": True,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error creating directory {dir_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Error creating directory: {str(e)}",
                "metadata": {}
            }
    
    def list_directory(
        self, 
        dir_path: str, 
        include_metadata: bool = False,
        recursive: bool = False
    ) -> Dict[str, Any]:
        """
        List contents of a directory.
        
        Args:
            dir_path: Path to the directory to list
            include_metadata: Whether to include metadata for each file
            recursive: Whether to list subdirectories recursively
            
        Returns:
            Dictionary with directory contents and metadata
        """
        # Normalize path
        full_path = self._get_full_path(dir_path)
        
        try:
            # Check if directory exists
            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "contents": [],
                    "error": f"Directory not found: {dir_path}",
                    "metadata": {}
                }
            
            # Check if path is a directory
            if not os.path.isdir(full_path):
                return {
                    "success": False,
                    "contents": [],
                    "error": f"Path is a file, not a directory: {dir_path}",
                    "metadata": {}
                }
            
            # List contents
            contents = []
            
            if recursive:
                # Recursive listing
                for root, dirs, files in os.walk(full_path):
                    rel_path = os.path.relpath(root, full_path)
                    if rel_path == ".":
                        rel_path = ""
                    
                    # Add directories
                    for dir_name in dirs:
                        dir_item = {
                            "name": dir_name,
                            "path": os.path.join(rel_path, dir_name) if rel_path else dir_name,
                            "type": "directory"
                        }
                        
                        if include_metadata:
                            dir_full_path = os.path.join(root, dir_name)
                            dir_item["metadata"] = {
                                "last_modified": datetime.datetime.fromtimestamp(os.path.getmtime(dir_full_path)).isoformat()
                            }
                        
                        contents.append(dir_item)
                    
                    # Add files
                    for file_name in files:
                        file_item = {
                            "name": file_name,
                            "path": os.path.join(rel_path, file_name) if rel_path else file_name,
                            "type": "file",
                            "extension": self._get_file_extension(file_name)
                        }
                        
                        if include_metadata:
                            file_full_path = os.path.join(root, file_name)
                            file_item["metadata"] = {
                                "size": os.path.getsize(file_full_path),
                                "last_modified": datetime.datetime.fromtimestamp(os.path.getmtime(file_full_path)).isoformat()
                            }
                        
                        contents.append(file_item)
            else:
                # Non-recursive listing
                for item_name in os.listdir(full_path):
                    item_path = os.path.join(full_path, item_name)
                    is_dir = os.path.isdir(item_path)
                    
                    item = {
                        "name": item_name,
                        "type": "directory" if is_dir else "file"
                    }
                    
                    if not is_dir:
                        item["extension"] = self._get_file_extension(item_name)
                    
                    if include_metadata:
                        item["metadata"] = {
                            "last_modified": datetime.datetime.fromtimestamp(os.path.getmtime(item_path)).isoformat()
                        }
                        
                        if not is_dir:
                            item["metadata"]["size"] = os.path.getsize(item_path)
                    
                    contents.append(item)
            
            # Sort contents: directories first, then files
            contents.sort(key=lambda x: (0 if x["type"] == "directory" else 1, x["name"]))
            
            dir_metadata = {
                "path": dir_path,
                "item_count": len(contents),
                "directories": sum(1 for item in contents if item["type"] == "directory"),
                "files": sum(1 for item in contents if item["type"] == "file")
            }
            
            return {
                "success": True,
                "contents": contents,
                "metadata": dir_metadata
            }
            
        except Exception as e:
            logger.error(f"Error listing directory {dir_path}: {str(e)}")
            return {
                "success": False,
                "contents": [],
                "error": f"Error listing directory: {str(e)}",
                "metadata": {}
            }
    
    def delete_file(self, file_path: str) -> Dict[str, Any]:
        """
        Delete a file.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            Dictionary with status and metadata
        """
        # Normalize path
        full_path = self._get_full_path(file_path)
        
        try:
            # Check if file exists
            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": f"File not found: {file_path}",
                    "metadata": {}
                }
            
            # Check if path is a directory
            if os.path.isdir(full_path):
                return {
                    "success": False,
                    "error": f"Path is a directory, not a file: {file_path}",
                    "metadata": {}
                }
            
            # Get file metadata before deletion
            metadata = {
                "path": file_path,
                "size": os.path.getsize(full_path),
                "last_modified": datetime.datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat(),
                "deleted_at": datetime.datetime.now().isoformat()
            }
            
            # Delete the file
            os.remove(full_path)
            
            logger.debug(f"Successfully deleted file: {file_path}")
            return {
                "success": True,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Error deleting file: {str(e)}",
                "metadata": {}
            }
    
    def delete_directory(self, dir_path: str, recursive: bool = False) -> Dict[str, Any]:
        """
        Delete a directory.
        
        Args:
            dir_path: Path to the directory to delete
            recursive: Whether to delete subdirectories and files
            
        Returns:
            Dictionary with status and metadata
        """
        # Normalize path
        full_path = self._get_full_path(dir_path)
        
        try:
            # Check if directory exists
            if not os.path.exists(full_path):
                return {
                    "success": False,
                    "error": f"Directory not found: {dir_path}",
                    "metadata": {}
                }
            
            # Check if path is a directory
            if not os.path.isdir(full_path):
                return {
                    "success": False,
                    "error": f"Path is a file, not a directory: {dir_path}",
                    "metadata": {}
                }
            
            # Get directory metadata before deletion
            metadata = {
                "path": dir_path,
                "deleted_at": datetime.datetime.now().isoformat()
            }
            
            # Delete the directory
            if recursive:
                shutil.rmtree(full_path)
            else:
                os.rmdir(full_path)
            
            logger.debug(f"Successfully deleted directory: {dir_path}")
            return {
                "success": True,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error deleting directory {dir_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Error deleting directory: {str(e)}",
                "metadata": {}
            }
    
    def copy_file(self, source_path: str, dest_path: str, overwrite: bool = False) -> Dict[str, Any]:
        """
        Copy a file from source to destination.
        
        Args:
            source_path: Path to the source file
            dest_path: Path to the destination file
            overwrite: Whether to overwrite the destination if it exists
            
        Returns:
            Dictionary with status and metadata
        """
        # Normalize paths
        full_source_path = self._get_full_path(source_path)
        full_dest_path = self._get_full_path(dest_path)
        
        try:
            # Check if source file exists
            if not os.path.exists(full_source_path):
                return {
                    "success": False,
                    "error": f"Source file not found: {source_path}",
                    "metadata": {}
                }
            
            # Check if source is a directory
            if os.path.isdir(full_source_path):
                return {
                    "success": False,
                    "error": f"Source is a directory, not a file: {source_path}",
                    "metadata": {}
                }
            
            # Check if destination exists and whether to overwrite
            if os.path.exists(full_dest_path) and not overwrite:
                return {
                    "success": False,
                    "error": f"Destination file already exists: {dest_path} (set overwrite=True to overwrite)",
                    "metadata": {}
                }
            
            # Create parent directories if needed
            os.makedirs(os.path.dirname(full_dest_path), exist_ok=True)
            
            # Copy the file
            shutil.copy2(full_source_path, full_dest_path)
            
            metadata = {
                "source_path": source_path,
                "dest_path": dest_path,
                "size": os.path.getsize(full_dest_path),
                "last_modified": datetime.datetime.fromtimestamp(os.path.getmtime(full_dest_path)).isoformat()
            }
            
            logger.debug(f"Successfully copied file from {source_path} to {dest_path}")
            return {
                "success": True,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error copying file from {source_path} to {dest_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Error copying file: {str(e)}",
                "metadata": {}
            }
    
    def move_file(self, source_path: str, dest_path: str, overwrite: bool = False) -> Dict[str, Any]:
        """
        Move a file from source to destination.
        
        Args:
            source_path: Path to the source file
            dest_path: Path to the destination file
            overwrite: Whether to overwrite the destination if it exists
            
        Returns:
            Dictionary with status and metadata
        """
        # Normalize paths
        full_source_path = self._get_full_path(source_path)
        full_dest_path = self._get_full_path(dest_path)
        
        try:
            # Check if source file exists
            if not os.path.exists(full_source_path):
                return {
                    "success": False,
                    "error": f"Source file not found: {source_path}",
                    "metadata": {}
                }
            
            # Check if source is a directory
            if os.path.isdir(full_source_path):
                return {
                    "success": False,
                    "error": f"Source is a directory, not a file: {source_path}",
                    "metadata": {}
                }
            
            # Check if destination exists and whether to overwrite
            if os.path.exists(full_dest_path) and not overwrite:
                return {
                    "success": False,
                    "error": f"Destination file already exists: {dest_path} (set overwrite=True to overwrite)",
                    "metadata": {}
                }
            
            # Create parent directories if needed
            os.makedirs(os.path.dirname(full_dest_path), exist_ok=True)
            
            # Move the file
            shutil.move(full_source_path, full_dest_path)
            
            metadata = {
                "source_path": source_path,
                "dest_path": dest_path,
                "size": os.path.getsize(full_dest_path),
                "last_modified": datetime.datetime.fromtimestamp(os.path.getmtime(full_dest_path)).isoformat()
            }
            
            logger.debug(f"Successfully moved file from {source_path} to {dest_path}")
            return {
                "success": True,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error moving file from {source_path} to {dest_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Error moving file: {str(e)}",
                "metadata": {}
            }
    
    def create_temp_file(
        self, 
        content: Union[str, Dict, List], 
        prefix: str = "temp_", 
        suffix: str = ".txt",
        encoding: str = "utf-8"
    ) -> Dict[str, Any]:
        """
        Create a temporary file with the given content.
        
        Args:
            content: Content to write to the file
            prefix: Prefix for the temporary file name
            suffix: Suffix for the temporary file name
            encoding: Encoding to use when writing the file
            
        Returns:
            Dictionary with file path and metadata
        """
        try:
            # Create temp directory if it doesn't exist
            temp_dir = os.path.join(self.base_dir, self.output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Generate a unique file name
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            file_name = f"{prefix}{timestamp}{suffix}"
            file_path = os.path.join("temp", file_name)
            
            # Write the content to the file
            result = self.write_file(file_path, content, encoding=encoding)
            
            if result["success"]:
                logger.debug(f"Successfully created temporary file: {file_path}")
                result["path"] = file_path
                return result
            else:
                return result
            
        except Exception as e:
            logger.error(f"Error creating temporary file: {str(e)}")
            return {
                "success": False,
                "error": f"Error creating temporary file: {str(e)}",
                "metadata": {}
            }
    
    def _get_full_path(self, path: str) -> str:
        """
        Get the full absolute path from a relative path.
        
        Args:
            path: Relative path
            
        Returns:
            Absolute path
        """
        # Check if path starts with output directory
        if path.startswith("output/") or path.startswith("output\\"):
            # Path is relative to base directory
            return os.path.abspath(os.path.join(self.base_dir, path))
        
        # Check if path already starts with the base directory
        base_dir_abs = os.path.abspath(self.base_dir)
        if os.path.abspath(path).startswith(base_dir_abs):
            # Path is already relative to base directory
            return os.path.abspath(path)
        
        # Path is assumed to be relative to output directory
        return os.path.abspath(os.path.join(self.base_dir, self.output_dir, path))
    
    def _get_file_extension(self, file_path: str) -> str:
        """
        Get the file extension from a file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File extension without the dot
        """
        return os.path.splitext(file_path)[1].lstrip(".").lower()

    def _transform_destination_path(self, relative_path: str) -> str:
        """
        Transform the destination path based on file type and purpose.
        
        Args:
            relative_path: Original relative path
            
        Returns:
            Transformed path appropriate for the application
        """
        # Respecter simplement le chemin relatif spécifié par l'agent
        # Sauf s'il commence explicitement par output/
        if relative_path.startswith("output/"):
            return os.path.join(self.output_dir, relative_path[7:])
        
        # Désactiver toute redirection automatique vers des dossiers spécifiques
        return relative_path
