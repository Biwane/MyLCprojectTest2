"""
Code Indexer Tool Module

This module provides tools for indexing, analyzing, and working with codebases.
It enables agents to read, understand, and propose modifications to code.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
import re
from pathlib import Path

from core.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)

class CodeIndexerTool:
    """
    Tool for indexing and analyzing codebases.
    
    This tool provides functionality to read and index code files,
    understand their relationships, and enable code-aware AI operations.
    """
    
    def __init__(self, config: Dict[str, Any], knowledge_repository: KnowledgeRepository):
        """
        Initialize the code indexer tool.
        
        Args:
            config: Configuration dictionary with tool settings
            knowledge_repository: The knowledge repository to store code information
        """
        self.config = config
        self.knowledge_repository = knowledge_repository
        self.supported_extensions = config.get("supported_extensions", [
            ".py", ".js", ".html", ".css", ".java", ".json", ".yaml", ".yml", 
            ".md", ".txt", ".jsx", ".tsx", ".ts"
        ])
        self.exclude_dirs = config.get("exclude_dirs", [
            "__pycache__", "node_modules", ".git", "venv", "env", ".vscode",
            ".idea", "__MACOSX", "dist", "build", ".pytest_cache"
        ])
        self.max_file_size = config.get("max_file_size", 1 * 1024 * 1024)  # 1MB default
        
        logger.debug(f"Initialized CodeIndexerTool with {len(self.supported_extensions)} supported extensions")
    
    def index_codebase(
        self, 
        root_dir: str, 
        relative_to: Optional[str] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Index a codebase by reading and storing code files in the knowledge repository.
        
        Args:
            root_dir: Root directory of the codebase to index
            relative_to: If provided, file paths will be stored relative to this directory
            include_patterns: Optional list of glob patterns to include
            exclude_patterns: Optional list of glob patterns to exclude in addition to exclude_dirs
            
        Returns:
            Dictionary with indexing results and statistics
        """
        if not os.path.isdir(root_dir):
            return {
                "success": False,
                "error": f"Directory not found: {root_dir}",
                "files_indexed": 0
            }
        
        # Normalize paths
        root_dir = os.path.abspath(root_dir)
        base_dir = os.path.abspath(relative_to) if relative_to else root_dir
        
        # Compile exclude patterns
        exclude_patterns = exclude_patterns or []
        exclude_regexes = [re.compile(pattern) for pattern in exclude_patterns]
        
        # Statistics
        stats = {
            "files_indexed": 0,
            "files_skipped": 0,
            "bytes_indexed": 0,
            "by_extension": {}
        }
        
        indexed_files = []
        
        # Walk the directory tree
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Skip excluded directories
            dirnames[:] = [d for d in dirnames if d not in self.exclude_dirs]
            
            # Process files
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                
                # Get the extension
                _, extension = os.path.splitext(filename)
                
                # Skip if extension not supported
                if extension not in self.supported_extensions:
                    stats["files_skipped"] += 1
                    continue
                
                # Check if file matches exclude patterns
                if any(regex.search(file_path) for regex in exclude_regexes):
                    stats["files_skipped"] += 1
                    continue
                
                # Check file size
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size > self.max_file_size:
                        logger.warning(f"Skipping file due to size limit: {file_path} ({file_size} bytes)")
                        stats["files_skipped"] += 1
                        continue
                        
                    # Get relative path
                    rel_path = os.path.relpath(file_path, base_dir)
                    
                    # Read the file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        # Try with a different encoding or skip binary files
                        logger.warning(f"Skipping file due to encoding issues: {file_path}")
                        stats["files_skipped"] += 1
                        continue
                    
                    # Update statistics
                    stats["files_indexed"] += 1
                    stats["bytes_indexed"] += file_size
                    
                    ext = extension[1:]  # Remove the dot
                    if ext not in stats["by_extension"]:
                        stats["by_extension"][ext] = 0
                    stats["by_extension"][ext] += 1
                    
                    # Store in knowledge repository
                    self.knowledge_repository.store_external_knowledge(
                        source=rel_path,
                        content=content,
                        metadata={
                            "type": "code_file",
                            "file_path": rel_path,
                            "extension": ext,
                            "size": file_size,
                            "language": self._get_language_from_extension(ext)
                        }
                    )
                    
                    indexed_files.append(rel_path)
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    stats["files_skipped"] += 1
        
        return {
            "success": True,
            "indexed_files": indexed_files,
            "stats": stats
        }
    
    def analyze_codebase_structure(self, root_dir: str) -> Dict[str, Any]:
        """
        Analyze the structure of a codebase to understand its organization
        and relationships between files.
        
        Args:
            root_dir: Root directory of the codebase to analyze
            
        Returns:
            Dictionary with codebase structure analysis
        """
        if not os.path.isdir(root_dir):
            return {
                "success": False,
                "error": f"Directory not found: {root_dir}"
            }
        
        # Normalize path
        root_dir = os.path.abspath(root_dir)
        
        # Build directory tree
        tree = self._build_directory_tree(root_dir)
        
        # Find Python modules and packages
        python_modules = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Check if this is a Python package (has __init__.py)
            is_package = "__init__.py" in filenames
            
            # Add Python files as modules
            for filename in filenames:
                if filename.endswith(".py"):
                    rel_path = os.path.relpath(os.path.join(dirpath, filename), root_dir)
                    python_modules.append({
                        "path": rel_path,
                        "in_package": is_package,
                        "name": os.path.splitext(filename)[0]
                    })
        
        # Try to find import relationships
        import_relationships = self._analyze_python_imports(root_dir, python_modules)
        
        # Build the file type breakdown
        file_types = {}
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                _, ext = os.path.splitext(filename)
                if ext:
                    ext = ext[1:]  # Remove the dot
                    if ext not in file_types:
                        file_types[ext] = 0
                    file_types[ext] += 1
        
        return {
            "success": True,
            "directory_tree": tree,
            "python_modules": python_modules,
            "import_relationships": import_relationships,
            "file_types": file_types
        }
    
    def _build_directory_tree(self, root_dir: str) -> Dict[str, Any]:
        """
        Build a nested dictionary representing the directory structure.
        
        Args:
            root_dir: Root directory to start from
            
        Returns:
            Nested dictionary of directories and files
        """
        tree = {"name": os.path.basename(root_dir), "type": "directory", "children": []}
        
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            
            # Skip excluded directories
            if os.path.isdir(item_path) and item in self.exclude_dirs:
                continue
                
            if os.path.isdir(item_path):
                subtree = self._build_directory_tree(item_path)
                tree["children"].append(subtree)
            else:
                _, ext = os.path.splitext(item)
                tree["children"].append({
                    "name": item,
                    "type": "file",
                    "extension": ext[1:] if ext else ""  # Remove the dot
                })
        
        return tree
    
    def _analyze_python_imports(self, root_dir: str, python_modules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze import statements in Python files to find relationships between modules.
        
        Args:
            root_dir: Root directory of the codebase
            python_modules: List of Python modules found in the codebase
            
        Returns:
            List of import relationships
        """
        import_relationships = []
        module_paths = {module["name"]: module["path"] for module in python_modules}
        
        for module in python_modules:
            module_path = os.path.join(root_dir, module["path"])
            
            try:
                with open(module_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract import statements using regex
                import_statements = []
                
                # Match regular imports: import module, import module.submodule
                import_matches = re.finditer(r'import\s+([\w.]+)(?:\s+as\s+\w+)?', content)
                for match in import_matches:
                    import_statements.append(match.group(1))
                
                # Match from imports: from module import something
                from_matches = re.finditer(r'from\s+([\w.]+)\s+import', content)
                for match in from_matches:
                    import_statements.append(match.group(1))
                
                # Check which imports refer to modules in our codebase
                for import_stmt in import_statements:
                    # Split by dots to handle nested imports
                    parts = import_stmt.split('.')
                    base_module = parts[0]
                    
                    if base_module in module_paths:
                        # This is a local module import
                        import_relationships.append({
                            "source": module["path"],
                            "target": module_paths[base_module],
                            "import_statement": import_stmt
                        })
                        
            except Exception as e:
                logger.error(f"Error analyzing imports in {module_path}: {str(e)}")
        
        return import_relationships
    
    def find_files_by_pattern(
        self, 
        root_dir: str, 
        pattern: str,
        search_content: bool = False
    ) -> List[str]:
        """
        Find files that match a pattern, either by name or content.
        
        Args:
            root_dir: Root directory to search in
            pattern: Pattern to search for
            search_content: Whether to search in file content
            
        Returns:
            List of matching file paths
        """
        if not os.path.isdir(root_dir):
            return []
        
        # Compile the pattern
        try:
            regex = re.compile(pattern)
        except re.error:
            # Treat as literal string if not a valid regex
            regex = re.compile(re.escape(pattern))
        
        matching_files = []
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Skip excluded directories
            dirnames[:] = [d for d in dirnames if d not in self.exclude_dirs]
            
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(file_path, root_dir)
                
                # Check if filename matches
                if regex.search(filename):
                    matching_files.append(rel_path)
                    continue
                
                # Skip if we're not searching content
                if not search_content:
                    continue
                    
                # Get the extension
                _, extension = os.path.splitext(filename)
                
                # Only search text files
                if extension not in self.supported_extensions:
                    continue
                    
                # Check file size
                try:
                    if os.path.getsize(file_path) > self.max_file_size:
                        continue
                        
                    # Search file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if regex.search(content):
                            matching_files.append(rel_path)
                    
                except Exception:
                    # Skip files with errors
                    pass
        
        return matching_files
    
    def search_code_knowledge(
        self, 
        query: str, 
        max_results: int = 5,
        file_pattern: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for code files in the knowledge repository that match a query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            file_pattern: Optional pattern to filter files
            
        Returns:
            List of matching code files with content and metadata
        """
        # Prepare filter metadata
        filter_metadata = {"type": "code_file"}
        
        # Use the knowledge repository to search
        results = self.knowledge_repository.search_knowledge(
            query=query,
            k=max_results,
            filter_metadata=filter_metadata
        )
        
        # Filter by file pattern if provided
        if file_pattern and results:
            try:
                pattern = re.compile(file_pattern)
                results = [
                    r for r in results 
                    if pattern.search(r.get("metadata", {}).get("file_path", ""))
                ]
            except re.error:
                # If invalid regex, just filter by substring
                results = [
                    r for r in results 
                    if file_pattern in r.get("metadata", {}).get("file_path", "")
                ]
        
        return results
    
    def get_code_file(self, file_path: str) -> Dict[str, Any]:
        """
        Get a code file from the knowledge repository by path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file content and metadata
        """
        # Search for the file in the repository
        results = self.knowledge_repository.search_knowledge(
            query=file_path,
            k=1,
            filter_metadata={"type": "code_file", "file_path": file_path}
        )
        
        if not results:
            # Try a more general search
            results = self.knowledge_repository.search_knowledge(
                query=file_path,
                k=1,
                filter_metadata={"type": "code_file"}
            )
        
        if results:
            return results[0]
        
        return {
            "content": "",
            "metadata": {},
            "error": f"File not found: {file_path}"
        }
    
    def _get_language_from_extension(self, extension: str) -> str:
        """
        Map file extension to programming language.
        
        Args:
            extension: File extension without the dot
            
        Returns:
            Programming language name
        """
        extension_map = {
            "py": "python",
            "js": "javascript",
            "jsx": "javascript",
            "ts": "typescript",
            "tsx": "typescript",
            "html": "html",
            "css": "css",
            "java": "java",
            "c": "c",
            "cpp": "cpp",
            "cs": "csharp",
            "go": "go",
            "rb": "ruby",
            "php": "php",
            "swift": "swift",
            "kt": "kotlin",
            "rs": "rust",
            "json": "json",
            "yaml": "yaml",
            "yml": "yaml",
            "md": "markdown",
            "txt": "text",
            "sh": "shell",
            "bat": "batch",
            "ps1": "powershell"
        }
        
        return extension_map.get(extension.lower(), "unknown")