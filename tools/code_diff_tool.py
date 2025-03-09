"""
Code Diff Tool Module

This module provides tools for generating and applying code differences (diffs)
to help with proposing and implementing code changes.
"""

import os
import logging
import difflib
import re
from typing import Dict, Any, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class CodeDiffTool:
    """
    Tool for generating and applying code differences (diffs).
    
    This tool helps with proposing code changes, generating readable diffs,
    and applying those changes to actual code files.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the code diff tool.
        
        Args:
            config: Configuration dictionary with tool settings
        """
        self.config = config
        self.context_lines = config.get("context_lines", 3)
        self.output_dir = config.get("output_dir", "output")
        self.patch_dir = os.path.join(self.output_dir, "patches")
        
        # Create directories if they don't exist
        os.makedirs(self.patch_dir, exist_ok=True)
        
        logger.debug(f"Initialized CodeDiffTool with context_lines: {self.context_lines}")
    
    def generate_diff(
        self, 
        original_content: str, 
        modified_content: str,
        file_path: str,
        context_lines: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a unified diff between original content and modified content.
        
        Args:
            original_content: The original content
            modified_content: The modified content
            file_path: Path of the file being modified (for reference)
            context_lines: Number of context lines to include in the diff
            
        Returns:
            Dictionary with diff information
        """
        context_lines = context_lines or self.context_lines
        
        # Split content into lines
        original_lines = original_content.splitlines(keepends=True)
        modified_lines = modified_content.splitlines(keepends=True)
        
        # Get filename from path
        filename = os.path.basename(file_path)
        
        # Generate unified diff
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            n=context_lines
        )
        
        # Convert to string
        diff_text = "".join(diff)
        
        # Return information
        return {
            "file_path": file_path,
            "diff": diff_text,
            "has_changes": bool(diff_text),
            "summary": self._summarize_diff(diff_text)
        }
    
    def generate_code_change(
        self, 
        original_code: str,
        change_description: str,
        file_path: str,
        language: str
    ) -> Dict[str, Any]:
        """
        Generate a modified version of the code based on a change description.
        
        Args:
            original_code: The original code
            change_description: Description of the changes to make
            file_path: Path to the file being modified
            language: Programming language of the code
            
        Returns:
            Dictionary with the original code, modified code, and diff
        """
        # Format the prompt for the LLM to generate the modified code
        prompt = f"""
        Given the following code and change request, please generate the modified version of the code.
        Do not explain the changes, just provide the complete modified code.
        
        Original file: {file_path}
        Language: {language}
        
        Original code:
        ```{language}
        {original_code}
        ```
        
        Change request:
        {change_description}
        
        Please provide the complete modified code below:
        ```{language}
        """
        
        # Note: This method would typically use an LLM to generate the modified code
        # For this implementation, we'll just return a mock response
        
        # As a placeholder, we'll just make a simple modification
        modified_code = self._mock_code_modification(original_code, change_description)
        
        # Generate the diff
        diff_result = self.generate_diff(original_code, modified_code, file_path)
        
        return {
            "file_path": file_path,
            "original_code": original_code,
            "modified_code": modified_code,
            "diff": diff_result["diff"],
            "has_changes": diff_result["has_changes"],
            "change_description": change_description
        }
    
    def _mock_code_modification(self, original_code: str, change_description: str) -> str:
        """
        Mock implementation for code modification based on a description.
        
        Args:
            original_code: The original code
            change_description: Description of the changes to make
            
        Returns:
            Modified code
        """
        # This is a placeholder for LLM-generated code modifications
        # In a real implementation, this would use an API call to a language model
        
        # For this example, we'll just make a simple change based on keywords in the description
        modified_code = original_code
        
        # Add a comment at the top with the change description
        modified_code = f"# Modified based on: {change_description}\n" + modified_code
        
        return modified_code
    
    def apply_diff(self, file_path: str, diff_content: str) -> Dict[str, Any]:
        """
        Apply a diff to a file on disk.
        
        Args:
            file_path: Path to the file to modify
            diff_content: The diff content to apply
            
        Returns:
            Dictionary with the result of the operation
        """
        import subprocess
        
        try:
            # Write the diff to a temporary file
            timestamp = int(os.path.getmtime(os.path.abspath(__file__)))
            patch_path = os.path.join(self.patch_dir, f"patch_{timestamp}.diff")
            
            with open(patch_path, 'w', encoding='utf-8') as f:
                f.write(diff_content)
            
            # Apply the patch using the patch command
            result = subprocess.run(
                ["patch", file_path, patch_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "file_path": file_path,
                    "patch_path": patch_path,
                    "output": result.stdout
                }
            else:
                return {
                    "success": False,
                    "file_path": file_path,
                    "patch_path": patch_path,
                    "error": f"Failed to apply patch: {result.stderr}"
                }
            
        except Exception as e:
            logger.error(f"Error applying diff to {file_path}: {str(e)}")
            return {
                "success": False,
                "file_path": file_path,
                "error": f"Error: {str(e)}"
            }
    
    def _summarize_diff(self, diff_text: str) -> Dict[str, int]:
        """
        Summarize a diff by counting additions and deletions.
        
        Args:
            diff_text: The diff text to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        lines = diff_text.splitlines()
        additions = sum(1 for line in lines if line.startswith('+') and not line.startswith('+++'))
        deletions = sum(1 for line in lines if line.startswith('-') and not line.startswith('---'))
        
        return {
            "lines_added": additions,
            "lines_deleted": deletions,
            "total_changes": additions + deletions
        }
    
    def save_modification_proposal(
        self, 
        change_result: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save a code modification proposal to a file.
        
        Args:
            change_result: Result from generate_code_change
            output_file: Optional custom output file path
            
        Returns:
            Dictionary with the path to the saved proposal
        """
        if not output_file:
            # Generate a filename based on the original file
            filename = os.path.basename(change_result["file_path"])
            base, ext = os.path.splitext(filename)
            output_file = os.path.join(self.output_dir, f"{base}_modified{ext}")
        
        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Write the modified code to the file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(change_result["modified_code"])
            
            # Save the diff to a separate file
            diff_path = f"{output_file}.diff"
            with open(diff_path, 'w', encoding='utf-8') as f:
                f.write(change_result["diff"])
            
            return {
                "success": True,
                "original_file": change_result["file_path"],
                "modified_file": output_file,
                "diff_file": diff_path
            }
            
        except Exception as e:
            logger.error(f"Error saving modification proposal: {str(e)}")
            return {
                "success": False,
                "error": f"Error: {str(e)}"
            }