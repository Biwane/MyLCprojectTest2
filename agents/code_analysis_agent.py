"""
Code Analysis Agent Module

This module implements the CodeAnalysisAgent class, which specializes in analyzing
code, understanding its structure, and proposing modifications based on evolution requests.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Union
import re

from agents.base_agent import BaseAgent
from core.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)

class CodeAnalysisAgent(BaseAgent):
    """
    Agent specialized in code analysis and proposing code modifications.
    
    This agent can understand codebases, analyze dependencies between modules,
    identify points where modifications are needed, and propose specific changes
    to implement feature requests or improvements.
    """
    
    def __init__(
        self, 
        agent_executor,
        role: str = "code_analyst",
        config: Dict[str, Any] = None,
        knowledge_repository: Optional[KnowledgeRepository] = None
    ):
        """
        Initialize the code analysis agent.
        
        Args:
            agent_executor: The LangChain agent executor
            role: The specific role of this agent
            config: Configuration dictionary with agent settings
            knowledge_repository: Knowledge repository for accessing shared information
        """
        config = config or {}
        super().__init__(agent_executor, role, config, knowledge_repository)
        
        # Code analysis specific configuration
        self.supported_languages = config.get("supported_languages", ["python", "javascript", "java", "html", "css"])
        self.max_file_size = config.get("max_file_size", 1024 * 1024)  # 1MB default
        
        logger.debug(f"Initialized CodeAnalysisAgent with role: {role}")
    
    def get_capabilities(self) -> List[str]:
        """
        Get the list of capabilities this agent has.
        
        Returns:
            List of capability descriptions
        """
        return [
            "Code structure analysis",
            "Identification of modification points for feature requests",
            "Dependency mapping between modules and components",
            "Proposing specific code changes with diffs",
            "Understanding of software architectural patterns",
            "Code quality assessment"
        ]
    
    def analyze_codebase(self, query: str, relevant_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze a codebase to understand its structure and propose modifications
        based on a specific query or feature request.
        
        Args:
            query: The analysis query or feature request description
            relevant_files: Optional list of relevant file paths to focus the analysis on
            
        Returns:
            Analysis results with insights and proposed modifications
        """
        # Create the analysis prompt
        analysis_prompt = self._create_analysis_prompt(query, relevant_files)
        
        # Execute the task
        result = self.execute_task(analysis_prompt)
        
        # Process and structure the analysis results
        structured_results = self._structure_analysis_results(result, query)
        
        return structured_results
    
    def _create_analysis_prompt(self, query: str, relevant_files: Optional[List[str]] = None) -> str:
        """
        Create a prompt for code analysis based on the query and relevant files.
        
        Args:
            query: The analysis query or feature request
            relevant_files: Optional list of relevant file paths
            
        Returns:
            Formatted analysis prompt
        """
        prompt = f"""
        I need you to analyze code to help with the following request:
        
        REQUEST:
        {query}
        
        """
        
        if relevant_files:
            prompt += "Focus your analysis on these files which are most relevant to the request:\n"
            for file_path in relevant_files:
                prompt += f"- {file_path}\n"
            
            # Try to retrieve content for each file
            prompt += "\nHere is the content of the relevant files:\n\n"
            
            for file_path in relevant_files:
                if self.knowledge_repository:
                    # Search for the file in the knowledge repository
                    results = self.knowledge_repository.search_knowledge(
                        query=file_path,
                        filter_metadata={"type": "code_file"}
                    )
                    
                    if results:
                        file_content = results[0].get("content", "")
                        if file_content:
                            prompt += f"FILE: {file_path}\n```\n{file_content}\n```\n\n"
                        else:
                            prompt += f"FILE: {file_path}\n(Content not available)\n\n"
                    else:
                        prompt += f"FILE: {file_path}\n(File not found in knowledge repository)\n\n"
                else:
                    prompt += f"FILE: {file_path}\n(Unable to access file content)\n\n"
        
        prompt += """
        Based on your analysis, please provide:
        
        1. Overall Code Understanding: A summary of the code structure and how it relates to the request
        2. Key Components: The main components or modules that would need to be modified
        3. Dependencies: Dependencies between components that might be affected by changes
        4. Modification Points: Specific points in the code where changes would be needed
        5. Proposed Changes: Detailed description of the changes needed, with code examples where helpful
        6. Implementation Plan: A step-by-step plan for implementing the changes
        
        Please be specific and provide code snippets or pseudo-code where appropriate to illustrate your proposed changes.
        """
        
        return prompt
    
    def _structure_analysis_results(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Structure the raw analysis results into a consistent format.
        
        Args:
            result: Raw execution result
            query: Original analysis query
            
        Returns:
            Structured analysis results
        """
        output = result.get("output", "")
        
        # Extract sections from the output using simple heuristics
        sections = {
            "code_understanding": self._extract_section(output, "Overall Code Understanding", "Key Components"),
            "key_components": self._extract_section(output, "Key Components", "Dependencies"),
            "dependencies": self._extract_section(output, "Dependencies", "Modification Points"),
            "modification_points": self._extract_section(output, "Modification Points", "Proposed Changes"),
            "proposed_changes": self._extract_section(output, "Proposed Changes", "Implementation Plan"),
            "implementation_plan": self._extract_section(output, "Implementation Plan", None)
        }
        
        # Extract code snippets from proposed changes
        code_snippets = self._extract_code_snippets(sections["proposed_changes"])
        
        # Create structured analysis result
        analysis_result = {
            "query": query,
            "analysis": sections,
            "code_snippets": code_snippets,
            "raw_output": output
        }
        
        return analysis_result
    
    def _extract_section(self, text: str, section_start: str, section_end: Optional[str] = None) -> str:
        """
        Extract a section from text based on starting and ending headers.
        
        Args:
            text: The text to extract from
            section_start: The header that starts the section
            section_end: The header that ends the section (if None, goes to the end)
            
        Returns:
            The extracted section text or empty string if not found
        """
        if section_start not in text:
            return ""
        
        # Split at the section start
        parts = text.split(section_start, 1)
        if len(parts) < 2:
            return ""
        
        # Extract the content after the section start
        content = parts[1].strip()
        
        # If there's a section end, split at that point
        if section_end and section_end in content:
            content = content.split(section_end, 1)[0].strip()
        
        return content
    
    def _extract_code_snippets(self, text: str) -> List[Dict[str, str]]:
        """
        Extract code snippets from text surrounded by code blocks.
        
        Args:
            text: The text to extract code snippets from
            
        Returns:
            List of dictionaries with language and code content
        """
        snippets = []
        
        # Look for code blocks (```language\ncode\n```)
        pattern = r'```(\w*)\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for language, code in matches:
            # If no language specified, try to guess
            if not language:
                language = "text"
            
            snippets.append({
                "language": language,
                "code": code.strip()
            })
        
        return snippets
    
    def generate_diff(self, original_code: str, proposed_changes: str, file_path: str) -> Dict[str, Any]:
        """
        Generate a diff between original code and proposed changes.
        
        Args:
            original_code: The original code
            proposed_changes: Description of the changes to make
            file_path: Path to the file being modified
            
        Returns:
            Dictionary with diff information
        """
        # Create the diff generation prompt
        prompt = f"""
        Please generate a unified diff for the following code modification:
        
        Original file: {file_path}
        
        Original code:
        ```
        {original_code}
        ```
        
        Proposed changes:
        {proposed_changes}
        
        Generate a unified diff showing exactly what needs to be changed in the file.
        Use the standard unified diff format with context lines.
        Show only the sections that need to be modified with a few lines of context around each change.
        """
        
        # Execute the task
        result = self.execute_task(prompt)
        
        # Extract the diff from the result
        diff_text = self._extract_diff_from_result(result.get("output", ""))
        
        return {
            "file_path": file_path,
            "diff": diff_text,
            "success": bool(diff_text)
        }
    
    def _extract_diff_from_result(self, text: str) -> str:
        """
        Extract a unified diff from text.
        
        Args:
            text: Text that may contain a diff
            
        Returns:
            Extracted diff or empty string if not found
        """
        # Try to find the diff content between code blocks
        pattern = r'```diff\n(.*?)\n```'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # If no code block with diff, look for lines starting with + or -
        diff_lines = []
        in_diff = False
        
        for line in text.split('\n'):
            stripped = line.strip()
            if stripped.startswith('+++') or stripped.startswith('---') or stripped.startswith('@@'):
                in_diff = True
                diff_lines.append(line)
            elif in_diff and (stripped.startswith('+') or stripped.startswith('-') or stripped.startswith(' ')):
                diff_lines.append(line)
            elif in_diff and stripped == '':
                diff_lines.append(line)
            elif in_diff:
                # End of diff section
                in_diff = False
        
        return '\n'.join(diff_lines) if diff_lines else ""
    
    def get_role_description(self) -> str:
        """
        Get a description of this agent's role.
        
        Returns:
            Description of the agent's role
        """
        return (
            f"I am a {self.role} agent specializing in code analysis and modification. "
            f"I can understand codebases, analyze dependencies between components, "
            f"identify points where modifications are needed, and propose specific "
            f"changes to implement feature requests or improvements."
        )