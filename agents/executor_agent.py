"""
Executor Agent Module

This module implements the ExecutorAgent class, which specializes in executing
concrete tasks and implementing solutions based on plans and specifications provided
by other agents in the team.
"""

import logging
import os
import time
from typing import Dict, Any, List, Optional, Union
import json

from agents.base_agent import BaseAgent
from core.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)

class ExecutorAgent(BaseAgent):
    """
    Agent specialized in execution and implementation of concrete tasks.
    
    This agent takes plans and specifications and turns them into actual
    implementations, including code generation, configurations, or other
    executable solutions.
    """
    
    def __init__(
        self, 
        agent_executor,
        role: str = "executor",
        config: Dict[str, Any] = None,
        knowledge_repository: Optional[KnowledgeRepository] = None
    ):
        """
        Initialize the executor agent.
        
        Args:
            agent_executor: The LangChain agent executor
            role: The specific role of this executor agent
            config: Configuration dictionary with agent settings
            knowledge_repository: Knowledge repository for accessing shared information
        """
        config = config or {}
        super().__init__(agent_executor, role, config, knowledge_repository)
        
        # Executor-specific configuration
        self.execution_timeout = config.get("execution_timeout", 120)
        self.validate_results = config.get("validate_results", True)
        self.output_dir = config.get("output_dir", "output")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.debug(f"Initialized ExecutorAgent with role: {role}")
    
    def get_capabilities(self) -> List[str]:
        """
        Get the list of capabilities this agent has.
        
        Returns:
            List of capability descriptions
        """
        return [
            "Implementation of solutions from specifications",
            "Code generation and scripting",
            "System configuration and setup",
            "File and resource management",
            "Command execution and automation",
            "Integration between components",
            "Testing and validation"
        ]
    
    def execute_implementation(
        self, 
        specifications: Dict[str, Any], 
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a task based on provided specifications.
        
        Args:
            specifications: Detailed specifications for the implementation
            context: Optional additional context
            
        Returns:
            Dictionary containing implementation results
        """
        # Combine specifications and context into a prompt
        implementation_prompt = self._create_implementation_prompt(specifications, context)
        
        # Start the execution timer
        start_time = time.time()
        
        # Execute the implementation
        result = self.execute_task(implementation_prompt)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        logger.info(f"Implementation executed in {execution_time:.2f} seconds")
        
        # Process the result and extract artifacts
        processed_result = self._process_implementation_result(result, specifications)
        
        # Validate the result if configured
        if self.validate_results:
            validation_result = self._validate_implementation(processed_result, specifications)
            processed_result["validation"] = validation_result
        
        return processed_result
    
    def _create_implementation_prompt(
        self, 
        specifications: Dict[str, Any], 
        context: Optional[str] = None
    ) -> str:
        """
        Create an implementation prompt based on specifications.
        
        Args:
            specifications: The specifications for the implementation
            context: Optional additional context
            
        Returns:
            Formatted implementation prompt
        """
        # Extract key information from specifications
        task_type = specifications.get("type", "general")
        description = specifications.get("description", "Implement the solution")
        requirements = specifications.get("requirements", [])
        deliverables = specifications.get("deliverables", [])
        constraints = specifications.get("constraints", [])
        
        # Build the prompt
        prompt_parts = [
            f"Task: {description}",
            "",
            "Implementation Specifications:"
        ]
        
        # Add requirements if any
        if requirements:
            prompt_parts.append("\nRequirements:")
            for i, req in enumerate(requirements, 1):
                prompt_parts.append(f"{i}. {req}")
        
        # Add deliverables if any
        if deliverables:
            prompt_parts.append("\nDeliverables:")
            for i, deliv in enumerate(deliverables, 1):
                prompt_parts.append(f"{i}. {deliv}")
        
        # Add constraints if any
        if constraints:
            prompt_parts.append("\nConstraints:")
            for i, constraint in enumerate(constraints, 1):
                prompt_parts.append(f"{i}. {constraint}")
        
        # Add additional context if provided
        if context:
            prompt_parts.append("\nAdditional Context:")
            prompt_parts.append(context)
        
        # Add task-specific instructions
        prompt_parts.append("\nImplementation Instructions:")
        
        if task_type == "code_generation":
            prompt_parts.append(
                "Please implement the code according to the specifications. "
                "Include clear comments, error handling, and follow best practices. "
                "Format your response with the actual code, followed by a brief explanation of how it works."
            )
        elif task_type == "configuration":
            prompt_parts.append(
                "Please provide the configuration settings and files according to the specifications. "
                "Include clear instructions on how to apply the configuration. "
                "Format your response with the configuration content, followed by implementation steps."
            )
        elif task_type == "documentation":
            prompt_parts.append(
                "Please create the documentation according to the specifications. "
                "Format your response as complete documentation ready for use."
            )
        else:
            prompt_parts.append(
                "Please implement the solution according to the specifications. "
                "Provide a complete and detailed implementation that can be directly used."
            )
        
        # Add format instructions for output
        prompt_parts.append("\nFormat your response as follows:")
        prompt_parts.append("1. Implementation: Your solution implementation")
        prompt_parts.append("2. Explanation: Brief explanation of how your implementation works")
        prompt_parts.append("3. Usage Instructions: How to use or apply your implementation")
        prompt_parts.append("4. Notes: Any important notes, assumptions, or limitations")
        
        if task_type == "code_generation":
            prompt_parts.append("\nFor code, use proper formatting with language-specific syntax highlighting.")
            
        # Combine all parts into the final prompt
        return "\n".join(prompt_parts)
    
    def _process_implementation_result(
        self, 
        result: Dict[str, Any], 
        specifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process the implementation result and extract any artifacts.
        
        Args:
            result: Raw execution result
            specifications: Original specifications
            
        Returns:
            Processed result with extracted artifacts
        """
        output = result.get("output", "")
        task_type = specifications.get("type", "general")
        
        # Initialize processed result
        processed = {
            "implementation": "",
            "explanation": "",
            "usage_instructions": "",
            "notes": "",
            "artifacts": [],
            "original_output": output
        }
        
        # Try to extract sections from the output
        if "Implementation:" in output:
            parts = output.split("Implementation:", 1)
            if len(parts) > 1:
                implementation_text = parts[1].split("\n\n", 1)[0]
                if len(parts[1].split("\n\n")) > 1:
                    remaining = parts[1].split("\n\n", 1)[1]
                else:
                    remaining = ""
                processed["implementation"] = implementation_text.strip()
            else:
                remaining = output
        else:
            # If no explicit Implementation section, use the output until the first section header
            first_section = min(
                [output.find(s) for s in ["Explanation:", "Usage Instructions:", "Notes:"] if s in output] + [len(output)]
            )
            processed["implementation"] = output[:first_section].strip()
            remaining = output[first_section:]
        
        # Extract explanation
        if "Explanation:" in remaining:
            parts = remaining.split("Explanation:", 1)
            if len(parts) > 1:
                explanation_text = parts[1].split("\n\n", 1)[0]
                processed["explanation"] = explanation_text.strip()
                if len(parts[1].split("\n\n")) > 1:
                    remaining = parts[1].split("\n\n", 1)[1]
                else:
                    remaining = ""
        
        # Extract usage instructions
        if "Usage Instructions:" in remaining:
            parts = remaining.split("Usage Instructions:", 1)
            if len(parts) > 1:
                usage_text = parts[1].split("\n\n", 1)[0]
                processed["usage_instructions"] = usage_text.strip()
                if len(parts[1].split("\n\n")) > 1:
                    remaining = parts[1].split("\n\n", 1)[1]
                else:
                    remaining = ""
        
        # Extract notes
        if "Notes:" in remaining:
            parts = remaining.split("Notes:", 1)
            if len(parts) > 1:
                notes_text = parts[1].strip()
                processed["notes"] = notes_text
        
        # Extract code artifacts for code_generation tasks
        if task_type == "code_generation":
            artifacts = self._extract_code_artifacts(output, specifications)
            processed["artifacts"] = artifacts
            
            # Create files for the artifacts
            output_files = []
            for artifact in artifacts:
                file_path = self._save_artifact(artifact)
                if file_path:
                    output_files.append(file_path)
            
            processed["output_files"] = output_files
        
        return processed
    
    def _extract_code_artifacts(self, output: str, specifications: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract code artifacts from the output.
        
        Args:
            output: The raw output text
            specifications: The original specifications
            
        Returns:
            List of extracted code artifacts
        """
        artifacts = []
        
        # Look for code blocks in markdown format (```language...```)
        import re
        code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', output, re.DOTALL)
        
        # Process each code block
        for i, (language, code) in enumerate(code_blocks):
            language = language.strip() if language else "txt"
            
            # Determine filename
            language_to_extension = {
                "python": "py",
                "java": "java",
                "javascript": "js",
                "typescript": "ts",
                "html": "html",
                "css": "css",
                "json": "json",
                "xml": "xml",
                "yaml": "yaml",
                "sql": "sql",
                "bash": "sh",
                "shell": "sh",
                "apex": "cls",
                "visualforce": "page",
                "soql": "soql",
                "aura": "cmp",
                "lwc": "js"
            }
            
            # Get the extension for the language
            extension = language_to_extension.get(language.lower(), "txt")
            
            # Try to determine a meaningful filename
            filename = None
            
            # Look for class/function definitions or comments that might suggest a filename
            if language.lower() == "python":
                class_match = re.search(r'class\s+([A-Za-z0-9_]+)', code)
                if class_match:
                    filename = f"{class_match.group(1).lower()}.{extension}"
                else:
                    def_match = re.search(r'def\s+([A-Za-z0-9_]+)', code)
                    if def_match:
                        filename = f"{def_match.group(1).lower()}.{extension}"
            elif language.lower() in ["java", "apex"]:
                class_match = re.search(r'class\s+([A-Za-z0-9_]+)', code)
                if class_match:
                    filename = f"{class_match.group(1)}.{extension}"
            elif language.lower() in ["javascript", "typescript"]:
                class_match = re.search(r'class\s+([A-Za-z0-9_]+)', code)
                if class_match:
                    filename = f"{class_match.group(1)}.{extension}"
                else:
                    function_match = re.search(r'function\s+([A-Za-z0-9_]+)', code)
                    if function_match:
                        filename = f"{function_match.group(1)}.{extension}"
            
            # Fallback if no specific filename could be determined
            if not filename:
                filename = f"artifact_{i+1}.{extension}"
            
            # Create the artifact entry
            artifact = {
                "type": "code",
                "language": language,
                "content": code,
                "filename": filename
            }
            
            artifacts.append(artifact)
        
        # If no artifacts were found using markdown code blocks, try alternative approaches
        if not artifacts:
            # Try to find code sections based on indentation and context
            lines = output.split("\n")
            in_code_block = False
            current_language = None
            current_code = []
            
            for line in lines:
                # Check for language indicators
                if not in_code_block and ":" in line and any(lang in line.lower() for lang in ["code", "python", "java", "javascript", "html"]):
                    in_code_block = True
                    language_indicator = line.lower()
                    
                    if "python" in language_indicator:
                        current_language = "python"
                    elif "java" in language_indicator and "javascript" not in language_indicator:
                        current_language = "java"
                    elif "javascript" in language_indicator:
                        current_language = "javascript"
                    elif "html" in language_indicator:
                        current_language = "html"
                    elif "apex" in language_indicator:
                        current_language = "apex"
                    else:
                        current_language = "txt"
                    
                    continue
                
                # Check for end of code block
                if in_code_block and (not line.strip() or line.startswith("This code") or line.startswith("The code")):
                    if current_code:
                        extension = language_to_extension.get(current_language.lower(), "txt")
                        filename = f"extracted_code_{len(artifacts)+1}.{extension}"
                        
                        artifact = {
                            "type": "code",
                            "language": current_language,
                            "content": "\n".join(current_code),
                            "filename": filename
                        }
                        
                        artifacts.append(artifact)
                        
                        in_code_block = False
                        current_language = None
                        current_code = []
                    
                    continue
                
                # Add code lines
                if in_code_block:
                    current_code.append(line)
            
            # Add the last code block if there is one
            if in_code_block and current_code:
                extension = language_to_extension.get(current_language.lower(), "txt")
                filename = f"extracted_code_{len(artifacts)+1}.{extension}"
                
                artifact = {
                    "type": "code",
                    "language": current_language,
                    "content": "\n".join(current_code),
                    "filename": filename
                }
                
                artifacts.append(artifact)
        
        return artifacts
    
    def _save_artifact(self, artifact: Dict[str, Any]) -> Optional[str]:
        """
        Save an artifact to a file.
        
        Args:
            artifact: The artifact to save
            
        Returns:
            Path to the saved file or None if save failed
        """
        artifact_type = artifact.get("type")
        
        if artifact_type == "code":
            # Get artifact properties
            filename = artifact.get("filename", "artifact.txt")
            content = artifact.get("content", "")
            
            # Create full path
            file_path = os.path.join(self.output_dir, filename)
            
            try:
                # Create directory if needed
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Write content to file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                
                logger.debug(f"Saved artifact to {file_path}")
                return file_path
                
            except Exception as e:
                logger.error(f"Error saving artifact to {file_path}: {str(e)}")
                return None
        
        return None
    
    def _validate_implementation(
        self, 
        processed_result: Dict[str, Any], 
        specifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate the implementation against specifications.
        
        Args:
            processed_result: The processed implementation result
            specifications: The original specifications
            
        Returns:
            Validation results
        """
        # For now, a simplified validation
        validation = {
            "passed": True,
            "issues": [],
            "suggestions": []
        }
        
        # Check for empty implementation
        if not processed_result.get("implementation"):
            validation["passed"] = False
            validation["issues"].append("Implementation is empty")
        
        # Check for missing artifacts in code_generation task
        if specifications.get("type") == "code_generation" and not processed_result.get("artifacts"):
            validation["passed"] = False
            validation["issues"].append("No code artifacts found in the implementation")
        
        # Check for missing usage instructions
        if not processed_result.get("usage_instructions"):
            validation["suggestions"].append("Usage instructions are missing or incomplete")
        
        # Task-specific validation
        task_type = specifications.get("type", "general")
        
        if task_type == "code_generation":
            # Check code artifacts for basic issues
            for artifact in processed_result.get("artifacts", []):
                code = artifact.get("content", "")
                language = artifact.get("language", "").lower()
                
                # Check for empty code
                if not code.strip():
                    validation["passed"] = False
                    validation["issues"].append(f"Empty code artifact: {artifact.get('filename')}")
                
                # Very basic syntax checks
                if language == "python":
                    if "import" not in code and "def " not in code and "class " not in code:
                        validation["suggestions"].append(f"Python code may be incomplete: {artifact.get('filename')}")
                elif language == "javascript":
                    if "function" not in code and "class" not in code and "const" not in code and "let" not in code:
                        validation["suggestions"].append(f"JavaScript code may be incomplete: {artifact.get('filename')}")
        
        return validation
    
    def generate_code(
        self, 
        code_specs: Dict[str, Any], 
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Generate code based on specifications.
        
        Args:
            code_specs: Specifications for the code to generate
            language: Programming language to use
            
        Returns:
            Dictionary containing the generated code and metadata
        """
        # Create specialized specifications for code generation
        specifications = {
            "type": "code_generation",
            "description": code_specs.get("description", "Generate code based on specifications"),
            "requirements": code_specs.get("requirements", []),
            "deliverables": code_specs.get("deliverables", []),
            "constraints": code_specs.get("constraints", []),
            "language": language
        }
        
        # Add language-specific context
        language_context = {
            "python": "Use Python 3.8+ features and best practices.",
            "javascript": "Use modern JavaScript (ES6+) features and best practices.",
            "java": "Use Java 11+ features and best practices.",
            "apex": "Follow Salesforce Apex best practices and governor limits."
        }.get(language.lower(), "")
        
        # Add specific language requirements
        if language.lower() == "python":
            specifications["constraints"].append("Follow PEP 8 style guidelines")
            specifications["constraints"].append("Include docstrings for all functions and classes")
        elif language.lower() == "javascript":
            specifications["constraints"].append("Use ES6+ syntax")
            specifications["constraints"].append("Add JSDoc comments for functions")
        elif language.lower() == "apex":
            specifications["constraints"].append("Consider Salesforce governor limits")
            specifications["constraints"].append("Include proper exception handling")
        
        # Execute the implementation with specialized context
        return self.execute_implementation(specifications, language_context)
    
    def configure_system(
        self, 
        config_specs: Dict[str, Any], 
        system_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Generate configuration files or settings.
        
        Args:
            config_specs: Specifications for the configuration
            system_type: Type of system to configure
            
        Returns:
            Dictionary containing the configuration and metadata
        """
        # Create specialized specifications for configuration
        specifications = {
            "type": "configuration",
            "description": config_specs.get("description", "Create configuration based on specifications"),
            "requirements": config_specs.get("requirements", []),
            "deliverables": config_specs.get("deliverables", []),
            "constraints": config_specs.get("constraints", []),
            "system_type": system_type
        }
        
        # Add system-specific context
        system_context = {
            "salesforce": "Configure Salesforce-specific settings and metadata.",
            "web": "Configure web application settings.",
            "database": "Configure database settings and schema.",
            "network": "Configure network-related settings."
        }.get(system_type.lower(), "")
        
        # Execute the implementation with specialized context
        return self.execute_implementation(specifications, system_context)
    
    def create_documentation(
        self, 
        doc_specs: Dict[str, Any], 
        doc_type: str = "user_guide"
    ) -> Dict[str, Any]:
        """
        Create documentation based on specifications.
        
        Args:
            doc_specs: Specifications for the documentation
            doc_type: Type of documentation to create
            
        Returns:
            Dictionary containing the documentation and metadata
        """
        # Create specialized specifications for documentation
        specifications = {
            "type": "documentation",
            "description": doc_specs.get("description", "Create documentation based on specifications"),
            "requirements": doc_specs.get("requirements", []),
            "deliverables": doc_specs.get("deliverables", []),
            "constraints": doc_specs.get("constraints", []),
            "doc_type": doc_type
        }
        
        # Add documentation-specific context
        doc_context = {
            "user_guide": "Create user-facing documentation explaining how to use the system.",
            "api_reference": "Create technical API reference documentation.",
            "technical_spec": "Create a detailed technical specification document.",
            "installation_guide": "Create step-by-step installation instructions."
        }.get(doc_type.lower(), "")
        
        # Execute the implementation with specialized context
        return self.execute_implementation(specifications, doc_context)
    
    def get_role_description(self) -> str:
        """
        Get a description of this agent's role.
        
        Returns:
            Description of the agent's role
        """
        return (
            f"I am a {self.role} agent specializing in implementing solutions and executing tasks. "
            f"I can generate code, create configurations, implement designs, and produce working "
            f"artifacts based on specifications. I focus on turning plans and requirements into "
            f"concrete, functional implementations."
        )
