"""
Code Generation Tool Module

This module provides tools for generating code based on specifications or requirements.
It leverages language models to create code in various programming languages and
can handle different types of code generation tasks.
"""

import logging
import os
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class CodeGenerationTool:
    """
    Tool for generating code based on specifications or requirements.
    
    This tool leverages language models to generate code in various programming
    languages and can handle different types of code generation tasks including
    functions, classes, scripts, or complete applications.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the code generation tool.
        
        Args:
            config: Configuration dictionary with code generation settings
        """
        self.config = config
        self.model_name = config.get("model", "gpt-4o")
        self.temperature = config.get("temperature", 0.1)
        self.output_dir = config.get("output_dir", "output")
        self.language_support = config.get("language_support", [
            "python", "javascript", "java", "csharp", "apex"
        ])
        
        # Initialize language model
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.debug(f"Initialized CodeGenerationTool with model: {self.model_name}")
    
    def generate_code(
        self, 
        specification: str, 
        language: str, 
        code_type: str = "function",
        save_to_file: bool = True,
        filename: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate code based on a specification.
        
        Args:
            specification: Description of what the code should do
            language: Programming language to use
            code_type: Type of code to generate (function, class, script, app)
            save_to_file: Whether to save the generated code to a file
            filename: Optional filename to save the code to
            additional_context: Additional context or requirements
            
        Returns:
            Dictionary containing the generated code and metadata
        """
        # Check if the language is supported
        if language.lower() not in [lang.lower() for lang in self.language_support]:
            logger.warning(f"Language {language} not in explicitly supported languages: {self.language_support}")
        
        # Create the prompt for code generation
        prompt = self._create_code_generation_prompt(
            specification, language, code_type, additional_context
        )
        
        # Generate the code
        try:
            logger.debug(f"Generating {code_type} in {language}")
            response = self.llm.invoke(prompt)
            
            # Extract code from the response
            generated_code, code_explanation = self._extract_code_from_response(response.content, language)
            
            # Determine filename if not provided
            file_path = None
            if save_to_file:
                file_path = self._save_code_to_file(generated_code, language, filename)
            
            return {
                "code": generated_code,
                "language": language,
                "explanation": code_explanation,
                "file_path": file_path,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return {
                "code": f"# Error generating code: {str(e)}",
                "language": language,
                "explanation": f"An error occurred during code generation: {str(e)}",
                "file_path": None,
                "success": False
            }
    
    def _create_code_generation_prompt(
        self,
        specification: str,
        language: str,
        code_type: str,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Create a prompt for code generation.
        
        Args:
            specification: Description of what the code should do
            language: Programming language to use
            code_type: Type of code to generate
            additional_context: Additional context or requirements
            
        Returns:
            Formatted prompt string
        """
        # Base prompt template
        prompt = f"""
        Generate {language} code for the following specification:
        
        SPECIFICATION:
        {specification}
        """
        
        # Add code type specific instructions
        if code_type.lower() == "function":
            prompt += "\nCreate a well-structured function that accomplishes this task."
        elif code_type.lower() == "class":
            prompt += "\nCreate a well-structured class with appropriate methods."
        elif code_type.lower() == "script":
            prompt += "\nCreate a complete script that can be executed."
        elif code_type.lower() == "app":
            prompt += "\nCreate a basic application structure for this requirement."
        
        # Add language-specific best practices
        prompt += f"\n\nFollow these {language} best practices:"
        
        if language.lower() == "python":
            prompt += """
            - Follow PEP 8 style guidelines
            - Include docstrings for functions and classes
            - Use type hints where appropriate
            - Handle errors with try/except blocks
            - Use meaningful variable and function names
            """
        elif language.lower() == "javascript":
            prompt += """
            - Use modern ES6+ syntax
            - Add JSDoc comments for functions
            - Handle errors appropriately
            - Use const and let instead of var
            - Follow standard JavaScript conventions
            """
        elif language.lower() == "java":
            prompt += """
            - Follow Java naming conventions
            - Include JavaDoc comments
            - Handle exceptions appropriately
            - Use proper access modifiers
            - Follow object-oriented principles
            """
        elif language.lower() == "csharp":
            prompt += """
            - Follow C# naming conventions
            - Include XML documentation comments
            - Use proper exception handling
            - Follow C# coding standards
            - Consider SOLID principles
            """
        elif language.lower() == "apex":
            prompt += """
            - Consider Salesforce governor limits
            - Include proper error handling
            - Follow Salesforce security best practices
            - Include test methods
            - Use bulkified patterns
            """
        
        # Add additional context if provided
        if additional_context:
            prompt += f"\n\nADDITIONAL CONTEXT:\n{additional_context}"
        
        # Add formatting instructions
        prompt += """
        
        FORMAT YOUR RESPONSE AS FOLLOWS:
        1. First provide the complete code solution, formatted with proper syntax highlighting
        2. After the code, provide a brief explanation of how it works
        3. Mention any assumptions made
        4. Suggest potential improvements or alternatives
        
        THE CODE MUST BE ENCLOSED IN A CODE BLOCK WITH THE APPROPRIATE LANGUAGE TAG.
        """
        
        return prompt
    
    def _extract_code_from_response(self, response: str, language: str) -> tuple:
        """
        Extract code and explanation from the response.
        
        Args:
            response: The response from the language model
            language: The programming language
            
        Returns:
            Tuple of (code, explanation)
        """
        code = ""
        explanation = ""
        
        # Try to extract code blocks with markdown formatting
        import re
        code_block_pattern = rf"```(?:{language})?\s*(.*?)\s*```"
        code_blocks = re.findall(code_block_pattern, response, re.DOTALL)
        
        if code_blocks:
            # Get the first code block
            code = code_blocks[0].strip()
            
            # Get explanation after the last code block
            last_code_end = response.rfind("```")
            if last_code_end != -1 and last_code_end + 3 < len(response):
                explanation = response[last_code_end + 3:].strip()
        else:
            # If no code blocks found, try to extract based on context
            lines = response.split("\n")
            code_section = False
            code_lines = []
            explanation_lines = []
            
            for line in lines:
                if not code_section and any(indicator in line.lower() for indicator in ["here's the code", "code:", "solution:"]):
                    code_section = True
                    continue
                elif code_section and any(indicator in line.lower() for indicator in ["explanation:", "how it works:", "here's how"]):
                    code_section = False
                    explanation_lines.append(line)
                    continue
                
                if code_section:
                    code_lines.append(line)
                elif not code_section and line.strip():
                    explanation_lines.append(line)
            
            if code_lines:
                code = "\n".join(code_lines).strip()
            if explanation_lines:
                explanation = "\n".join(explanation_lines).strip()
            
            # If still no code found, assume the whole response is code
            if not code:
                code = response.strip()
        
        return code, explanation
    
    def _save_code_to_file(
        self,
        code: str,
        language: str,
        filename: Optional[str] = None
    ) -> str:
        """
        Save generated code to a file.
        
        Args:
            code: The generated code
            language: The programming language
            filename: Optional filename to use
            
        Returns:
            Path to the saved file
        """
        # Map languages to file extensions
        extensions = {
            "python": "py",
            "javascript": "js",
            "java": "java",
            "csharp": "cs",
            "apex": "cls",
            "html": "html",
            "css": "css",
            "sql": "sql"
        }
        
        # Get the file extension for the language
        extension = extensions.get(language.lower(), "txt")
        
        # Generate a filename if not provided
        if not filename:
            # Try to determine a reasonable filename from the code
            if language.lower() == "python":
                # Look for class or function definitions
                import re
                class_match = re.search(r"class\s+([A-Za-z0-9_]+)", code)
                if class_match:
                    filename = f"{class_match.group(1).lower()}.{extension}"
                else:
                    func_match = re.search(r"def\s+([A-Za-z0-9_]+)", code)
                    if func_match:
                        filename = f"{func_match.group(1).lower()}.{extension}"
            elif language.lower() in ["java", "apex", "csharp"]:
                # Look for class definitions
                import re
                class_match = re.search(r"class\s+([A-Za-z0-9_]+)", code)
                if class_match:
                    filename = f"{class_match.group(1)}.{extension}"
            
            # Default filename if we couldn't determine one
            if not filename:
                timestamp = int(os.path.getmtime(os.path.abspath(__file__)))
                filename = f"generated_code_{timestamp}.{extension}"
        
        # Ensure filename has the correct extension
        if not filename.endswith(f".{extension}"):
            filename = f"{filename}.{extension}"
        
        # Create the full file path
        file_path = os.path.join(self.output_dir, filename)
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the code to the file
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            logger.info(f"Saved generated code to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving code to file: {str(e)}")
            return ""
    
    def implement_function(
        self, 
        function_name: str, 
        description: str, 
        language: str, 
        parameters: Optional[List[Dict[str, str]]] = None,
        return_type: Optional[str] = None,
        save_to_file: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a function based on a description.
        
        Args:
            function_name: Name of the function
            description: Description of what the function should do
            language: Programming language to use
            parameters: List of parameter dictionaries with name and type
            return_type: Return type of the function
            save_to_file: Whether to save the generated function to a file
            
        Returns:
            Dictionary containing the generated function and metadata
        """
        # Create parameter string
        params_str = ""
        if parameters:
            params = []
            for param in parameters:
                param_name = param.get("name", "")
                param_type = param.get("type", "")
                
                if language.lower() == "python":
                    if param_type:
                        params.append(f"{param_name}: {param_type}")
                    else:
                        params.append(param_name)
                elif language.lower() in ["java", "csharp", "apex"]:
                    if param_type:
                        params.append(f"{param_type} {param_name}")
                    else:
                        params.append(f"Object {param_name}")
                elif language.lower() == "javascript":
                    params.append(param_name)
                
            params_str = ", ".join(params)
        
        # Create return type string
        return_str = ""
        if return_type:
            if language.lower() == "python":
                return_str = f" -> {return_type}"
            elif language.lower() in ["java", "csharp", "apex"]:
                return_str = f" Returns: {return_type}"
            elif language.lower() == "javascript":
                return_str = f" @returns {{{return_type}}}"
        
        # Create function specification
        specification = f"""
        Function Name: {function_name}
        Description: {description}
        Parameters: {params_str}
        {return_str}
        """
        
        # Generate the function code
        return self.generate_code(
            specification=specification,
            language=language,
            code_type="function",
            save_to_file=save_to_file,
            filename=f"{function_name}.{self._get_extension(language)}"
        )
    
    def implement_class(
        self, 
        class_name: str, 
        description: str, 
        language: str, 
        methods: Optional[List[Dict[str, Any]]] = None,
        properties: Optional[List[Dict[str, Any]]] = None,
        save_to_file: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a class based on a description.
        
        Args:
            class_name: Name of the class
            description: Description of what the class should do
            language: Programming language to use
            methods: List of method specifications
            properties: List of property specifications
            save_to_file: Whether to save the generated class to a file
            
        Returns:
            Dictionary containing the generated class and metadata
        """
        # Create methods string
        methods_str = ""
        if methods:
            methods_str = "Methods:\n"
            for method in methods:
                method_name = method.get("name", "")
                method_desc = method.get("description", "")
                method_params = method.get("parameters", [])
                method_return = method.get("return_type", "")
                
                # Format parameters
                params_list = []
                for param in method_params:
                    param_name = param.get("name", "")
                    param_type = param.get("type", "")
                    if param_type:
                        params_list.append(f"{param_name}: {param_type}")
                    else:
                        params_list.append(param_name)
                
                params_str = ", ".join(params_list)
                
                # Add method to string
                methods_str += f"  - {method_name}({params_str})"
                if method_return:
                    methods_str += f" -> {method_return}"
                methods_str += f": {method_desc}\n"
        
        # Create properties string
        props_str = ""
        if properties:
            props_str = "Properties:\n"
            for prop in properties:
                prop_name = prop.get("name", "")
                prop_type = prop.get("type", "")
                prop_desc = prop.get("description", "")
                
                props_str += f"  - {prop_name}: {prop_type} - {prop_desc}\n"
        
        # Create class specification
        specification = f"""
        Class Name: {class_name}
        Description: {description}
        {props_str}
        {methods_str}
        """
        
        # Generate the class code
        return self.generate_code(
            specification=specification,
            language=language,
            code_type="class",
            save_to_file=save_to_file,
            filename=f"{class_name}.{self._get_extension(language)}"
        )
    
    def _get_extension(self, language: str) -> str:
        """
        Get the file extension for a language.
        
        Args:
            language: The programming language
            
        Returns:
            File extension for the language
        """
        extensions = {
            "python": "py",
            "javascript": "js",
            "java": "java",
            "csharp": "cs",
            "apex": "cls",
            "html": "html",
            "css": "css",
            "sql": "sql"
        }
        
        return extensions.get(language.lower(), "txt")
