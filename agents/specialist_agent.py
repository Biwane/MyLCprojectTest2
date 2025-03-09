"""
Specialist Agent Module

This module implements the SpecialistAgent class, which provides domain-specific
expertise in various fields. It can be configured for different specializations
such as development, sales, marketing, etc.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union

from agents.base_agent import BaseAgent
from core.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)

class SpecialistAgent(BaseAgent):
    """
    Agent specialized in providing domain-specific expertise.
    
    This agent can be configured for different specializations such as
    software development, data science, marketing, sales, etc.
    """
    
    def __init__(
        self, 
        agent_executor,
        role: str = "specialist",
        config: Dict[str, Any] = None,
        knowledge_repository: Optional[KnowledgeRepository] = None
    ):
        """
        Initialize the specialist agent.
        
        Args:
            agent_executor: The LangChain agent executor
            role: The specific role of this specialist agent
            config: Configuration dictionary with agent settings
            knowledge_repository: Knowledge repository for accessing shared information
        """
        config = config or {}
        super().__init__(agent_executor, role, config, knowledge_repository)
        
        # Extract specialization from role
        self.specialization = self._extract_specialization(role)
        
        # Specialist-specific configuration
        self.domain_knowledge = config.get("domain_knowledge", {})
        self.best_practices = config.get("best_practices", [])
        self.reference_materials = config.get("reference_materials", [])
        
        logger.debug(f"Initialized SpecialistAgent with specialization: {self.specialization}")
    
    def _extract_specialization(self, role: str) -> str:
        """
        Extract specialization from the role string.
        
        Args:
            role: The role string (e.g., "specialist_salesforce_developer")
            
        Returns:
            Extracted specialization
        """
        parts = role.split('_', 1)
        if len(parts) > 1:
            return parts[1]
        return "general"  # Default if no specialization specified
    
    def get_capabilities(self) -> List[str]:
        """
        Get the list of capabilities this agent has.
        
        Returns:
            List of capability descriptions
        """
        # Base capabilities
        capabilities = [
            "Domain-specific expertise and knowledge",
            "Application of best practices in specialty area",
            "Analysis of domain-specific problems",
            "Generation of specialized solutions",
            "Technical implementation in specialty area"
        ]
        
        # Add specialization-specific capabilities
        if self.specialization == "salesforce_developer":
            capabilities.extend([
                "Apex code development and optimization",
                "Lightning component design and implementation",
                "Salesforce integration patterns",
                "SOQL and SOSL query optimization",
                "Salesforce deployment and CI/CD practices"
            ])
        elif self.specialization == "salesforce_admin":
            capabilities.extend([
                "Salesforce configuration and setup",
                "User management and security settings",
                "Workflow and process automation",
                "Report and dashboard creation",
                "Data management and maintenance"
            ])
        elif self.specialization == "web_developer":
            capabilities.extend([
                "Frontend development (HTML, CSS, JavaScript)",
                "Backend system implementation",
                "API design and development",
                "Responsive design implementation",
                "Web performance optimization"
            ])
        elif self.specialization == "data_scientist":
            capabilities.extend([
                "Data analysis and interpretation",
                "Statistical modeling and machine learning",
                "Data visualization and reporting",
                "Predictive analytics",
                "Big data processing techniques"
            ])
        
        return capabilities
    
    def provide_expertise(self, problem_description: str) -> Dict[str, Any]:
        """
        Provide domain-specific expertise on a given problem.
        
        Args:
            problem_description: Description of the problem or question
            
        Returns:
            Dictionary with expert analysis and recommendations
        """
        prompt = self._create_expertise_prompt(problem_description)
        
        # Execute the task
        result = self.execute_task(prompt)
        
        # Process and structure the response
        structured_result = self._structure_expertise_result(result, problem_description)
        
        return structured_result
    
    def _create_expertise_prompt(self, problem_description: str) -> str:
        """
        Create an expertise request prompt.
        
        Args:
            problem_description: Description of the problem
            
        Returns:
            Formatted expertise prompt
        """
        # Add specialization-specific context
        specialization_context = self._get_specialization_context()
        
        # Create the prompt
        prompt = f"""
        As a specialist in {self.specialization}, please provide your expert analysis and recommendations for the following:
        
        Problem/Question: {problem_description}
        
        {specialization_context}
        
        Please structure your response as follows:
        1. Analysis: Your assessment of the problem/question
        2. Key Considerations: Important factors or constraints to consider
        3. Recommendations: Your suggested approach or solution
        4. Best Practices: Relevant best practices to apply
        5. Implementation Notes: Guidance on implementing your recommendations
        
        Be specific, practical, and thorough in your expertise.
        """
        
        return prompt
    
    def _get_specialization_context(self) -> str:
        """
        Get context information specific to this agent's specialization.
        
        Returns:
            Context information as a string
        """
        # Specialization-specific contexts
        contexts = {
            "salesforce_developer": """
            When analyzing, consider:
            - Salesforce governor limits and their impact
            - Security and sharing model implications
            - Maintainability and upgradability of code
            - Integration with existing systems
            - Testing and deployment considerations
            
            Reference latest Salesforce development standards and patterns.
            """,
            
            "salesforce_admin": """
            When analyzing, consider:
            - Declarative vs programmatic solutions
            - Security and permission implications
            - User experience and adoption
            - Maintenance and administration overhead
            - Scalability for future growth
            
            Prioritize declarative solutions where appropriate.
            """,
            
            "web_developer": """
            When analyzing, consider:
            - Browser compatibility requirements
            - Responsive design needs
            - Performance optimization
            - Accessibility requirements
            - Security best practices
            - SEO implications
            
            Balance modern techniques with broad compatibility.
            """,
            
            "data_scientist": """
            When analyzing, consider:
            - Data quality and availability
            - Statistical validity of approaches
            - Computational efficiency
            - Interpretability of models
            - Deployment and operationalization
            - Ethical implications
            
            Focus on practical, implementable solutions with clear value.
            """
        }
        
        # Return the context for this specialization, or a default if not found
        return contexts.get(self.specialization, "Please provide detailed, specialized guidance based on your expertise.")
    
    def _structure_expertise_result(
        self, 
        result: Dict[str, Any], 
        problem_description: str
    ) -> Dict[str, Any]:
        """
        Structure the expertise result into a consistent format.
        
        Args:
            result: Raw execution result
            problem_description: Original problem description
            
        Returns:
            Structured expertise result
        """
        output = result.get("output", "")
        
        # Attempt to parse structured sections from the output
        sections = {
            "analysis": "",
            "key_considerations": [],
            "recommendations": [],
            "best_practices": [],
            "implementation_notes": ""
        }
        
        # Extract sections using simple heuristics
        if "Analysis:" in output or "ANALYSIS:" in output:
            parts = output.split("Analysis:", 1) if "Analysis:" in output else output.split("ANALYSIS:", 1)
            if len(parts) > 1:
                analysis_text = parts[1].split("\n\n", 1)[0].strip()
                sections["analysis"] = analysis_text
        
        if "Key Considerations:" in output or "KEY CONSIDERATIONS:" in output:
            parts = output.split("Key Considerations:", 1) if "Key Considerations:" in output else output.split("KEY CONSIDERATIONS:", 1)
            if len(parts) > 1:
                considerations_text = parts[1].split("\n\n", 1)[0].strip()
                # Split into bullet points or numbered items
                considerations = [c.strip() for c in considerations_text.split("\n") if c.strip()]
                sections["key_considerations"] = considerations
        
        if "Recommendations:" in output or "RECOMMENDATIONS:" in output:
            parts = output.split("Recommendations:", 1) if "Recommendations:" in output else output.split("RECOMMENDATIONS:", 1)
            if len(parts) > 1:
                recommendations_text = parts[1].split("\n\n", 1)[0].strip()
                # Split into bullet points or numbered items
                recommendations = [r.strip() for r in recommendations_text.split("\n") if r.strip()]
                sections["recommendations"] = recommendations
        
        if "Best Practices:" in output or "BEST PRACTICES:" in output:
            parts = output.split("Best Practices:", 1) if "Best Practices:" in output else output.split("BEST PRACTICES:", 1)
            if len(parts) > 1:
                practices_text = parts[1].split("\n\n", 1)[0].strip()
                # Split into bullet points or numbered items
                practices = [p.strip() for p in practices_text.split("\n") if p.strip()]
                sections["best_practices"] = practices
        
        if "Implementation Notes:" in output or "IMPLEMENTATION NOTES:" in output:
            parts = output.split("Implementation Notes:", 1) if "Implementation Notes:" in output else output.split("IMPLEMENTATION NOTES:", 1)
            if len(parts) > 1:
                notes_text = parts[1].strip()
                sections["implementation_notes"] = notes_text
        
        # If we couldn't parse structured sections, use the entire output as analysis
        if not sections["analysis"] and not any(sections.values()):
            sections["analysis"] = output
        
        # Create the final structured result
        structured_result = {
            "problem_description": problem_description,
            "expertise_data": sections,
            "specialization": self.specialization,
            "raw_output": output
        }
        
        return structured_result
    
    def evaluate_solution(self, solution: str, requirements: str = None) -> Dict[str, Any]:
        """
        Evaluate a proposed solution from a domain-specialist perspective.
        
        Args:
            solution: The proposed solution to evaluate
            requirements: Optional requirements to evaluate against
            
        Returns:
            Dictionary with evaluation results
        """
        # Create evaluation prompt
        prompt = f"""
        As a specialist in {self.specialization}, please evaluate the following solution:
        
        Solution to Evaluate:
        {solution}
        """
        
        # Add requirements if provided
        if requirements:
            prompt += f"""
            
            Requirements to evaluate against:
            {requirements}
            """
        
        prompt += """
        
        Please provide your evaluation structured as follows:
        1. Strengths: What aspects of the solution are well-designed or effective
        2. Weaknesses: Where the solution falls short or could be improved
        3. Alignment with Best Practices: How well the solution follows standards in this field
        4. Risks: Potential issues or challenges with this approach
        5. Recommendations: Specific suggestions for improvement
        6. Overall Assessment: Your general evaluation (excellent, good, adequate, problematic)
        
        Be specific and constructive in your feedback.
        """
        
        # Execute the evaluation
        result = self.execute_task(prompt)
        
        # Extract and structure the evaluation
        output = result.get("output", "")
        
        # Simple structure for evaluation response
        evaluation = {
            "strengths": self._extract_section(output, "Strengths:"),
            "weaknesses": self._extract_section(output, "Weaknesses:"),
            "alignment": self._extract_section(output, "Alignment with Best Practices:"),
            "risks": self._extract_section(output, "Risks:"),
            "recommendations": self._extract_section(output, "Recommendations:"),
            "overall_assessment": self._extract_section(output, "Overall Assessment:"),
            "raw_evaluation": output
        }
        
        return {
            "solution_evaluated": solution[:200] + "..." if len(solution) > 200 else solution,
            "evaluation": evaluation,
            "specialization": self.specialization
        }
    
    def _extract_section(self, text: str, section_header: str) -> str:
        """
        Extract a section from text based on a header.
        
        Args:
            text: The text to extract from
            section_header: The section header to look for
            
        Returns:
            The extracted section text or empty string if not found
        """
        if section_header in text:
            parts = text.split(section_header, 1)
            if len(parts) > 1:
                section_text = parts[1].split("\n\n", 1)[0].strip()
                return section_text
        return ""
    
    def implement_solution(self, task_description: str, specifications: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Implement a solution based on task description and specifications.
        
        Args:
            task_description: Description of the task to implement
            specifications: Optional specifications to follow
            
        Returns:
            Dictionary with implementation results
        """
        # Create implementation prompt
        prompt = f"""
        As a specialist in {self.specialization}, please implement a solution for the following task:
        
        Task: {task_description}
        """
        
        # Add specifications if provided
        if specifications:
            prompt += "\n\nSpecifications:\n"
            for key, value in specifications.items():
                prompt += f"- {key}: {value}\n"
        
        prompt += """
        
        Please provide your implementation with:
        1. A clear description of your approach
        2. The actual implementation (code, configuration, etc.)
        3. Instructions for deployment or use
        4. Any assumptions or limitations
        
        Make your solution as complete and ready-to-use as possible.
        """
        
        # Execute the implementation task
        result = self.execute_task(prompt)
        
        # Structure the result
        return {
            "task": task_description,
            "implementation": result.get("output", ""),
            "specialization": self.specialization,
            "metadata": result.get("metadata", {})
        }
    
    def get_role_description(self) -> str:
        """
        Get a description of this agent's role.
        
        Returns:
            Description of the agent's role
        """
        return (
            f"I am a specialist in {self.specialization} with deep domain expertise. "
            f"I can provide expert analysis, evaluate solutions from my domain perspective, "
            f"and implement specialized solutions following best practices in my field."
        )
