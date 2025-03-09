"""
Reviewer Agent Module

This module implements the ReviewerAgent class, which specializes in evaluating and
reviewing the work of other agents, providing feedback, suggestions for improvement,
and quality assurance.
"""

import logging
from typing import Dict, Any, List, Optional, Union

from agents.base_agent import BaseAgent
from core.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)

class ReviewerAgent(BaseAgent):
    """
    Agent specialized in reviewing and evaluating the work of other agents.
    
    This agent examines solutions, implementations, and other outputs to assess
    quality, identify issues, and suggest improvements, serving as a quality
    assurance mechanism for the team.
    """
    
    def __init__(
        self, 
        agent_executor,
        role: str = "reviewer",
        config: Dict[str, Any] = None,
        knowledge_repository: Optional[KnowledgeRepository] = None
    ):
        """
        Initialize the reviewer agent.
        
        Args:
            agent_executor: The LangChain agent executor
            role: The specific role of this reviewer agent
            config: Configuration dictionary with agent settings
            knowledge_repository: Knowledge repository for accessing shared information
        """
        config = config or {}
        super().__init__(agent_executor, role, config, knowledge_repository)
        
        # Reviewer-specific configuration
        self.review_criteria = config.get("review_criteria", [
            "correctness",
            "completeness",
            "efficiency",
            "maintainability"
        ])
        
        logger.debug(f"Initialized ReviewerAgent with role: {role}")
    
    def get_capabilities(self) -> List[str]:
        """
        Get the list of capabilities this agent has.
        
        Returns:
            List of capability descriptions
        """
        return [
            "Quality assessment of solutions and implementations",
            "Identification of errors, bugs, and issues",
            "Evaluation against requirements and specifications",
            "Suggestions for improvements and optimizations",
            "Code review and analysis",
            "Documentation review",
            "Compliance checking against standards and best practices"
        ]
    
    def review_solution(
        self, 
        solution: Dict[str, Any], 
        requirements: Dict[str, Any], 
        review_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Review a solution against requirements.
        
        Args:
            solution: The solution to review
            requirements: The requirements to evaluate against
            review_type: Type of review to perform
            
        Returns:
            Dictionary containing the review results
        """
        # Create the review prompt
        review_prompt = self._create_review_prompt(solution, requirements, review_type)
        
        # Execute the review
        result = self.execute_task(review_prompt)
        
        # Process and structure the review results
        structured_review = self._structure_review_results(result, review_type)
        
        # Add metadata
        structured_review["review_type"] = review_type
        structured_review["solution_type"] = solution.get("type", "unknown")
        
        return structured_review
    
    def _create_review_prompt(
        self, 
        solution: Dict[str, Any], 
        requirements: Dict[str, Any], 
        review_type: str
    ) -> str:
        """
        Create a review prompt for the given solution and requirements.
        
        Args:
            solution: The solution to review
            requirements: The requirements to evaluate against
            review_type: Type of review to perform
            
        Returns:
            Formatted review prompt
        """
        # Extract key information
        solution_type = solution.get("type", "general")
        solution_content = solution.get("content", "")
        solution_description = solution.get("description", "")
        
        # If content is a dictionary, format it as a string
        if isinstance(solution_content, dict):
            solution_content = json.dumps(solution_content, indent=2)
        elif isinstance(solution_content, list):
            solution_content = "\n".join([str(item) for item in solution_content])
        
        # Extract requirements
        req_description = requirements.get("description", "")
        req_criteria = requirements.get("criteria", [])
        req_constraints = requirements.get("constraints", [])
        
        # Build the prompt
        prompt_parts = [
            f"Review Type: {review_type}",
            "",
            "Solution Description:",
            solution_description,
            "",
            "Solution to Review:",
            solution_content,
            "",
            "Requirements and Criteria:",
            req_description
        ]
        
        # Add specific requirements criteria
        if req_criteria:
            prompt_parts.append("\nRequirements Criteria:")
            for i, criterion in enumerate(req_criteria, 1):
                prompt_parts.append(f"{i}. {criterion}")
        
        # Add constraints
        if req_constraints:
            prompt_parts.append("\nConstraints:")
            for i, constraint in enumerate(req_constraints, 1):
                prompt_parts.append(f"{i}. {constraint}")
        
        # Add review-type specific instructions
        if review_type == "code_review":
            prompt_parts.append("\nCode Review Instructions:")
            prompt_parts.append(
                "Please perform a thorough code review focusing on correctness, "
                "efficiency, security, maintainability, and adherence to best practices. "
                "Identify any bugs, vulnerabilities, or potential issues."
            )
        elif review_type == "design_review":
            prompt_parts.append("\nDesign Review Instructions:")
            prompt_parts.append(
                "Please evaluate the design for completeness, coherence, scalability, "
                "and alignment with requirements. Consider architectural soundness, "
                "component relationships, and overall effectiveness."
            )
        elif review_type == "documentation_review":
            prompt_parts.append("\nDocumentation Review Instructions:")
            prompt_parts.append(
                "Please review the documentation for clarity, completeness, accuracy, "
                "organization, and usefulness. Ensure it effectively communicates the "
                "necessary information to its intended audience."
            )
        else:
            prompt_parts.append("\nReview Instructions:")
            prompt_parts.append(
                "Please conduct a comprehensive review evaluating how well the solution "
                "meets the requirements and criteria. Identify strengths, weaknesses, "
                "and areas for improvement."
            )
        
        # Add review structure guidelines
        prompt_parts.append("\nPlease structure your review as follows:")
        prompt_parts.append("1. Overall Assessment: A brief summary of your evaluation")
        prompt_parts.append("2. Strengths: What aspects of the solution are well done")
        prompt_parts.append("3. Issues: Problems, bugs, or concerns that need to be addressed")
        prompt_parts.append("4. Improvement Suggestions: Specific recommendations for enhancement")
        prompt_parts.append("5. Compliance: How well the solution meets the requirements")
        
        if review_type == "code_review":
            prompt_parts.append("6. Code Quality: Assessment of the code's quality and maintainability")
            prompt_parts.append("7. Security Analysis: Identification of any security concerns")
        
        prompt_parts.append("\nPlease be specific, constructive, and actionable in your feedback.")
        
        return "\n".join(prompt_parts)
    
    def _structure_review_results(self, result: Dict[str, Any], review_type: str) -> Dict[str, Any]:
        """
        Structure the raw review results into a consistent format.
        
        Args:
            result: Raw execution result
            review_type: Type of review performed
            
        Returns:
            Structured review results
        """
        output = result.get("output", "")
        
        # Initialize structured review
        structured_review = {
            "overall_assessment": "",
            "strengths": [],
            "issues": [],
            "improvement_suggestions": [],
            "compliance": "",
            "rating": None,
            "raw_review": output
        }
        
        # Add code-specific fields for code reviews
        if review_type == "code_review":
            structured_review["code_quality"] = ""
            structured_review["security_analysis"] = ""
        
        # Extract overall assessment
        if "Overall Assessment:" in output:
            parts = output.split("Overall Assessment:", 1)
            if len(parts) > 1:
                assessment_text = parts[1].split("\n\n", 1)[0]
                structured_review["overall_assessment"] = assessment_text.strip()
        
        # Extract strengths
        if "Strengths:" in output:
            parts = output.split("Strengths:", 1)
            if len(parts) > 1:
                strengths_text = parts[1].split("\n\n", 1)[0]
                strengths = [s.strip() for s in strengths_text.split("\n") if s.strip()]
                # Clean up bullet points
                strengths = [s[2:].strip() if s.startswith('- ') else 
                            s[s.find('.')+1:].strip() if s[0].isdigit() and '.' in s[:3] else 
                            s for s in strengths]
                structured_review["strengths"] = strengths
        
        # Extract issues
        if "Issues:" in output:
            parts = output.split("Issues:", 1)
            if len(parts) > 1:
                issues_text = parts[1].split("\n\n", 1)[0]
                issues = [i.strip() for i in issues_text.split("\n") if i.strip()]
                # Clean up bullet points
                issues = [i[2:].strip() if i.startswith('- ') else 
                         i[i.find('.')+1:].strip() if i[0].isdigit() and '.' in i[:3] else 
                         i for i in issues]
                structured_review["issues"] = issues
        
        # Extract improvement suggestions
        if "Improvement Suggestions:" in output:
            parts = output.split("Improvement Suggestions:", 1)
            if len(parts) > 1:
                suggestions_text = parts[1].split("\n\n", 1)[0]
                suggestions = [s.strip() for s in suggestions_text.split("\n") if s.strip()]
                # Clean up bullet points
                suggestions = [s[2:].strip() if s.startswith('- ') else 
                              s[s.find('.')+1:].strip() if s[0].isdigit() and '.' in s[:3] else 
                              s for s in suggestions]
                structured_review["improvement_suggestions"] = suggestions
        
        # Extract compliance
        if "Compliance:" in output:
            parts = output.split("Compliance:", 1)
            if len(parts) > 1:
                compliance_text = parts[1].split("\n\n", 1)[0]
                structured_review["compliance"] = compliance_text.strip()
        
        # Extract code quality for code reviews
        if review_type == "code_review" and "Code Quality:" in output:
            parts = output.split("Code Quality:", 1)
            if len(parts) > 1:
                quality_text = parts[1].split("\n\n", 1)[0]
                structured_review["code_quality"] = quality_text.strip()
        
        # Extract security analysis for code reviews
        if review_type == "code_review" and "Security Analysis:" in output:
            parts = output.split("Security Analysis:", 1)
            if len(parts) > 1:
                security_text = parts[1].split("\n\n", 1)[0]
                structured_review["security_analysis"] = security_text.strip()
        
        # Determine a numeric rating based on the review
        structured_review["rating"] = self._calculate_rating(structured_review)
        
        return structured_review
    
    def _calculate_rating(self, structured_review: Dict[str, Any]) -> float:
        """
        Calculate a numeric rating based on the structured review.
        
        Args:
            structured_review: The structured review data
            
        Returns:
            Numeric rating between 0 and 10
        """
        # This is a simplified rating algorithm
        # A real implementation would be more sophisticated
        
        # Start with a neutral score
        rating = 5.0
        
        # Analyze overall assessment tone
        assessment = structured_review.get("overall_assessment", "").lower()
        if any(word in assessment for word in ["excellent", "outstanding", "exceptional"]):
            rating += 2.0
        elif any(word in assessment for word in ["good", "solid", "strong"]):
            rating += 1.0
        elif any(word in assessment for word in ["poor", "inadequate", "fails"]):
            rating -= 2.0
        elif any(word in assessment for word in ["issue", "concern", "problem"]):
            rating -= 1.0
        
        # Adjust based on strengths and issues
        strengths_count = len(structured_review.get("strengths", []))
        issues_count = len(structured_review.get("issues", []))
        
        # More strengths than issues is good
        if strengths_count > issues_count:
            rating += min(2.0, (strengths_count - issues_count) * 0.5)
        # More issues than strengths is bad
        elif issues_count > strengths_count:
            rating -= min(2.0, (issues_count - strengths_count) * 0.5)
        
        # Check for critical issues
        critical_issues = 0
        for issue in structured_review.get("issues", []):
            if any(word in issue.lower() for word in ["critical", "severe", "major", "serious"]):
                critical_issues += 1
        
        # Deduct for critical issues
        rating -= min(3.0, critical_issues * 1.0)
        
        # Analyze compliance
        compliance = structured_review.get("compliance", "").lower()
        if "fully" in compliance and "meet" in compliance:
            rating += 1.0
        elif "partially" in compliance:
            rating -= 0.5
        elif "not" in compliance and "meet" in compliance:
            rating -= 1.0
        
        # Ensure rating is within bounds
        rating = max(0.0, min(10.0, rating))
        
        return round(rating, 1)
    
    def synthesize_reviews(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize multiple reviews into a consolidated review.
        
        Args:
            reviews: List of individual reviews
            
        Returns:
            Consolidated review
        """
        if not reviews:
            return {
                "overall_assessment": "No reviews provided",
                "strengths": [],
                "issues": [],
                "improvement_suggestions": [],
                "compliance": "N/A",
                "rating": None
            }
        
        # Create a prompt to synthesize the reviews
        synthesis_prompt = self._create_synthesis_prompt(reviews)
        
        # Execute the synthesis
        result = self.execute_task(synthesis_prompt)
        
        # Process and structure the synthesis results
        structured_synthesis = self._structure_review_results(result, "synthesis")
        
        # Calculate an average rating
        ratings = [review.get("rating", 0) for review in reviews if review.get("rating") is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else None
        structured_synthesis["rating"] = avg_rating
        
        # Add metadata
        structured_synthesis["review_count"] = len(reviews)
        structured_synthesis["review_types"] = list(set(review.get("review_type", "unknown") for review in reviews))
        
        return structured_synthesis
    
    def _create_synthesis_prompt(self, reviews: List[Dict[str, Any]]) -> str:
        """
        Create a prompt to synthesize multiple reviews.
        
        Args:
            reviews: List of reviews to synthesize
            
        Returns:
            Synthesis prompt
        """
        prompt_parts = [
            "Task: Synthesize the following reviews into a consolidated review.",
            "",
            f"Number of reviews to synthesize: {len(reviews)}",
            "",
            "Reviews:"
        ]
        
        # Add each review
        for i, review in enumerate(reviews, 1):
            prompt_parts.append(f"\nReview {i} ({review.get('review_type', 'unknown')}):")
            prompt_parts.append(f"Overall Assessment: {review.get('overall_assessment', 'N/A')}")
            
            # Add strengths
            prompt_parts.append("Strengths:")
            for strength in review.get("strengths", []):
                prompt_parts.append(f"- {strength}")
            
            # Add issues
            prompt_parts.append("Issues:")
            for issue in review.get("issues", []):
                prompt_parts.append(f"- {issue}")
            
            # Add improvement suggestions
            prompt_parts.append("Improvement Suggestions:")
            for suggestion in review.get("improvement_suggestions", []):
                prompt_parts.append(f"- {suggestion}")
            
            prompt_parts.append(f"Compliance: {review.get('compliance', 'N/A')}")
            prompt_parts.append(f"Rating: {review.get('rating', 'N/A')}")
        
        # Add synthesis instructions
        prompt_parts.append("\nPlease synthesize these reviews into a consolidated review that:")
        prompt_parts.append("1. Provides a balanced overall assessment")
        prompt_parts.append("2. Identifies common strengths across reviews")
        prompt_parts.append("3. Highlights important issues that need addressing")
        prompt_parts.append("4. Consolidates improvement suggestions")
        prompt_parts.append("5. Provides an overall compliance assessment")
        
        prompt_parts.append("\nStructure your synthesis as follows:")
        prompt_parts.append("1. Overall Assessment: A comprehensive summary")
        prompt_parts.append("2. Strengths: Common and significant strengths")
        prompt_parts.append("3. Issues: Important problems that need addressing")
        prompt_parts.append("4. Improvement Suggestions: Consolidated recommendations")
        prompt_parts.append("5. Compliance: Overall assessment of requirement compliance")
        
        return "\n".join(prompt_parts)
    
    def review_code(self, code: str, language: str, requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform a specialized code review.
        
        Args:
            code: The code to review
            language: The programming language of the code
            requirements: Optional requirements to evaluate against
            
        Returns:
            Code review results
        """
        # Create a solution object
        solution = {
            "type": "code",
            "content": code,
            "description": f"{language} code review"
        }
        
        # Default requirements if none provided
        if not requirements:
            requirements = {
                "description": f"Review {language} code for quality, correctness, and best practices",
                "criteria": [
                    "Correctness: The code should function as intended",
                    "Readability: The code should be easy to read and understand",
                    "Maintainability: The code should be easy to maintain and extend",
                    "Efficiency: The code should be efficient and performant",
                    "Security: The code should be secure and free of vulnerabilities"
                ],
                "constraints": []
            }
            
            # Add language-specific criteria
            if language.lower() == "python":
                requirements["criteria"].append("Follows PEP 8 style guidelines")
                requirements["criteria"].append("Uses Python idioms and best practices")
            elif language.lower() == "javascript":
                requirements["criteria"].append("Follows modern JavaScript conventions")
                requirements["criteria"].append("Properly handles asynchronous operations")
            elif language.lower() == "java":
                requirements["criteria"].append("Follows Java coding conventions")
                requirements["criteria"].append("Uses appropriate OOP principles")
            elif language.lower() == "apex":
                requirements["criteria"].append("Considers Salesforce governor limits")
                requirements["criteria"].append("Follows Salesforce security best practices")
        
        # Perform the review
        return self.review_solution(solution, requirements, "code_review")
    
    def review_documentation(self, documentation: str, doc_type: str, requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Review documentation for quality and completeness.
        
        Args:
            documentation: The documentation to review
            doc_type: Type of documentation (user_guide, api_reference, etc.)
            requirements: Optional requirements to evaluate against
            
        Returns:
            Documentation review results
        """
        # Create a solution object
        solution = {
            "type": "documentation",
            "content": documentation,
            "description": f"{doc_type} documentation review"
        }
        
        # Default requirements if none provided
        if not requirements:
            requirements = {
                "description": f"Review {doc_type} documentation for quality, clarity, and completeness",
                "criteria": [
                    "Clarity: The documentation should be clear and easy to understand",
                    "Completeness: The documentation should cover all relevant aspects",
                    "Accuracy: The documentation should be accurate and up-to-date",
                    "Organization: The documentation should be well-structured and organized",
                    "Usefulness: The documentation should be helpful to its intended audience"
                ],
                "constraints": []
            }
            
            # Add doc-type specific criteria
            if doc_type.lower() == "user_guide":
                requirements["criteria"].append("Includes clear instructions for common tasks")
                requirements["criteria"].append("Provides helpful examples")
            elif doc_type.lower() == "api_reference":
                requirements["criteria"].append("Documents all public APIs")
                requirements["criteria"].append("Includes parameter and return value descriptions")
            elif doc_type.lower() == "technical_spec":
                requirements["criteria"].append("Provides detailed technical information")
                requirements["criteria"].append("Explains design decisions and trade-offs")
        
        # Perform the review
        return self.review_solution(solution, requirements, "documentation_review")
    
    def get_role_description(self) -> str:
        """
        Get a description of this agent's role.
        
        Returns:
            Description of the agent's role
        """
        return (
            f"I am a {self.role} agent specializing in evaluating and improving the "
            f"quality of work. I can review code, documentation, designs, and other "
            f"deliverables, identifying strengths, issues, and opportunities for "
            f"improvement. My goal is to ensure the highest quality of the team's output."
        )
