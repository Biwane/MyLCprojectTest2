"""
Planner Agent Module

This module implements the PlannerAgent class, which specializes in strategic 
planning, task decomposition, and creating structured execution plans for the team.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union

from agents.base_agent import BaseAgent
from core.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)

class PlannerAgent(BaseAgent):
    """
    Agent specialized in strategic planning and task decomposition.
    
    This agent analyzes complex tasks, breaks them down into manageable subtasks,
    identifies dependencies, and creates structured plans for execution by the team.
    """
    
    def __init__(
        self, 
        agent_executor,
        role: str = "planner",
        config: Dict[str, Any] = None,
        knowledge_repository: Optional[KnowledgeRepository] = None
    ):
        """
        Initialize the planner agent.
        
        Args:
            agent_executor: The LangChain agent executor
            role: The specific role of this planner agent
            config: Configuration dictionary with agent settings
            knowledge_repository: Knowledge repository for accessing shared information
        """
        config = config or {}
        super().__init__(agent_executor, role, config, knowledge_repository)
        
        # Planner-specific configuration
        self.planning_depth = config.get("planning_depth", "medium")
        self.include_contingencies = config.get("include_contingencies", True)
        self.max_subtasks = config.get("max_subtasks", 10)
        
        logger.debug(f"Initialized PlannerAgent with role: {role}, planning depth: {self.planning_depth}")
    
    def get_capabilities(self) -> List[str]:
        """
        Get the list of capabilities this agent has.
        
        Returns:
            List of capability descriptions
        """
        return [
            "Task decomposition and breakdown",
            "Dependency identification between subtasks",
            "Resource allocation planning",
            "Timeline and milestone creation",
            "Risk assessment and contingency planning",
            "Critical path analysis"
        ]
    
    def create_plan(self, task_description: str, team_composition: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a comprehensive execution plan for a given task.
        
        Args:
            task_description: Description of the task to plan
            team_composition: Optional information about the available team members
            
        Returns:
            Dictionary containing the structured plan
        """
        # Construct a detailed planning prompt
        planning_prompt = self._create_planning_prompt(task_description, team_composition)
        
        # Execute the planning task
        result = self.execute_task(planning_prompt)
        
        # Extract and structure the plan
        structured_plan = self._structure_plan(result, task_description)
        
        return structured_plan
    
    def _create_planning_prompt(self, task_description: str, team_composition: Dict[str, Any] = None) -> str:
        """
        Create a detailed planning prompt for the given task.
        
        Args:
            task_description: Description of the task to plan
            team_composition: Optional information about the available team members
            
        Returns:
            Formatted planning prompt
        """
        # Adjust depth instructions based on configuration
        depth_instructions = {
            "light": "Create a high-level plan with major phases and key deliverables.",
            "medium": "Create a balanced plan with main phases broken down into specific tasks, key dependencies, and estimated complexity.",
            "detailed": "Create a comprehensive plan with detailed task breakdowns, specific assignments, clear dependencies, contingencies, and precise complexity estimates."
        }
        
        depth_instruction = depth_instructions.get(self.planning_depth, depth_instructions["medium"])
        
        # Start building the prompt
        prompt_parts = [
            f"Task Description: {task_description}",
            "",
            f"{depth_instruction}",
            "",
            "Please structure your plan as follows:",
            "1. Project Overview: Brief summary of the task and approach",
            "2. Goals & Deliverables: Clear list of what will be produced",
            "3. Task Breakdown: Detailed breakdown of work items"
        ]
        
        # Add team-specific instructions if team composition is provided
        if team_composition:
            prompt_parts.append("4. Team Assignments: Mapping of tasks to team members")
            prompt_parts.append("5. Dependencies: Relationships and dependencies between tasks")
            
            # Add information about available team members
            prompt_parts.append("\nAvailable Team Members:")
            for member_id, member_info in team_composition.items():
                member_role = member_info.get("role", "Unknown role")
                prompt_parts.append(f"- {member_id}: {member_role}")
        else:
            prompt_parts.append("4. Dependencies: Relationships and dependencies between tasks")
        
        # Add timeline and risk assessment
        prompt_parts.append("5. Timeline & Milestones: Key checkpoints and estimated durations")
        
        if self.include_contingencies:
            prompt_parts.append("6. Risk Assessment: Potential issues and contingency plans")
        
        # Add formatting instructions
        prompt_parts.append("\nFor the Task Breakdown section, format each task as a JSON object with:")
        prompt_parts.append("- id: A unique identifier for the task")
        prompt_parts.append("- description: Clear description of what needs to be done")
        prompt_parts.append("- estimated_complexity: Low, Medium, or High")
        prompt_parts.append("- dependencies: List of task IDs that must be completed first")
        
        if team_composition:
            prompt_parts.append("- assigned_to: ID of the team member best suited for this task")
        
        prompt_parts.append("\nReturn the Task Breakdown as a valid JSON array.")
        
        return "\n".join(prompt_parts)
    
    def _structure_plan(self, result: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """
        Structure the raw planning result into a consistent format.
        
        Args:
            result: Raw execution result
            task_description: Original task description
            
        Returns:
            Structured execution plan
        """
        output = result.get("output", "")
        
        # Initialize the structured plan
        structured_plan = {
            "task_description": task_description,
            "project_overview": "",
            "goals_deliverables": [],
            "tasks": [],
            "dependencies": [],
            "timeline_milestones": [],
            "risk_assessment": []
        }
        
        # Extract project overview
        if "Project Overview:" in output:
            parts = output.split("Project Overview:", 1)
            if len(parts) > 1:
                overview_text = parts[1].split("\n\n", 1)[0].strip()
                structured_plan["project_overview"] = overview_text
        
        # Extract goals and deliverables
        if "Goals & Deliverables:" in output:
            parts = output.split("Goals & Deliverables:", 1)
            if len(parts) > 1:
                deliverables_text = parts[1].split("\n\n", 1)[0].strip()
                # Split by lines and clean up
                deliverables = [d.strip() for d in deliverables_text.split("\n") if d.strip()]
                # Remove bullet points or numbering
                deliverables = [d[2:].strip() if d.startswith('- ') else 
                               d[d.find('.')+1:].strip() if d[0].isdigit() and '.' in d[:3] else 
                               d for d in deliverables]
                structured_plan["goals_deliverables"] = deliverables
        
        # Try to extract JSON task breakdown
        try:
            # Look for JSON array in the text
            import re
            json_match = re.search(r'\[\s*\{.*\}\s*\]', output, re.DOTALL)
            if json_match:
                json_content = json_match.group(0)
                tasks = json.loads(json_content)
                structured_plan["tasks"] = tasks
            else:
                # Fallback: Extract task breakdown manually
                if "Task Breakdown:" in output:
                    parts = output.split("Task Breakdown:", 1)
                    if len(parts) > 1:
                        tasks_text = parts[1].split("\n\n", 1)[0].strip()
                        # Parse tasks manually (simplified)
                        tasks = self._parse_tasks_manually(tasks_text)
                        structured_plan["tasks"] = tasks
        except Exception as e:
            logger.error(f"Error extracting tasks from plan: {str(e)}")
            # Empty list already set as default
        
        # Extract dependencies if not in tasks
        if not structured_plan["tasks"] and "Dependencies:" in output:
            parts = output.split("Dependencies:", 1)
            if len(parts) > 1:
                dependencies_text = parts[1].split("\n\n", 1)[0].strip()
                # Simple parsing of dependencies
                dependencies = [d.strip() for d in dependencies_text.split("\n") if d.strip()]
                structured_plan["dependencies"] = dependencies
        
        # Extract timeline and milestones
        if "Timeline & Milestones:" in output:
            parts = output.split("Timeline & Milestones:", 1)
            if len(parts) > 1:
                timeline_text = parts[1].split("\n\n", 1)[0].strip()
                # Simple parsing of timeline
                timeline = [t.strip() for t in timeline_text.split("\n") if t.strip()]
                structured_plan["timeline_milestones"] = timeline
        
        # Extract risk assessment if included
        if "Risk Assessment:" in output:
            parts = output.split("Risk Assessment:", 1)
            if len(parts) > 1:
                risk_text = parts[1].strip()
                # Simple parsing of risks
                risks = [r.strip() for r in risk_text.split("\n") if r.strip()]
                structured_plan["risk_assessment"] = risks
        
        # Add raw output for reference
        structured_plan["raw_output"] = output
        
        return structured_plan
    
    def _parse_tasks_manually(self, tasks_text: str) -> List[Dict[str, Any]]:
        """
        Manually parse tasks from text when JSON parsing fails.
        
        Args:
            tasks_text: Text containing task descriptions
            
        Returns:
            List of parsed task dictionaries
        """
        tasks = []
        current_task = {}
        task_lines = tasks_text.split('\n')
        
        for line in task_lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a new task (starts with number or has ID:)
            if line[0].isdigit() and '.' in line[:3] or line.lower().startswith('task'):
                # Save previous task if it exists
                if current_task:
                    tasks.append(current_task)
                    current_task = {}
                
                # Extract task name/description
                task_desc = line[line.find('.')+1:].strip() if '.' in line[:3] else line
                current_task = {"description": task_desc, "id": f"task_{len(tasks) + 1}"}
            
            # Extract task properties
            elif ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                # Handle special cases
                if key == 'dependencies' or key == 'assigned_to':
                    # Convert comma-separated list to array
                    value = [v.strip() for v in value.split(',') if v.strip()]
                
                current_task[key] = value
        
        # Add the last task if it exists
        if current_task:
            tasks.append(current_task)
        
        return tasks
    
    def analyze_dependencies(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze task dependencies to identify critical path and potential bottlenecks.
        
        Args:
            tasks: List of tasks with dependencies
            
        Returns:
            Dictionary with dependency analysis
        """
        # Create dependency graph
        dependency_graph = {}
        for task in tasks:
            task_id = task.get("id", "")
            if not task_id:
                continue
                
            dependencies = task.get("dependencies", [])
            dependency_graph[task_id] = dependencies
        
        # Identify tasks with no dependencies (entry points)
        entry_tasks = [task.get("id") for task in tasks if not task.get("dependencies")]
        
        # Identify tasks that no other tasks depend on (exit points)
        all_dependencies = [dep for deps in dependency_graph.values() for dep in deps]
        exit_tasks = [task_id for task_id in dependency_graph if task_id not in all_dependencies]
        
        # Simple critical path determination (placeholder for actual algorithm)
        # A real implementation would use a proper CPM algorithm
        critical_path = self._simple_critical_path(tasks, dependency_graph)
        
        return {
            "entry_points": entry_tasks,
            "exit_points": exit_tasks,
            "critical_path": critical_path,
            "dependency_graph": dependency_graph
        }
    
    def _simple_critical_path(self, tasks: List[Dict[str, Any]], dependency_graph: Dict[str, List[str]]) -> List[str]:
        """
        Simple approximation of critical path (not a true CPM algorithm).
        
        Args:
            tasks: List of tasks
            dependency_graph: Graph of task dependencies
            
        Returns:
            List of task IDs approximating the critical path
        """
        # This is a simplified placeholder - a real implementation would use proper CPM
        # with forward and backward passes to calculate float and identify critical path
        
        # For this demo, just find a path from an entry to an exit with highest complexities
        task_complexity = {}
        for task in tasks:
            task_id = task.get("id", "")
            complexity = task.get("estimated_complexity", "medium").lower()
            
            # Convert complexity to numeric value
            complexity_value = {"low": 1, "medium": 2, "high": 3}.get(complexity, 2)
            task_complexity[task_id] = complexity_value
        
        # Find entry tasks (tasks with no dependencies)
        entry_tasks = [task.get("id") for task in tasks if not task.get("dependencies")]
        
        # Find exit tasks (tasks that no other tasks depend on)
        all_dependencies = [dep for deps in dependency_graph.values() for dep in deps]
        exit_tasks = [task_id for task_id in dependency_graph if task_id not in all_dependencies]
        
        # Simplified path finding - just a placeholder
        if not entry_tasks or not exit_tasks:
            return []
            
        # Just return a simple chain for demonstration purposes
        current = entry_tasks[0]
        path = [current]
        
        while current not in exit_tasks:
            # Find tasks that depend on current
            next_tasks = []
            for task_id, deps in dependency_graph.items():
                if current in deps:
                    next_tasks.append(task_id)
            
            if not next_tasks:
                break
                
            # Choose the task with highest complexity
            next_task = max(next_tasks, key=lambda t: task_complexity.get(t, 0))
            path.append(next_task)
            current = next_task
        
        return path
    
    def create_gantt_chart(self, tasks: List[Dict[str, Any]]) -> str:
        """
        Create a text-based Gantt chart representation of the plan.
        
        Args:
            tasks: List of tasks with dependencies
            
        Returns:
            Text representation of a Gantt chart
        """
        # Simple text-based Gantt chart
        chart = ["Gantt Chart:\n"]
        
        # Sort tasks based on dependencies (simple topological sort)
        sorted_tasks = self._topological_sort(tasks)
        
        # Create a simple timeline representation
        timeline = {}
        current_time = 0
        
        for task in sorted_tasks:
            task_id = task.get("id", "")
            description = task.get("description", "").split('\n')[0][:30]  # Truncate for display
            dependencies = task.get("dependencies", [])
            
            # Determine start time based on dependencies
            start_time = 0
            for dep in dependencies:
                if dep in timeline and timeline[dep]["end"] > start_time:
                    start_time = timeline[dep]["end"]
            
            # Calculate duration based on complexity
            complexity = task.get("estimated_complexity", "medium").lower()
            duration = {"low": 1, "medium": 2, "high": 3}.get(complexity, 2)
            
            # Record in timeline
            timeline[task_id] = {
                "start": start_time,
                "end": start_time + duration,
                "description": description,
                "duration": duration
            }
        
        # Find the max time
        max_time = max([t["end"] for t in timeline.values()]) if timeline else 0
        
        # Create header
        chart.append("Task" + " " * 26 + "|" + "".join([str(i % 10) for i in range(max_time + 1)]))
        chart.append("-" * 30 + "+" + "-" * (max_time + 1))
        
        # Add tasks to chart
        for task_id, task_info in timeline.items():
            # Create the task line
            task_name = f"{task_id}: {task_info['description']}"
            if len(task_name) > 29:
                task_name = task_name[:26] + "..."
            
            line = task_name + " " * (30 - len(task_name)) + "|"
            
            # Add the timeline
            for i in range(max_time + 1):
                if task_info["start"] <= i < task_info["end"]:
                    line += "#"
                else:
                    line += " "
            
            chart.append(line)
        
        return "\n".join(chart)
    
    def _topological_sort(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort tasks based on dependencies (topological sort).
        
        Args:
            tasks: List of tasks with dependencies
            
        Returns:
            Sorted list of tasks
        """
        # Create a dictionary of task_id to task
        task_dict = {task.get("id", f"task_{i}"): task for i, task in enumerate(tasks)}
        
        # Create adjacency list
        graph = {}
        for task in tasks:
            task_id = task.get("id", "")
            if not task_id:
                continue
            
            graph[task_id] = task.get("dependencies", [])
        
        # Perform topological sort
        visited = set()
        temp_mark = set()
        result = []
        
        def visit(node):
            if node in temp_mark:
                # Circular dependency, handle gracefully
                return
            if node not in visited:
                temp_mark.add(node)
                for dep in graph.get(node, []):
                    if dep in task_dict:  # Make sure the dependency exists
                        visit(dep)
                temp_mark.remove(node)
                visited.add(node)
                result.append(task_dict[node])
        
        # Visit all nodes
        for node in graph:
            if node not in visited:
                visit(node)
        
        # Reverse to get correct order
        return result[::-1]
    
    def get_role_description(self) -> str:
        """
        Get a description of this agent's role.
        
        Returns:
            Description of the agent's role
        """
        return (
            f"I am a {self.role} agent specializing in strategic planning and task management. "
            f"I can analyze complex tasks, break them down into manageable components, "
            f"identify dependencies, and create structured execution plans. "
            f"I can also perform critical path analysis and help with resource allocation."
        )