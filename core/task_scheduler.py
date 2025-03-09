"""
Task Scheduler Module

This module is responsible for scheduling and organizing tasks for execution by the agent team.
It handles dependency resolution, parallelization opportunities, and creates an optimal
execution plan for completing tasks efficiently.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class TaskScheduler:
    """
    Scheduler for optimizing task execution across a team of agents.
    Handles dependencies between tasks and identifies parallelization opportunities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the task scheduler.
        
        Args:
            config: Configuration dictionary with scheduler settings
        """
        self.config = config
        self.max_parallel_tasks = config.get("max_parallel_tasks", 3)
        self.prioritize_by_complexity = config.get("prioritize_by_complexity", True)
        
        logger.debug(f"Initialized TaskScheduler with max_parallel_tasks: {self.max_parallel_tasks}")
    
    def create_schedule(
        self, 
        subtasks: List[Dict[str, Any]], 
        available_agents: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Create an execution schedule based on subtasks and available agents.
        
        Args:
            subtasks: List of subtask specifications with dependencies
            available_agents: List of available agent IDs
            
        Returns:
            Scheduled execution plan as a list of steps
        """
        logger.info(f"Creating schedule for {len(subtasks)} subtasks with {len(available_agents)} agents")
        
        # Step 1: Validate and normalize subtasks
        normalized_subtasks = self._normalize_subtasks(subtasks)
        
        # Step 2: Build dependency graph
        dependency_graph, reverse_dependency_graph = self._build_dependency_graphs(normalized_subtasks)
        
        # Step 3: Validate for circular dependencies
        if self._has_circular_dependencies(dependency_graph):
            logger.warning("Circular dependencies detected in subtasks, resolving dependencies")
            dependency_graph, reverse_dependency_graph = self._resolve_circular_dependencies(
                dependency_graph, reverse_dependency_graph
            )
        
        # Step 4: Create execution schedule
        schedule = self._create_execution_plan(
            normalized_subtasks, 
            dependency_graph, 
            reverse_dependency_graph,
            available_agents
        )
        
        logger.info(f"Created execution schedule with {len(schedule)} steps")
        return schedule
    
    def _normalize_subtasks(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and normalize subtask specifications.
        
        Args:
            subtasks: List of subtask specifications
            
        Returns:
            Normalized subtask specifications
        """
        normalized = []
        
        # Keep track of existing IDs to avoid duplicates
        existing_ids = set()
        
        for i, subtask in enumerate(subtasks):
            # Create a new subtask dictionary with normalized fields
            normalized_subtask = {}
            
            # Ensure each subtask has an ID
            if "id" not in subtask or not subtask["id"]:
                subtask_id = f"subtask_{i}_{str(uuid.uuid4())[:8]}"
            else:
                subtask_id = subtask["id"]
                
            # Ensure ID is unique
            if subtask_id in existing_ids:
                subtask_id = f"{subtask_id}_{str(uuid.uuid4())[:8]}"
            
            existing_ids.add(subtask_id)
            normalized_subtask["id"] = subtask_id
            
            # Copy description
            normalized_subtask["description"] = subtask.get("description", f"Subtask {i}")
            
            # Normalize assigned agent
            normalized_subtask["assigned_agent"] = subtask.get("assigned_agent", "")
            
            # Normalize dependencies
            dependencies = subtask.get("dependencies", [])
            if isinstance(dependencies, str):
                dependencies = [dependencies]
            normalized_subtask["dependencies"] = dependencies
            
            # Normalize complexity
            complexity_map = {"low": 1, "medium": 2, "high": 3}
            if isinstance(subtask.get("complexity"), str):
                normalized_subtask["complexity"] = complexity_map.get(
                    subtask.get("complexity", "medium").lower(), 2
                )
            else:
                normalized_subtask["complexity"] = subtask.get("complexity", 2)
            
            # Copy any additional fields
            for key, value in subtask.items():
                if key not in normalized_subtask:
                    normalized_subtask[key] = value
            
            normalized.append(normalized_subtask)
        
        return normalized
    
    def _build_dependency_graphs(
        self, 
        subtasks: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Build dependency and reverse dependency graphs.
        
        Args:
            subtasks: List of normalized subtask specifications
            
        Returns:
            Tuple of (dependency_graph, reverse_dependency_graph)
        """
        # Map of subtask IDs
        id_to_subtask = {subtask["id"]: subtask for subtask in subtasks}
        
        # Initialize graphs
        dependency_graph = defaultdict(list)
        reverse_dependency_graph = defaultdict(list)
        
        # Build graphs
        for subtask in subtasks:
            subtask_id = subtask["id"]
            
            # Process dependencies
            for dep_id in subtask.get("dependencies", []):
                # Skip if dependency doesn't exist
                if dep_id not in id_to_subtask:
                    logger.warning(f"Dependency {dep_id} for subtask {subtask_id} not found, skipping")
                    continue
                
                # Add to dependency graph
                dependency_graph[subtask_id].append(dep_id)
                
                # Add to reverse dependency graph
                reverse_dependency_graph[dep_id].append(subtask_id)
        
        return dict(dependency_graph), dict(reverse_dependency_graph)
    
    def _has_circular_dependencies(self, dependency_graph: Dict[str, List[str]]) -> bool:
        """
        Check if the dependency graph has circular dependencies.
        
        Args:
            dependency_graph: Dependency graph
            
        Returns:
            True if circular dependencies exist, False otherwise
        """
        # Keep track of visited and recursion stack
        visited = set()
        rec_stack = set()
        
        def is_cyclic(node):
            visited.add(node)
            rec_stack.add(node)
            
            # Visit all neighbors
            for neighbor in dependency_graph.get(node, []):
                if neighbor not in visited:
                    if is_cyclic(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check all nodes
        for node in dependency_graph:
            if node not in visited:
                if is_cyclic(node):
                    return True
        
        return False
    
    def _resolve_circular_dependencies(
        self, 
        dependency_graph: Dict[str, List[str]], 
        reverse_dependency_graph: Dict[str, List[str]]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Resolve circular dependencies by removing the least important dependencies.
        
        Args:
            dependency_graph: Dependency graph
            reverse_dependency_graph: Reverse dependency graph
            
        Returns:
            Tuple of (updated_dependency_graph, updated_reverse_dependency_graph)
        """
        # Copy graphs
        dep_graph = {k: v.copy() for k, v in dependency_graph.items()}
        rev_dep_graph = {k: v.copy() for k, v in reverse_dependency_graph.items()}
        
        # Find and break cycles
        visited = set()
        rec_stack = set()
        cycle_edges = []
        
        def find_cycle(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            # Visit all neighbors
            for neighbor in dep_graph.get(node, []):
                if neighbor not in visited:
                    if find_cycle(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found a cycle, record the edge
                    cycle_idx = path.index(neighbor)
                    cycle = path[cycle_idx:]
                    for i in range(len(cycle) - 1):
                        cycle_edges.append((cycle[i], cycle[i + 1]))
                    cycle_edges.append((cycle[-1], cycle[0]))
                    return True
            
            rec_stack.remove(node)
            path.pop()
            return False
        
        # Find all cycles
        for node in dep_graph:
            if node not in visited:
                find_cycle(node, [])
        
        # Remove edges to break cycles
        for src, dest in cycle_edges:
            if src in dep_graph and dest in dep_graph[src]:
                dep_graph[src].remove(dest)
                logger.debug(f"Removed dependency edge: {src} -> {dest} to break circular dependency")
            
            if dest in rev_dep_graph and src in rev_dep_graph[dest]:
                rev_dep_graph[dest].remove(src)
        
        return dep_graph, rev_dep_graph
    
    def _create_execution_plan(
        self, 
        subtasks: List[Dict[str, Any]], 
        dependency_graph: Dict[str, List[str]], 
        reverse_dependency_graph: Dict[str, List[str]],
        available_agents: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Create an execution plan based on dependencies and available agents.
        
        Args:
            subtasks: List of normalized subtask specifications
            dependency_graph: Dependency graph
            reverse_dependency_graph: Reverse dependency graph
            available_agents: List of available agent IDs
            
        Returns:
            Execution plan as a list of steps
        """
        # Map of subtask IDs to subtasks
        id_to_subtask = {subtask["id"]: subtask for subtask in subtasks}
        
        # Calculate in-degree (number of dependencies) for each subtask
        in_degree = {subtask["id"]: len(dependency_graph.get(subtask["id"], [])) for subtask in subtasks}
        
        # Queue of ready tasks (no dependencies)
        ready_tasks = deque([subtask["id"] for subtask in subtasks if in_degree[subtask["id"]] == 0])
        
        # Create execution plan
        execution_plan = []
        completed_tasks = set()
        step_counter = 0
        
        while ready_tasks:
            # Create a new step
            step_counter += 1
            step = {
                "step_id": f"step_{step_counter}",
                "subtasks": []
            }
            
            # Select tasks for this step (up to max_parallel_tasks)
            selected_tasks = []
            selected_agents = set()
            
            # Sort ready tasks by complexity if configured
            ready_task_list = list(ready_tasks)
            if self.prioritize_by_complexity:
                ready_task_list.sort(
                    key=lambda task_id: id_to_subtask[task_id].get("complexity", 2),
                    reverse=True
                )
            
            # Select tasks for this step
            for _ in range(min(len(ready_task_list), self.max_parallel_tasks)):
                # Find a task that can be assigned
                best_task_idx = None
                best_task_score = -1
                
                for i, task_id in enumerate(ready_task_list):
                    if task_id in selected_tasks:
                        continue
                    
                    subtask = id_to_subtask[task_id]
                    agent_id = subtask.get("assigned_agent", "")
                    
                    # If no agent is assigned or assigned agent is already busy, skip
                    if agent_id and agent_id in selected_agents:
                        continue
                    
                    # Calculate a score for this task based on complexity and dependencies
                    complexity = subtask.get("complexity", 2)
                    dependent_count = len(reverse_dependency_graph.get(task_id, []))
                    
                    # Score favors high complexity and many dependents
                    score = (complexity * 10) + dependent_count
                    
                    if score > best_task_score:
                        best_task_score = score
                        best_task_idx = i
                
                # If no suitable task found, break
                if best_task_idx is None:
                    break
                
                # Add the best task to selected tasks
                task_id = ready_task_list[best_task_idx]
                subtask = id_to_subtask[task_id]
                agent_id = subtask.get("assigned_agent", "")
                
                selected_tasks.append(task_id)
                if agent_id:
                    selected_agents.add(agent_id)
                
                # Remove from ready tasks
                ready_tasks.remove(task_id)
            
            # Add selected tasks to the step
            for task_id in selected_tasks:
                subtask = id_to_subtask[task_id]
                step["subtasks"].append(subtask)
                
                # Mark as completed
                completed_tasks.add(task_id)
                
                # Update dependencies
                for dependent in reverse_dependency_graph.get(task_id, []):
                    in_degree[dependent] -= 1
                    
                    # If all dependencies are satisfied, add to ready tasks
                    if in_degree[dependent] == 0:
                        ready_tasks.append(dependent)
            
            # Add step to execution plan
            execution_plan.append(step)
        
        # Check if all tasks are scheduled
        if len(completed_tasks) < len(subtasks):
            unscheduled = [subtask["id"] for subtask in subtasks if subtask["id"] not in completed_tasks]
            logger.warning(f"Not all tasks were scheduled! Unscheduled tasks: {unscheduled}")
        
        return execution_plan
    
    def optimize_agent_assignments(
        self, 
        subtasks: List[Dict[str, Any]], 
        available_agents: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Optimize agent assignments based on expertise and workload balance.
        
        Args:
            subtasks: List of subtask specifications
            available_agents: List of available agent IDs
            
        Returns:
            Updated subtask specifications with optimized agent assignments
        """
        # This is a placeholder for a more sophisticated assignment algorithm
        # In a real implementation, this would consider agent specialization,
        # workload balance, etc.
        
        # Currently just ensures each subtask has an assigned agent
        agent_workload = {agent: 0 for agent in available_agents}
        
        for subtask in subtasks:
            # Skip if already assigned
            if subtask.get("assigned_agent") in available_agents:
                agent_workload[subtask["assigned_agent"]] += 1
                continue
            
            # Find the agent with the least workload
            best_agent = min(agent_workload, key=agent_workload.get)
            
            # Assign agent
            subtask["assigned_agent"] = best_agent
            
            # Update workload
            agent_workload[best_agent] += 1
        
        return subtasks
    
    def visualize_schedule(self, schedule: List[Dict[str, Any]]) -> str:
        """
        Create a text visualization of the execution schedule.
        
        Args:
            schedule: Execution schedule
            
        Returns:
            Text visualization of the schedule
        """
        visualization = ["Schedule Visualization:"]
        
        for step in schedule:
            step_id = step["step_id"]
            subtasks = step["subtasks"]
            
            visualization.append(f"\n[{step_id}]")
            
            for subtask in subtasks:
                subtask_id = subtask["id"]
                description = subtask.get("description", "No description")
                agent = subtask.get("assigned_agent", "Unassigned")
                
                visualization.append(f"  - {subtask_id}: {description} (Agent: {agent})")
        
        return "\n".join(visualization)
