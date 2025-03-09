import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
import os
import sys

# Importer les modules nécessaires de l'application
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.evolution_workflow import EvolutionWorkflow
from core.team_manager import TeamManager
from core.agent_coordinator import AgentCoordinator
from core.knowledge_repository import KnowledgeRepository
from utils.config import Config

class ApplicationInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Agent Management Interface")
        self.root.geometry("800x600")
        
        # Initialiser les composants de l'application
        self.config = Config("config.yaml")
        self.knowledge_repository = KnowledgeRepository(self.config.get("knowledge_repository", {}))
        self.team_manager = TeamManager(self.config.get("team_manager", {}), self.knowledge_repository)
        self.agent_coordinator = AgentCoordinator(self.config.get("agent_coordinator", {}), self.knowledge_repository)
        self.evolution_workflow = EvolutionWorkflow(
            self.config.to_dict(),
            self.team_manager,
            self.agent_coordinator,
            self.knowledge_repository
        )
        
        # Créer l'interface
        self.create_interface()
        
        # Dictionnaire pour stocker les agents et leurs statuts
        self.agent_statuses = {}
        self.agent_labels = {}
        
    def create_interface(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Section de saisie
        input_frame = ttk.LabelFrame(main_frame, text="Request", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        self.input_entry = ttk.Entry(input_frame, width=70)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(side=tk.RIGHT)
        
        self.evolve_button = ttk.Button(button_frame, text="Evolve", command=self.evolve_functionality)
        self.evolve_button.pack(side=tk.LEFT, padx=5)
        
        self.autre_button = ttk.Button(button_frame, text="Autre", command=self.autre_functionality)
        self.autre_button.pack(side=tk.LEFT, padx=5)
        
        # Section inférieure (divisée en deux)
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Panneau des interactions
        interactions_frame = ttk.LabelFrame(bottom_frame, text="Agent Interactions", padding="10")
        interactions_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.interactions_text = scrolledtext.ScrolledText(interactions_frame, wrap=tk.WORD, state='disabled')
        self.interactions_text.pack(fill=tk.BOTH, expand=True)
        
        # Panneau des statuts d'agents
        agents_frame = ttk.LabelFrame(bottom_frame, text="Agent Statuses", padding="10")
        agents_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.agents_canvas = tk.Canvas(agents_frame)
        self.agents_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.status_frame = ttk.Frame(self.agents_canvas)
        self.agents_canvas.create_window((0, 0), window=self.status_frame, anchor=tk.NW)
        
        scrollbar = ttk.Scrollbar(agents_frame, orient=tk.VERTICAL, command=self.agents_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.agents_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.status_frame.bind("<Configure>", lambda e: self.agents_canvas.configure(
            scrollregion=self.agents_canvas.bbox("all")
        ))
        
    def evolve_functionality(self):
        request = self.input_entry.get()
        if not request:
            self.log_interaction("Please enter a request first.", "system")
            return
        
        self.log_interaction(f"Starting Evolve with request: {request}", "system")
        self.evolve_button.config(state="disabled")
        self.autre_button.config(state="disabled")
        
        # Lancer l'évolution dans un thread séparé
        threading.Thread(target=self._run_evolve, args=(request,)).start()
    
    def _run_evolve(self, request):
        try:
            # Mettre à jour l'interface avec les agents prédéfinis
            team_id = "team_code_evolution"  # ID de l'équipe prédéfinie
            team_data = self.knowledge_repository.get_team(team_id)
            
            if team_data:
                # Afficher les agents prédéfinis
                self.root.after(0, self._display_predefined_agents, team_data)
            
            # Exécuter l'évolution
            results = self.evolution_workflow.execute_evolution(request, team_id)
            
            # Afficher les résultats
            self.root.after(0, lambda: self.log_interaction(results.get("summary", "Evolution completed"), "result"))
            
        except Exception as e:
            self.root.after(0, lambda: self.log_interaction(f"Error: {str(e)}", "error"))
        finally:
            self.root.after(0, lambda: self.evolve_button.config(state="normal"))
            self.root.after(0, lambda: self.autre_button.config(state="normal"))
    
    def _display_predefined_agents(self, team_data):
        self.clear_agents()
        
        for agent_spec in team_data.get("agent_specs", []):
            role = agent_spec.get("role", "unknown")
            self.add_agent(role, "inactive")
    
    def autre_functionality(self):
        request = self.input_entry.get()
        if not request:
            self.log_interaction("Please enter a request first.", "system")
            return
        
        self.log_interaction(f"Starting standard team creation with request: {request}", "system")
        self.evolve_button.config(state="disabled")
        self.autre_button.config(state="disabled")
        
        # Lancer la création d'équipe dans un thread séparé
        threading.Thread(target=self._run_team_creation, args=(request,)).start()
    
    def _run_team_creation(self, request):
        try:
            # Créer l'équipe
            team_composition = self.team_manager.analyze_task(request)
            agent_team = self.team_manager.create_team(team_composition)
            
            # Afficher les agents créés dynamiquement
            self.root.after(0, self.clear_agents)
            for agent_id, agent in agent_team.items():
                self.root.after(0, lambda aid=agent_id: self.add_agent(aid, "inactive"))
            
            # Exécuter la tâche
            results = self.agent_coordinator.execute_task(request, agent_team)
            
            # Afficher les résultats
            self.root.after(0, lambda: self.log_interaction(results.get("summary", "Task completed"), "result"))
            
        except Exception as e:
            self.root.after(0, lambda: self.log_interaction(f"Error: {str(e)}", "error"))
        finally:
            self.root.after(0, lambda: self.evolve_button.config(state="normal"))
            self.root.after(0, lambda: self.autre_button.config(state="normal"))
    
    def log_interaction(self, message, category="info"):
        self.interactions_text.config(state='normal')
        
        # Utiliser des couleurs différentes selon la catégorie
        color_map = {
            "system": "blue",
            "error": "red",
            "result": "green",
            "info": "black"
        }
        color = color_map.get(category, "black")
        
        self.interactions_text.tag_config(category, foreground=color)
        self.interactions_text.insert(tk.END, message + "\n", category)
        self.interactions_text.see(tk.END)
        self.interactions_text.config(state='disabled')
    
    def update_agent_status(self, agent_name, status):
        if agent_name in self.agent_statuses:
            self.agent_statuses[agent_name] = status
            label = self.agent_labels[agent_name]
            
            # Mettre à jour la couleur selon le statut
            color_map = {
                "inactive": "lightgray",
                "active": "lightgreen",
                "completed": "lightblue",
                "error": "salmon"
            }
            color = color_map.get(status, "white")
            
            label.config(bg=color)
    
    def add_agent(self, agent_name, status="inactive"):
        if agent_name not in self.agent_statuses:
            frame = ttk.Frame(self.status_frame)
            frame.pack(fill=tk.X, pady=2)
            
            label = tk.Label(frame, text=agent_name, width=30, anchor=tk.W, padx=5)
            label.pack(side=tk.LEFT)
            
            status_label = tk.Label(frame, text=status, width=10)
            status_label.pack(side=tk.RIGHT)
            
            # Set initial color
            color_map = {
                "inactive": "lightgray",
                "active": "lightgreen",
                "completed": "lightblue",
                "error": "salmon"
            }
            color = color_map.get(status, "white")
            status_label.config(bg=color)
            
            self.agent_statuses[agent_name] = status
            self.agent_labels[agent_name] = status_label
            
            # Update scrollregion
            self.status_frame.update_idletasks()
            self.agents_canvas.configure(scrollregion=self.agents_canvas.bbox("all"))
    
    def clear_agents(self):
        # Supprimer tous les widgets du status_frame
        for widget in self.status_frame.winfo_children():
            widget.destroy()
        
        self.agent_statuses = {}
        self.agent_labels = {}

if __name__ == "__main__":
    root = tk.Tk()
    app = ApplicationInterface(root)
    root.mainloop()