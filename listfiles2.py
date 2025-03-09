import os
def list_files_and_directories(startpath, log_file, excluded_dirs=None, max_size=10 * 1024 * 1024):
    """
    Liste les fichiers et répertoires à partir d'un chemin de départ et écrit leur contenu dans un fichier log.
    
    Args:
        startpath (str): Chemin de départ pour l'analyse
        log_file (str): Nom du fichier de log
        excluded_dirs (list): Liste des noms de répertoires à exclure de l'analyse
        max_size (int): Taille maximale du fichier de log (en octets)
    """
    # Si excluded_dirs n'est pas spécifié, initialiser une liste vide
    if excluded_dirs is None:
        excluded_dirs = []
    
    current_log_file = log_file
    file_count = 1
    
    # Ouvrir le premier fichier de log
    log = open(current_log_file, 'w', encoding='utf-8')
    
    try:
        for root, dirs, files in os.walk(startpath):
            # Filtrer les répertoires à exclure
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * level
            log.write(f"{indent}{os.path.basename(root)}/\n")
            subindent = ' ' * 4 * (level + 1)
            
            for f in files:
                # Exclure les fichiers avec l'extension .whl
                if f.endswith('.whl'):
                    continue
                    
                file_path = os.path.join(root, f)
                log.write(f"{subindent}{f}\n")
                
                try:
                    # Vérifier si le fichier est trop grand pour être lu
                    file_size = os.path.getsize(file_path)
                    if file_size > max_size:
                        log.write(f"{subindent}Fichier trop volumineux pour être analysé ({file_size} octets)\n")
                        continue
                        
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        if content.strip():  # Vérifie si le contenu n'est pas vide ou seulement des espaces
                            # Exclure les lignes contenant des références à des fichiers .whl
                            filtered_content = "\n".join(
                                line for line in content.splitlines() if '.whl' not in line
                            )
                            if filtered_content.strip():  # Vérifie si le contenu filtré n'est pas vide
                                log.write(f"\nContent of {file_path}:\n")
                                log.write(filtered_content)
                                log.write("\n" + "-" * 80 + "\n")
                except Exception as e:
                    log.write(f"Error reading {file_path}: {e}\n")
                
                # Vérifie si le fichier de log dépasse la taille maximale
                log.flush()  # S'assurer que toutes les données sont écrites
                if os.path.getsize(current_log_file) > max_size:
                    # Fermer le fichier actuel avant de créer un nouveau
                    log.close()
                    
                    # Préparer le nom du nouveau fichier de log
                    file_parts = log_file.split('.')
                    if len(file_parts) > 1:
                        new_log_file = f"{file_parts[0]}_part{file_count+1}.{file_parts[1]}"
                    else:
                        new_log_file = f"{log_file}_part{file_count+1}"
                    
                    file_count += 1
                    current_log_file = new_log_file
                    
                    # Ouvrir le nouveau fichier de log
                    log = open(current_log_file, 'w', encoding='utf-8')
    finally:
        # S'assurer que le fichier est fermé correctement, même en cas d'erreur
        if not log.closed:
            log.close()

# Chemin de départ et fichier de log
startpath = "C:/AI PROJECT/My Langchain Crew 1"
log_file = "file_contents_log.txt"
# Liste des répertoires à exclure de l'analyse
excluded_dirs = [".git", "__pycache__", "venv", "node_modules", "books","vector_store"]
# Appel de la fonction pour lister les fichiers et dossiers et extraire les contenus
list_files_and_directories(startpath, log_file, excluded_dirs)