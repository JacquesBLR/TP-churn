# Utiliser une image Python officielle comme base
FROM python:3.11-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers de l'application
COPY app.py .
COPY templates/ templates/
COPY data/ data/
COPY tests tests

# Exposer le port 5000
EXPOSE 5000

# Définir la variable d'environnement pour Flask
ENV FLASK_APP=app.py

# Commande pour lancer l'application
CMD ["python", "app.py"]
