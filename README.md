# Convergence du Visuel et de l'Audible par l'Intelligence Artificielle

Ce projet explore la convergence entre le visuel et l'audible via l'intelligence artificielle, en utilisant des techniques de sonification pour transformer des images en sons. Le projet comprend une web app, des modèles d'analyse d'images, et un environnement pour la synthèse sonore.

## Structure du Projet

Le projet est organisé de la manière suivante :

- **images** : Dossier de dépôt des œuvres à sonifier. Les images chargées via la web app sont déposées ici.
- **models** : Contient les différents modèles utilisés pour l'analyse des œuvres.
- **models_training** : Regroupe les scripts Python utilisés pour entraîner les modèles d'analyse d'image.
- **recordings** : Dossier où sont enregistrés les fichiers audio après la synthèse sonore.
- **results** : Contient les fichiers JSON générés lors de l'analyse des images.
- **scripts** : Contient les scripts Shell utilisés pour l'automatisation du processus de sonification.
- **scscripts** : Regroupe le script SuperCollider utilisé pour la sonification.
- **static** : Dossier contenant les fichiers CSS et JavaScript utilisés par la web app.
- **templates** : Dossier qui abrite les fichiers HTML utilisés par la web app.
- **venv** : Environnement virtuel Python.

## Fichiers Principaux

- **app.py** : Script de configuration de la web app.
- **image_analysis.py** : Script d'analyse d'images faisant appel aux divers modèles.
- **requirements.txt** : Liste des librairies et frameworks Python nécessaires au projet.

## Synthèse Sonore

- **granularsounds** : Contient le son utilisé par le synthétiseur « \granular » dans le patch SuperCollider.
- **supercollider** : Contient l'installation du logiciel de synthèse sonore ainsi que le serveur audio pour l'enregistrement des fichiers générés.

## Environnement Virtuel

Un environnement virtuel Python (version 3.11.9) a été mis en place pour isoler l'environnement d'exécution. Toutes les librairies nécessaires sont installées via le fichier `requirements.txt` qui se trouve à la racine du projet.

