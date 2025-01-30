# README : LLM

## Table des matières
- [Description](#description)
- [Fonctionnalités principales](#fonctionnalités-principales)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Contribution](#contribution)
- [Auteurs](#auteurs)

## Description

SISE Classmate est un assistant conversationnel spécialisé dans le domaine de l'éducation fonctionnant grâce aux modèles d'intelligence artificielle de Mistral et Gemini.

Cette application offre une gamme complète de fonctionnalités, incluant des interactions avec une IA avancée, la génération de suggestions de messages, la personnalisation des paramètres, l'intégration de documents, l'enrichissement des réponses via internet, des quiz interactifs, la visualisation de statistiques détaillées, la personnalisation des noms de conversation et un espace sécurisé assuré par le module Guardian.

### Fonctionnalités principales

- Discuter avec l'IA : Posez vos questions et obtenez des réponses précises et approfondies sur vos cours. L'IA, ayant accès à plus de 6 000 cours, vous aide à réviser et à mieux comprendre les sujets abordés en classe.

- Obtenir des suggestions de messages : L'IA génère automatiquement cinq suggestions de questions pour faciliter vos interactions et mieux formuler vos demandes.

- Paramétrer l'IA selon vos besoins : Personnalisez les paramètres de l'IA, comme le fournisseur, le modèle et la température, pour une expérience adaptée à vos besoins.

- Ajouter des documents à la discussion : Importez des fichiers PDF pour permettre à l'IA d'analyser leur contenu et d'enrichir ses réponses.

- Enrichir les réponses grâce à internet : Activez la recherche en ligne pour obtenir des réponses actualisées et plus pertinentes.

- S'entraîner grâce à des quiz : Générez des quiz interactifs basés sur le sujet de discussion pour tester et renforcer vos connaissances.

- Visualiser les statistiques de conversation : Consultez des statistiques détaillées sur l'utilisation de l'application, incluant notamment la latence, le coût, l'énergie consommée et l'empreinte carbone des messages et plus encore.

- Personnaliser les noms de conversation : Obtenez automatiquement un nom pertinent pour chaque conversation ou personnalisez-le selon vos préférences.

- Assurez un espace d'échange sécurisé grâce au Guardian : Le Guardian est un module de sécurité qui bloque les messages inappropriés, non pertinents ou dangereux, en limitant les interactions aux sujets éducatifs, scolaires et de culture générale. Il protège également contre les attaques, telles que les tentatives d'injection SQL, garantissant un espace d'apprentissage sûr, sain et fiable.

### Structure du projet

```bash
├── .streamlit
│    └── config.toml
├── docs
│   └── README.md 
├── src
│   ├── app
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   └── components.py
│   ├── db
│   │   ├── __init__.py
│   │   ├── db.ipynb
│   │   ├── courses.py
│   │   ├── llm_database.db
│   │   └── scrapping_courses.py
│   ├── llm
│   │   ├── __init__.py
│   │   └── rag.py
│   ├── ml
│   │   ├── __init__.py
│   │   ├── best_prompt_model.pkl
│   │   ├── guardrail_dataset_test.json
│   │   ├── guardrail_dataset_train.json
│   │   └── promptClassifier.py
│   ├── security
│   │   ├── __init__.py
│   │   └── securite.py
│   ├── __init__.py
│   └── pipelines.py
├── .env # Placez le fichier .env à la racine du projet
├── .gitignore
├── main.py
└── requirements.txt
```

## Installation

Pour installer ce projet :

1. Clonez le dépôt sur votre machine locale, en utilisant la commande suivante :

```bash
git clone https://github.com/hugocollin/llm
```

2. Puis récupérez le fichier `.env` (qui vous a été envoyé par mail) contenant la clé API Mistral et Gemini et placez-le à la racine du projet (comme indiqué dans la [Structure du projet](#structure-du-projet)).

## Utilisation

Pour utiliser cette application vous avez 2 méthodes :

### I. Utilisez l'application en local

1. Installez et activez un environnement Python avec une version 3.11.

2. Ouvrez votre terminal et déplacez-vous à la racine du projet.

3. Exécutez la commande suivante pour installer les dépendances du projet :

```bash
pip install -r requirements.txt
```
*Le fichier `requirements.txt` contient les dépendances nécessaires à Streamlit Cloud pour exécuter l'application. En utilisant l'application en local, vous aurez peut être besoin d'installer des dépendances supplémentaires.*

4. Exécutez la commande suivante pour lancer l'application :

```bash
streamlit run main.py
```

5. Ouvrez votre navigateur et accédez à l'adresse suivante : [http://localhost:8501](http://localhost:8501)

### II. Utilisez l'application en ligne

Ouvrez votre navigateur et accédez à l'adresse suivante : [https://sise-classmate.streamlit.app](https://sise-classmate.streamlit.app)

## Contribution

Toutes les contributions sont les bienvenues ! Voici comment vous pouvez contribuer :

1. Forkez le projet.
2. Créez votre branche de fonctionnalité  (`git checkout -b feature/AmazingFeature`).
3. Commitez vos changements (`git commit -m 'Add some AmazingFeature'`).
4. Pushez sur la branche (`git push origin feature/AmazingFeature`).
5. Ouvrez une Pull Request. 

## Auteurs

Cette application a été développée par [KPAMEGAN Falonne](https://github.com/marinaKpamegan), [KARAMOKO Awa](https://github.com/karamoko17), [CISSE Lansana](https://github.com/lansanacisse) et [COLLIN Hugo](https://github.com/hugocollin), dans le cadre du Master 2 SISE.