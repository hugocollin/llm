"""
Ce fichier permet de lire un fichier CSV contenant des données de cours
et les de les insérer dans une base de données SQLite.
"""

import sqlite3
import csv


def creer_bdd(db_file : str):
    """
    Crée une base de données SQLite avec une table pour stocker les données des cours.

    Args:
        db_file (str): Nom du fichier de la base de données.
    """

    # Connexion à la base de données
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Création de la table cours
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS cours (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lien TEXT,
            nom_du_cours TEXT,
            nom_du_chapitre TEXT,
            nom_de_la_matiere TEXT,
            niveau TEXT,
            texte TEXT
        )
    """
    )

    # Sauvegarde des changements
    conn.commit()
    conn.close()


def inserer_donnees_de_csv(csv_file : str, db_file : str):
    """
    Lit un fichier CSV et insère les données dans une base de données SQLite.

    Args:
        csv_file (str): Nom du fichier CSV contenant les données.
        db_file (str): Nom du fichier de la base de données.
    """

    # Connexion à la base de données
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Lecture du fichier CSV et insertion des données dans la table cours
    with open(csv_file, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="|")
        next(reader)
        for row in reader:
            if len(row) == 6:
                cursor.execute(
                    """
                    INSERT INTO cours 
                    (lien, nom_du_cours, nom_du_chapitre, nom_de_la_matiere, niveau, texte)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (row[0], row[1], row[2], row[3], row[4], row[5]),
                )
    conn.commit()
    conn.close()


def afficher_donnees(db_file : str):
    """
    Affiche les données de la table cours.

    Args:
        db_file (str): Nom du fichier de la base de données.
    """

    # Connexion à la base de données
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Affichage des données
    cursor.execute("SELECT * FROM cours LIMIT 2")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

    # Fermeture de la connexion
    conn.close()

if __name__ == "__main__":
    # Définition du fichier CSV et de la base de données SQLite
    CSV_FILE = "cours.csv"
    DB_FILE = "cours_donnees.db"

    # Création de la base de données
    creer_bdd(DB_FILE)

    # Insertion des données du fichier CSV dans la base de données
    inserer_donnees_de_csv(CSV_FILE, DB_FILE)

    # Affichage des données de la base de données
    afficher_donnees(DB_FILE)
