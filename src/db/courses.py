import sqlite3
import csv
import os

# Définir le fichier CSV d'entrée et la base de données SQLite
csv_file = "cours.csv"  # Remplace par le chemin de ton fichier CSV
db_file = "cours_donnees.db"

# Fonction pour créer la base de données et la table
def creer_bdd(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # Créer une table si elle n'existe pas déjà
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cours (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lien TEXT,
            nom_du_cours TEXT,
            nom_du_chapitre TEXT,
            nom_de_la_matiere TEXT,
            niveau TEXT,
            texte TEXT
        )
    """)
    conn.commit()
    conn.close()

# Fonction pour lire le fichier CSV et insérer les données dans la base de données
def inserer_donnees_de_csv(csv_file, db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    with open(csv_file, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="|")  # Lire le CSV avec | comme séparateur
        next(reader)  # Sauter l'en-tête si le fichier en contient
        for row in reader:
            if len(row) == 6:  # Vérifier que la ligne contient exactement 6 colonnes
                cursor.execute("""
                    INSERT INTO cours (lien, nom_du_cours, nom_du_chapitre, nom_de_la_matiere, niveau, texte)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (row[0], row[1], row[2], row[3], row[4], row[5]))
    conn.commit()
    conn.close()

# Fonction pour afficher les données insérées
def afficher_donnees(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cours LIMIT 2")  # Limiter à 5 lignes pour l'exemple
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    conn.close()


# Créer la base de données et la table
creer_bdd(db_file)

# Lire les données depuis le fichier CSV et les insérer dans la base de données
inserer_donnees_de_csv(csv_file, db_file)

# Afficher les données pour vérifier l'insertion
print(f"Les données ont été sauvegardées dans la base de données SQLite : {os.path.abspath(db_file)}")
afficher_donnees(db_file)
