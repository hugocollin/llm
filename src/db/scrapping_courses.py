"""
Ce fichier définit la classe pour récupérer les cours.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd


class ScrappingCourses:
    """
    Classe pour récupérer les cours.
    """


    def __init__(self):
        """
        Constructeur de la classe ScrappingCourses.
        """
        self.base_url = "https://www.digischool.fr"
        self.urls_depart = [
            "https://www.digischool.fr/primaire",
            "https://www.digischool.fr/college",
            "https://www.digischool.fr/lycee"
        ]


    def get_links(self, url : str, prefix : str) -> list:
        """
        Méthode pour récupérer les liens des cours.

        Args:
            url (str): URL de la page.
            prefix (str): Préfixe des liens des cours.
    
        Returns:
            list: Liste des liens des cours.
        """

        # Récupération de la réponse de la requête
        response = requests.get(url, timeout=10)
        liens = []

        # Parsage du contenu de la page pour récupérer les liens des cours pour un niveau donné
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            balises_a = soup.find_all('a', href=True)
            liens_filtres = [a['href'] for a in balises_a if a['href'].startswith(prefix)]
            liens_complets = [self.base_url + lien for lien in liens_filtres]
            liens.extend(liens_complets)
        return liens


    def get_lesson(self, url: str, cours_data: list):
        """
        Méthode pour récupérer le contenu des cours.

        Args:
            url (str): URL de la page.
            cours_data (list): Liste pour stocker les données des cours.
        """

        # Récupération de la réponse de la requête
        response = requests.get(url, timeout=10)

        # Parsage du contenu de la page pour récupérer les liens des cours d'une matière donnée
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            balises_a = soup.find_all('a', href=True)
            liens_cours = [a['href'] for a in balises_a if a['href'].startswith('/cours/')]
            liens_complets = ["https://www.digischool.fr" + lien for lien in liens_cours]

            for lien in liens_complets:
                # Extraction du nom du cours
                partie_cours = lien.split('/cours/')[1]
                nom_cours = partie_cours.split('?')[0].replace('-', ' ')

                # Extraction du nom du chapitre
                if '%2C' in lien:
                    nom_chapitre = lien.split('%2C')[-1].replace('-', ' ')
                else:
                    nom_chapitre = "N/A"

                # Extraction du nom de la matière et du niveau
                parties = lien.split('%2C')
                if len(parties) >= 3:
                    nom_matiere = parties[-2].replace('-', ' ')
                    niveau = parties[-3].replace('-', ' ')
                else:
                    nom_matiere = "N/A"
                    niveau = "N/A"

                # Récupération de la réponse de la requête
                second_response = requests.get(lien, timeout=10)

                # Parsage du contenu de la page pour récupérer le contenu des cours
                if second_response.status_code == 200:
                    second_soup = BeautifulSoup(second_response.text, 'html.parser')
                    span_latex = second_soup.find('span', class_='__Latex__')
                    if span_latex:
                        article = span_latex.find_parent('article')
                        if article:
                            texte_nettoye = article.get_text(separator=' ', strip=True)
                        else:
                            texte_nettoye = "Balise </article> non trouvée."
                    else:
                        texte_nettoye = "Balise <span class=\"__Latex__\"> non trouvée."
                else:
                    texte_nettoye = "Erreur lors de la récupération de la page."

                # Stockage des informations
                cours_info = {
                    "lien": lien,
                    "nom_du_cours": nom_cours,
                    "nom_du_chapitre": nom_chapitre,
                    "nom_de_la_matiere": nom_matiere,
                    "niveau": niveau,
                    "texte": texte_nettoye
                }

                cours_data.append(cours_info)

            # Mise en forme des informations
            if len(liens_cours) > 0:
                exemple_lien = liens_cours[0]
                parties_exemple = exemple_lien.split('%2C')
                if len(parties_exemple) >= 3:
                    nom_matiere_ex = parties_exemple[-2].replace('-', ' ')
                    niveau_ex = parties_exemple[-3].replace('-', ' ')
                else:
                    nom_matiere_ex = "N/A"
                    niveau_ex = "N/A"
                print(
                    f"[INFO] Tous les cours de '{nom_matiere_ex}' "
                    f"de niveau '{niveau_ex}' ont été récupérés."
                )
        else:
            print("Erreur lors de la récupération de la page.")


    def run(self):
        """
        Méthode principale.
        """
        print("[INFO] Démarrage de la récupération des cours...")

        # Définition de la structure des liens
        toutes_les_pages = []
        for url in self.urls_depart:
            if '/primaire' in url:
                prefix = '/primaire/'
            elif '/college' in url:
                prefix = '/college/'
            elif '/lycee' in url:
                prefix = '/lycee/'
            else:
                prefix = '/'
            liens = self.get_links(url, prefix)
            toutes_les_pages.extend(liens)

        # Traitement spécial pour les cours de première et terminale
        sous_categories = [
            lien for lien in toutes_les_pages
            if '/lycee/premiere' in lien
            or '/lycee/terminale' in lien
        ]
        for sous_url in sous_categories:
            path_relatif = sous_url.replace(self.base_url, '')
            sous_prefix = path_relatif + '/'
            sous_liens = self.get_links(sous_url, sous_prefix)
            toutes_les_pages.extend(sous_liens)

        # Suppression des doublons
        toutes_les_pages = list(set(toutes_les_pages))

        cours_data = []

        # Itération sur les pages pour récupérer les cours
        for page in toutes_les_pages:
            self.get_lesson(page, cours_data)

        # Conversion des données en DataFrame et exportation en CSV
        df = pd.DataFrame(cours_data)
        df.to_csv('cours.csv', index=False, sep="|", encoding="utf-8")
        print("[INFO] Les données ont été sauvegardées.")


# Fonction principale
if __name__ == "__main__":
    sc = ScrappingCourses()
    sc.run()
