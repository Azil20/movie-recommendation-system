import pandas as pd
import json
import os # N'oublie pas d'ajouter l'import os

def get_data():
    """
    Charge, nettoie et prépare le dataset TMDB 5000 pour le pipeline NLP.
    """
    print("⏳ Étape 1/4 : Chargement du dataset TMDB 5000...")
    
    # Construit le chemin absolu vers le fichier CSV de manière dynamique
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'tmdb_5000_movies.csv')
    
    # Chargement du fichier avec le chemin sécurisé
    df = pd.read_csv(csv_path)
    
    print("🧹 Étape 2/4 : Nettoyage des données manquantes et doublons...")
    # Suppression des doublons stricts basés sur le titre
    df = df.drop_duplicates(subset=['title'])
    # Suppression des films n'ayant pas de résumé (indispensable pour le NLP)
    df = df.dropna(subset=['title', 'overview'])
    
    print("📅 Étape 3/4 : Ingénierie des caractéristiques (Feature Engineering)...")
    # Extraction propre de l'année de sortie pour l'interface utilisateur
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
    
    # Fonction locale pour extraire le texte des objets JSON de TMDB
    def extract_from_json(column_data):
        try:
            # Parse le JSON et extrait la valeur de la clé 'name'
            return " ".join([item['name'] for item in json.loads(column_data)])
        except (json.JSONDecodeError, TypeError):
            return "" # Retourne vide si erreur pour éviter les crashs
            
    # Application de la fonction sur les genres et mots-clés
    df['genres_clean'] = df['genres'].apply(extract_from_json)
    df['keywords_clean'] = df['keywords'].apply(extract_from_json)
    
    print("🍲 Étape 4/4 : Création de la soupe de métadonnées (Metadata Soup)...")
    # Concaténation des champs pertinents pour créer le contexte global du film
    df['metadata_soup'] = df['overview'] + " " + df['genres_clean'] + " " + df['keywords_clean']
    
    print(f"✅ Prétraitement terminé : {len(df)} films validés.")
    
    # Réinitialisation de l'index pour garantir l'alignement avec la matrice TF-IDF
    return df.reset_index(drop=True)

# Permet de tester ce fichier indépendamment
if __name__ == "__main__":
    df_test = get_data()
    print(df_test[['title', 'year', 'genres_clean']].head())