import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import joblib
import os
from preprocess import get_data

def train_model():
    print("\n" + "="*40)
    print("DÉMARRAGE DU PIPELINE HYBRIDE (NLP + SVD)")
    print("="*40 + "\n")
    
    # 1. RÉCUPÉRATION ET NLP (Content-Based)
    df = get_data()
    # On crée une colonne ID propre pour faire le lien avec les notes
    df['movie_id'] = df.index 
    
    print("Vectorisation du texte (TF-IDF)...")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['metadata_soup'])
    
    print("🧮 Calcul de la matrice Cosinus...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 2. FILTRAGE COLLABORATIF (SVD)
    print("🤖 Intégration des notes utilisateurs (Collaboratif avec Sklearn)...")
    
    np.random.seed(42)
    mock_ratings = pd.DataFrame({
        'userId': np.random.randint(1, 100, 10000),
        'movieId': np.random.randint(0, len(df), 10000),
        'rating': np.random.uniform(1, 5, 10000).round(1)
    })

    # Création de la matrice Utilisateurs-Films (avec pivot_table pour gérer les doublons)
    user_movie_matrix = mock_ratings.pivot_table(index='userId', columns='movieId', values='rating', aggfunc='mean').fillna(0)
    
    # Application de SVD
    print("📈 Entraînement du modèle SVD (Scikit-Learn)...")
    svd = TruncatedSVD(n_components=20, random_state=42)
    user_factors = svd.fit_transform(user_movie_matrix)
    
    # Reconstruction de la matrice de prédictions complètes
    predicted_ratings_matrix = np.dot(user_factors, svd.components_)
    preds_df = pd.DataFrame(predicted_ratings_matrix, columns=user_movie_matrix.columns, index=user_movie_matrix.index)

    # 3. EXPORTATION
    print("📦 Préparation des données pour l'API Web...")
    movies_ui = df[['movie_id', 'title', 'year', 'genres_clean']].rename(columns={'genres_clean': 'genres'})

    model_data = {
        'similarity_matrix': cosine_sim,
        'movies_df': movies_ui,
        'predicted_ratings': preds_df # La nouvelle matrice des prédictions
    }

    models_dir = os.path.join('..', 'app', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'movie_model.pkl')
    joblib.dump(model_data, model_path)

    print("\n" + "="*40)
    print(f"🎉 SUCCÈS ! Modèle Hybride sauvegardé dans :\n{model_path}")
    print("="*40)

if __name__ == "__main__":
    train_model()