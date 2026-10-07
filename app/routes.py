from flask import render_template, request
from app import app
import joblib
import os
import pandas as pd

# 1. Chargement du modèle IA Hybride au démarrage
model_path = os.path.join(os.path.dirname(__file__), 'models', 'movie_model.pkl')
try:
    model_data = joblib.load(model_path)
    cosine_sim = model_data['similarity_matrix']
    movies_df = model_data['movies_df']
    # LA LIGNE FAUTIVE A ÉTÉ SUPPRIMÉE ICI
    movies_df['title_lower'] = movies_df['title'].str.lower()
except Exception as e:
    print(f"Erreur de chargement de l'IA : {e}")
    movies_df = None
    svd_model = None

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    search_type = ""
    error = ""
    
    # On simule un utilisateur connecté (User n°1). 
    # Plus tard, tu pourras utiliser Flask-Session pour récupérer le vrai ID.
    current_user_id = 1 

    if request.method == 'POST':
        user_input = request.form.get('user_input', '').strip().lower()

        if movies_df is not None and user_input:
            
            # --- SCÉNARIO 1 : RECOMMANDATION HYBRIDE (SVD sklearn) ---
            if user_input in movies_df['title_lower'].values:
                search_type = "Recommandation Hybride (NLP + Factorisation de Matrice)"
                
                idx = movies_df[movies_df['title_lower'] == user_input].index[0]
                
                # ÉTAPE A : Le NLP sélectionne un "pool" de 50 candidats pertinents
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                candidate_indices = [i[0] for i in sim_scores[1:51]] 
                
                # Récupération de la matrice des prédictions
                predicted_ratings = model_data.get('predicted_ratings')
                hybrid_candidates = []
                
                # ÉTAPE B : Réordonner avec la matrice SVD
                for i in candidate_indices:
                    movie = movies_df.iloc[i]
                    
                    # On vérifie si l'utilisateur (id=1) et le film existent dans la matrice
                    if predicted_ratings is not None and current_user_id in predicted_ratings.index and movie['movie_id'] in predicted_ratings.columns:
                        predicted_rating = predicted_ratings.loc[current_user_id, movie['movie_id']]
                    else:
                        predicted_rating = 2.5 # Note moyenne par défaut si inconnu
                        
                    hybrid_candidates.append({
                        'title': movie['title'], 
                        'year': movie['year'], 
                        'genres': movie['genres'],
                        'match_score': round((predicted_rating / 5) * 100) # Conversion en %
                    })
                
                # On trie la liste finale par ordre de note décroissante
                hybrid_candidates = sorted(hybrid_candidates, key=lambda x: x['match_score'], reverse=True)
                
                # Top 5
                recommendations = hybrid_candidates[:5]

            # --- SCÉNARIO 2 : FILTRE BASIQUE (Inchangé) ---
            else:
                search_type = "Recherche par Filtre de Genres"
                keywords = user_input.replace(',', ' ').split()
                
                def match_genres(genre_string):
                    if not isinstance(genre_string, str): return False
                    return any(word in genre_string.lower() for word in keywords)
                
                matched_movies = movies_df[movies_df['genres'].apply(match_genres)]
                
                if not matched_movies.empty:
                    top_matches = matched_movies.head(5)
                    for _, movie in top_matches.iterrows():
                        recommendations.append({
                            'title': movie['title'], 
                            'year': movie['year'], 
                            'genres': movie['genres'],
                            'match_score': "N/A"
                        })
                else:
                    error = f"Oups, je n'ai pas trouvé de film nommé '{user_input}' ni de genre correspondant."

    return render_template('index.html', recommendations=recommendations, search_type=search_type, error=error)

@app.route('/stats')
def stats():
    if movies_df is not None:
        stats_data = {
            'total': len(movies_df),
            'avg_year': int(movies_df['year'][movies_df['year'] > 0].mean()),
            'algorithm': 'NLP (TF-IDF) + Collaborative (SVD)',
            'architecture': 'Hybrid Two-Stage Recommender'
        }
    else:
        stats_data = {}

    return render_template('stats.html', stats=stats_data)