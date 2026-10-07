# 🎬 CineMatch — Hybrid Movie Recommendation System

> An intelligent **Hybrid Recommendation Engine** combining **Content-Based Filtering (NLP TF-IDF + Cosine Similarity)** with **Collaborative Filtering (TruncatedSVD Matrix Factorization)**, served via an interactive Flask web application.

---

## 🚀 Key Features

- **Hybrid Recommendation Architecture:**
  - **Content-Based NLP:** Extracts plot, genres, and metadata into a TF-IDF vector space to calculate semantic cosine similarities across films.
  - **Collaborative Filtering:** Applies **TruncatedSVD** (Singular Value Decomposition) to factorize user-item interaction matrices and predict latent preferences.
- **Dataset:** TMDB 5000 Movies Dataset (	mdb_5000_movies.csv).
- **Interactive Web Interface:** Real-time search, movie exploration, dynamic recommendations, and analytics dashboard.
- **Visual Analytics:** Generates genre distributions, similarity heatmaps, and SVD cumulative explained variance curves.

---

## 📁 Project Structure

`
movie-recommendation-system/
├── app/
│   ├── __init__.py           # Flask application factory
│   ├── routes.py             # Recommendation & API routes
│   └── templates/
│       ├── index.html        # Interactive recommendation portal
│       └── stats.html        # Analytics and model insights
├── ml/
│   ├── preprocess.py         # Data cleaning & metadata soup construction
│   ├── train.py              # Hybrid model training (TF-IDF + SVD)
│   ├── evaluate.py           # Evaluation metrics & report generation
│   └── rapports/             # Generated visual evaluation plots
│       ├── 1_genres_distribution.png
│       ├── 2_similarity_matrix.png
│       └── 3_svd_variance_curve.png
├── notebooks/
│   └── eda_movies.ipynb      # Exploratory Data Analysis notebook
├── tmdb_5000_movies.csv      # TMDB 5000 Movies Dataset
├── requirements.txt          # Python dependencies
├── run.py                    # Application entry point
└── README.md
`

---

## ⚡ Quick Start

### 1. Clone & Install Dependencies

`ash
git clone https://github.com/Azil20/movie-recommendation-system.git
cd movie-recommendation-system
pip install -r requirements.txt
`

### 2. Train the Hybrid Model

`ash
cd ml
python train.py
cd ..
`

### 3. Launch the Web Application

`ash
python run.py
`

Navigate to: http://localhost:5000

---

## 📊 Analytics & Visualizations

The pipeline generates analytical reports available under ml/rapports/:
- **Genre Distribution:** Breakdown of movie genres in the dataset.
- **Similarity Matrix:** Heatmap of cosine similarities between movie embeddings.
- **SVD Variance Curve:** Cumulative explained variance across latent dimensions.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Machine Learning:** Scikit-Learn (TruncatedSVD, TfidfVectorizer, cosine_similarity), NumPy, Pandas
- **Web Framework:** Flask, Jinja2, HTML5/CSS3
- **Visualization:** Matplotlib, Seaborn
