from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Effort Culinaire & Popularité",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🍳 Analyse de l'Effort Culinaire et de la Popularité des Recettes")

tab_analyse, tab_about = st.tabs(["Étude", "À propos de l'EDA"])

st.markdown(
    """
    <style>
    div[data-testid="stTabs"] button p {
        font-size: 2rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Section des définitions
# Génération de données simulées pour la démonstration
@st.cache_data
def generate_sample_data(n_recipes=1000):
    """Génère des données simulées pour la démonstration"""
    np.random.seed(42)
    
    # Variables d'effort culinaire
    n_ingredients = np.random.poisson(8, n_recipes) + 3
    n_steps = np.random.poisson(6, n_recipes) + 2
    minutes = np.random.lognormal(3.5, 0.8, n_recipes)
    log_minutes = np.log(minutes)
    
    # Calcul d'un score d'effort composite
    effort_score = (
        (n_ingredients - 3) / 15 + 
        (n_steps - 2) / 20 + 
        (log_minutes - 2) / 4
    ) / 3
    
    # Variables de popularité (corrélées négativement avec l'effort)
    base_rating = 4.2 - 0.3 * effort_score + np.random.normal(0, 0.3, n_recipes)
    avg_rating = np.clip(base_rating, 1, 5)
    
    n_ratings = np.random.poisson(20 * np.exp(-0.5 * effort_score), n_recipes) + 1
    n_reviews = np.random.poisson(5 * np.exp(-0.3 * effort_score), n_recipes)
    
    # Âge des recettes en mois
    age_months = np.random.exponential(24, n_recipes)
    interactions_per_month = (n_ratings + n_reviews) / np.maximum(age_months, 1)
    
    return pd.DataFrame({
        'recipe_id': range(1, n_recipes + 1),
        'n_ingredients': n_ingredients,
        'n_steps': n_steps,
        'minutes': minutes,
        'log_minutes': log_minutes,
        'effort_score': effort_score,
        'avg_rating': avg_rating,
        'n_ratings': n_ratings,
        'n_reviews': n_reviews,
        'age_months': age_months,
        'interactions_per_month': interactions_per_month
    })

# Chargement des données
data = generate_sample_data()

with tab_analyse:
    st.sidebar.header("⚙️ Paramètres")

    with st.expander("📖 Définitions des variables", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Effort Culinaire (X)")
            st.markdown("""
            - **Temps de préparation** (`log_minutes`)
            - **Nombre d'étapes** (`n_steps`)
            - **Nombre d'ingrédients** (`n_ingredients`)
            """)

        with col2:
            st.subheader("📈 Popularité (Y)")
            st.markdown("""
            **Satisfaction :**
            - Note moyenne (`avg_rating`)
            - Nombre de notes (`n_ratings`)

            **Engagement :**
            - Nombre de reviews (`n_reviews`)
            - Interactions par mois (`interactions_per_month`)
            """)

    # Métriques principales
    st.subheader("📊 Vue d'ensemble des données")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Nombre de recettes", f"{len(data):,}")

    with col2:
        st.metric("Note moyenne", f"{data['avg_rating'].mean():.2f}")

    with col3:
        st.metric("Temps moyen (min)", f"{data['minutes'].mean():.0f}")

    with col4:
        st.metric("Ingrédients moyens", f"{data['n_ingredients'].mean():.1f}")

    # Graphiques interactifs
    st.subheader("📈 Analyses interactives")

    # Sélection des variables dans la sidebar
    effort_var = st.sidebar.selectbox(
        "Variable d'effort culinaire:",
        ['n_ingredients', 'n_steps', 'log_minutes', 'effort_score'],
        index=3
    )

    popularity_var = st.sidebar.selectbox(
        "Variable de popularité:",
        ['avg_rating', 'n_ratings', 'n_reviews', 'interactions_per_month'],
        index=0
    )

    # Graphique de corrélation principal
    fig = px.scatter(
        data,
        x=effort_var,
        y=popularity_var,
        size='n_ratings',
        color='avg_rating',
        hover_data=['recipe_id', 'minutes', 'n_ingredients', 'n_steps'],
        title=f"Relation entre {effort_var} et {popularity_var}",
        color_continuous_scale='viridis'
    )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Graphiques complémentaires
    col1, col2 = st.columns(2)

    with col1:
        # Distribution de l'effort culinaire
        fig_hist = px.histogram(
            data,
            x=effort_var,
            title=f"Distribution de {effort_var}",
            nbins=30
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Boxplot par quartiles d'effort
        data['effort_quartile'] = pd.qcut(data[effort_var], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        fig_box = px.box(
            data,
            x='effort_quartile',
            y=popularity_var,
            title=f"{popularity_var} par quartile d'effort"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Corrélations
    st.subheader("🔗 Matrice de corrélation")
    correlation_vars = ['n_ingredients', 'n_steps', 'log_minutes', 'avg_rating', 'n_ratings', 'interactions_per_month']
    corr_matrix = data[correlation_vars].corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        title="Matrice de corrélation",
        color_continuous_scale='RdBu_r',
        aspect="auto"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Tableau des données
    if st.checkbox("Afficher les données"):
        st.subheader("📋 Données détaillées")
        st.dataframe(data.head(100), use_container_width=True)

with tab_about:
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    try:
        readme_content = readme_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.error("Impossible de charger le contenu du README.")
    else:
        st.markdown(readme_content)
