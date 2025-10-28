"""
Application Streamlit d'analyse de l'effort culinaire et de la popularité des recettes.

Cette application permet d'explorer la relation entre la complexité des recettes
(effort culinaire) et leur popularité auprès des utilisateurs à travers des
visualisations interactives et des analyses statistiques.

Modules requis:
    - streamlit: Interface utilisateur web
    - pandas: Manipulation des données
    - numpy: Calculs numériques
    - plotly.express: Visualisations interactives
    - pathlib: Gestion des chemins de fichiers

Exemple:
    Pour lancer l'application::

        $ streamlit run src/webapp.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import re
import base64

# Ajouter le répertoire parent au path pour importer les modules locaux
sys.path.append(str(Path(__file__).resolve().parents[1]))


# Initialise le logger au niveau module

logger = None  # Sera initialisé par _setup_logging()

def _setup_logging() -> logging.Logger:
    """
    Configure et initialise le système de logging.
    
    Returns
    -------
    logging.Logger
        Logger configuré pour l'application
        
    Notes
    -----
    Utilise le logger personnalisé si disponible, sinon fallback vers logging standard
    """
    global logger
    try:
        from src.logger import logger as custom_logger
        logger = custom_logger
        return logger
    except ImportError:
        # Fallback si le logger n'est pas disponible
        logger = logging.getLogger(__name__)
        return logger

# Initialiser le logger
logger = _setup_logging()

def _import_analysis_modules() -> tuple[bool, object, object]:
    """
    Importe les modules d'analyse des données.
    
    Returns
    -------
    tuple[bool, object, object]
        - use_real_data : True si les modules sont importés avec succès
        - build_analysis_dataset : Fonction de construction du dataset
        - utils : Module utilitaire d'analyse
        
    Notes
    -----
    En cas d'échec d'import, retourne (False, None, None) et utilise des données simulées
    """
    try:
        from dataset_analysis.dataset_preprocessing import build_analysis_dataset
        from dataset_analysis import utils
        logger.info("Modules d'analyse chargés avec succès")
        return True, build_analysis_dataset, utils
    except ImportError as e:
        logger.error(f"Impossible de charger les modules d'analyse : {e}")
        st.warning("Modules d'analyse non trouvés. Utilisation de données simulées.")
        return False, None, None


def _configure_streamlit() -> None:
    """
    Configure la page Streamlit et les styles CSS.
    
    Notes
    -----
    Définit le titre, l'icône, le layout et les styles personnalisés pour les onglets
    """
    st.set_page_config(
        page_title="Effort Culinaire & Popularité",
        page_icon="🍳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🍳 Analyse de l'Effort Culinaire et de la Popularité des Recettes")
    
    # Styles CSS pour les onglets
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


@st.cache_data
def load_real_datasets(_build_analysis_dataset_func) -> tuple[pd.DataFrame | None, 
                                                           pd.DataFrame | None, 
                                                           pd.DataFrame | None, 
                                                           bool]:
    """
    Charge les datasets réels d'analyse.
    
    Parameters
    ----------
    _build_analysis_dataset_func : callable
        Fonction de construction du dataset d'analyse
        
    Returns
    -------
    tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, bool]
        - recipes_df : DataFrame des recettes ou None
        - interactions_df : DataFrame des interactions ou None  
        - analysis_df : DataFrame d'analyse ou None
        - success : True si le chargement a réussi
        
    Notes
    -----
    Utilise le cache Streamlit pour éviter les rechargements répétés
    """
    try:
        logger.info("Tentative de chargement des données réelles...")
        recipes_df, interactions_df, analysis_df = _build_analysis_dataset_func(save=False)
        logger.info(
            f"Données chargées: {len(recipes_df)} recettes, "
            f"{len(interactions_df)} interactions, {len(analysis_df)} analysables"
        )
        return recipes_df, interactions_df, analysis_df, True
    except FileNotFoundError as e:
        logger.error(f"Fichiers de données non trouvés : {e}")
        st.error("Fichiers de données manquants. "
                "Exécutez d'abord le script de téléchargement des données.")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données réelles : {e}", exc_info=True)
        st.error(f"Erreur lors du chargement des données réelles : {e}")
    return None, None, None, False


@st.cache_data
def generate_sample_data(n_recipes: int = 1000) -> pd.DataFrame:
    """
    Génère des données simulées compatibles avec l'analyse.
    
    Parameters
    ---------- 
    n_recipes : int, default=1000
        Nombre de recettes simulées à générer
        
    Returns
    -------
    pd.DataFrame
        DataFrame contenant les recettes simulées avec toutes les variables nécessaires
        
    Notes
    -----
    Génère des variables d'effort culinaire et de popularité avec des corrélations réalistes
    """
    logger.info(f"Génération de {n_recipes} recettes simulées...")
    
    try:
        np.random.seed(42)
        
        # Variables d'effort culinaire
        n_ingredients = np.random.poisson(8, n_recipes) + 3
        n_steps = np.random.poisson(6, n_recipes) + 2
        minutes = np.random.lognormal(3.5, 0.8, n_recipes)
        log_minutes = np.log(minutes)
        avg_words_per_step = np.random.normal(15, 5, n_recipes).clip(5, 30)
        
        # Score d'effort composite
        effort_score = (
            (n_ingredients - 3) / 15 * 0.25 + 
            (n_steps - 2) / 20 * 0.25 + 
            (log_minutes - 2) / 4 * 0.30 +
            (avg_words_per_step - 5) / 25 * 0.20
        ) * 100
        
        # Variables de popularité
        base_rating = 4.2 - 0.3 * (effort_score / 100) + np.random.normal(0, 0.3, n_recipes)
        avg_rating = np.clip(base_rating, 1, 5)
        bayes_mean = avg_rating + np.random.normal(0, 0.1, n_recipes)
        bayes_mean = np.clip(bayes_mean, 1, 5)
        
        wilson_lb = np.random.beta(2, 1, n_recipes) * 0.8 + 0.1
        
        n_interactions = np.random.poisson(20 * np.exp(-0.5 * (effort_score / 100)), n_recipes) + 1
        age_months = np.random.exponential(24, n_recipes)
        interactions_per_month = n_interactions / np.maximum(1, age_months)
        log1p_interactions_per_month_w = np.log1p(interactions_per_month)
        
        # Variables catégorielles
        def categorize_effort(score: float) -> str:
            """Catégorise le score d'effort en niveau de difficulté."""
            if score <= 15:
                return "Très Facile"
            elif score <= 20:
                return "Facile"
            elif score <= 25:
                return "Modéré"
            elif score <= 30:
                return "Difficile"
            else:
                return "Très Difficile"
        
        effort_category = [categorize_effort(score) for score in effort_score]
        
        logger.info("Données simulées générées avec succès")
        
        return pd.DataFrame({
            'id': range(1, n_recipes + 1),
            'n_ingredients': n_ingredients,
            'n_steps': n_steps,
            'minutes': minutes,
            'log_minutes': log_minutes,
            'avg_words_per_step': avg_words_per_step,
            'effort_score': effort_score,
            'effort_category': effort_category,
            'avg_rating': avg_rating,
            'bayes_mean': bayes_mean,
            'wilson_lb': wilson_lb,
            'age_months': age_months,
            'interactions_per_month': interactions_per_month,
            'log1p_interactions_per_month_w': log1p_interactions_per_month_w
        })
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération des données simulées : {e}", exc_info=True)
        st.error(f"Erreur lors de la génération des données simulées : {e}")
        return pd.DataFrame()


def display_variable_definitions() -> None:
    """
    Affiche les définitions des variables d'effort culinaire et de popularité.
    
    Notes
    -----
    Crée un expander avec deux colonnes contenant les descriptions détaillées
    """
    with st.expander("Définitions des variables", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Variables d'effort culinaire (prédicteurs)")
            st.markdown("""
            **Variables continues :**
            - `log_minutes` : Transformation logarithmique du temps (quantitative continue)
            - `n_steps` : Nombre d'étapes de préparation (discrète)
            - `n_ingredients` : Majorité des recettes entre 5 et 15 ingrédients
            - `avg_words_per_step` : Complexité textuelle des instructions (quantitative continue)
            - `effort_score` : Score composite d'effort (0-100, quantitative continue)
            
            **Variables catégorielles :**
            - `effort_category` : Catégorisation de l'effort (qualitative ordinale : 
              "Très Facile", "Facile", "Modéré", "Difficile", "Très Difficile")
            - `effort_quartile` : Quartiles d'effort (qualitative ordinale)
            - `complexity` : Complexité procédurale
            - `category_minutes` : Catégories de durée       
            """)

        with col2:
            st.subheader("Variables de popularité (variables réponse)")
            st.markdown("""
            **Satisfaction :**
            - `bayes_mean` : Estimateur régularisé (recommandé pour classement)
            - `wilson_lb` : Mesure conservatrice de qualité
            - `avg_rating` : Moyenne simple (baseline)
            - `median_rating` : Mesure robuste
            
            **Engagement :**
            - `log1p_interactions_per_month_w` : Métrique principale (normalisée et robuste)
            - `n_interactions` : Volume brut
            - `n_unique_users` : Diversité de l'audience
            - `log1p_n_interactions_w` : Volume transformé
            
            **Variables de contrôle :**
            - `age_months` : Effet temporel (obligatoire)
            - `n_interactions` : Pondération par le volume
            """)


def display_data_overview(data: pd.DataFrame, 
                         has_real_data: bool, 
                         recipes_df: pd.DataFrame | None = None,
                         interactions_df: pd.DataFrame | None = None) -> None:
    """
    Affiche la vue d'ensemble des données avec métriques clés.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset principal d'analyse
    has_real_data : bool
        True si les données réelles sont utilisées
    recipes_df : pd.DataFrame, optional
        DataFrame des recettes (pour données réelles)
    interactions_df : pd.DataFrame, optional
        DataFrame des interactions (pour données réelles)
        
    Notes
    -----
    Affiche les compteurs généraux et les moyennes des variables importantes
    """
    st.subheader("Vue d'ensemble des données")
    
    try:
        if has_real_data and recipes_df is not None and interactions_df is not None:
            # Première ligne : Compteurs généraux
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Recettes analysables", f"{len(data):,}")
            
            with col2:
                st.metric("Interactions totales", f"{len(interactions_df):,}")
            
            # Moyennes des variables clés
            _display_variable_metrics(data)
        else:
            # Données simulées
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Recettes analysables", f"{len(data):,}")
            
            with col2:
                st.metric("Interactions totales", "Simulées")
            
            _display_variable_metrics(data)
        
        logger.debug("Métriques générales calculées avec succès")
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul des métriques : {e}", exc_info=True)
        st.error(f"Erreur lors du calcul des métriques : {e}")


def _display_variable_metrics(data: pd.DataFrame) -> None:
    """
    Affiche les métriques des variables principales sous forme de colonnes.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset contenant les variables à analyser
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'log_minutes' in data.columns:
            st.metric("log_minutes (moyenne)", f"{data['log_minutes'].mean():.3f}")
        if 'avg_words_per_step' in data.columns:
            st.metric("avg_words_per_step (moyenne)", f"{data['avg_words_per_step'].mean():.2f}")
    
    with col2:
        if 'bayes_mean' in data.columns:
            st.metric("bayes_mean (moyenne)", f"{data['bayes_mean'].mean():.3f}")
        if 'wilson_lb' in data.columns:
            st.metric("wilson_lb (moyenne)", f"{data['wilson_lb'].mean():.3f}")
    
    with col3:
        # Gestion des variations de nom pour effort_score
        if 'effort_score' in data.columns:
            st.metric("effort_score (moyenne)", f"{data['effort_score'].mean():.2f}")
        elif 'score_effort' in data.columns:
            st.metric("score_effort (moyenne)", f"{data['score_effort'].mean():.2f}")
        if 'n_ingredients' in data.columns:
            st.metric("n_ingredients (moyenne)", f"{data['n_ingredients'].mean():.1f}")

def display_variable_statistics(data: pd.DataFrame) -> None:
    """
    Affiche les statistiques descriptives pour une variable sélectionnée.
    Un warning est généré lorsqu'aucune des données sélectionnées n'est disponible.
    Parameters
    ----------
    data : pd.DataFrame
        Dataset contenant les variables à analyser
        
    Notes
    -----
    Permet à l'utilisateur de sélectionner une variable et affiche ses statistiques
    """
    st.subheader("Distribution des variables")
    st.markdown("Sélectionnez une variable pour visualiser sa distribution dans le dataset.")
    
    try:
        # Variables disponibles pour l'analyse
        histogram_vars = {
            'log_minutes': 'Temps de préparation (log)',
            'avg_words_per_step': 'Complexité descriptive',
            'bayes_mean': 'Note bayésienne',
            'wilson_lb': 'Wilson Lower Bound',
            'effort_score': 'Score d\'effort',
            'n_ingredients': 'Nombre d\'ingrédients'
        }
        
        # Filtrer les variables disponibles dans les données
        available_vars = {k: v for k, v in histogram_vars.items() if k in data.columns}
        logger.debug(f"Variables disponibles pour histogrammes : {list(available_vars.keys())}")
        
        if available_vars:
            selected_var = st.selectbox(
                "Variable à visualiser :",
                options=list(available_vars.keys()),
                format_func=lambda x: available_vars[x],
                index=0
            )
            
            logger.debug(f"Variable sélectionnée pour histogramme : {selected_var}")
            


            
            # Affichage des statistiques sous forme de métriques
            _display_descriptive_statistics(data, selected_var, available_vars[selected_var])
        
        else:
            logger.warning("Aucune variable d'histogramme disponible dans les données")
            st.warning("Aucune variable d'histogramme disponible dans les données.")
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération des histogrammes : {e}", exc_info=True)
        st.error(f"Erreur lors de la génération des histogrammes : {e}")

def _display_descriptive_statistics(data: pd.DataFrame, 
                                  selected_var: str, 
                                  var_label: str) -> None:
    """
    Affiche les statistiques descriptives d'une variable.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset contenant la variable
    selected_var : str
        Nom de la variable sélectionnée
    var_label : str
        Libellé descriptif de la variable
    """
    st.subheader(f"Statistiques de {var_label}")
    
    stats_data = data[selected_var].describe()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Moyenne", f"{stats_data['mean']:.3f}")
        st.metric("Médiane", f"{stats_data['50%']:.3f}")
    
    with col2:
        st.metric("Écart-type", f"{stats_data['std']:.3f}")
        st.metric("Minimum", f"{stats_data['min']:.3f}")
    
    with col3:
        st.metric("Q1 (25%)", f"{stats_data['25%']:.3f}")
        st.metric("Q3 (75%)", f"{stats_data['75%']:.3f}")
    
    with col4:
        st.metric("Maximum", f"{stats_data['max']:.3f}")
        st.metric("Observations", f"{int(stats_data['count']):,}")


def display_correlation_analysis(data: pd.DataFrame, has_real_data: bool) -> None:
    """
    Affiche l'analyse de corrélation interactive entre effort culinaire et popularité.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset d'analyse
    has_real_data : bool
        True si les données réelles sont utilisées
        
    Notes
    -----
    Permet la sélection interactive des variables et génère des graphiques de corrélation
    """
    st.subheader("Analyses de corrélation")

    try:
        # Variables disponibles selon le type de données
        if has_real_data:
            effort_vars = ['log_minutes', 'n_steps', 'n_ingredients', 'avg_words_per_step', 'effort_score']
            popularity_vars = ['bayes_mean', 'wilson_lb', 'log1p_interactions_per_month_w', 
                             'n_interactions', 'n_unique_users']
        else:
            effort_vars = ['log_minutes', 'n_steps', 'n_ingredients', 'effort_score']
            popularity_vars = ['bayes_mean', 'wilson_lb', 'n_interactions', 'interactions_per_month']

        # Filtrer les variables existantes
        available_effort = [var for var in effort_vars if var in data.columns]
        available_popularity = [var for var in popularity_vars if var in data.columns]
        
        logger.debug(f"Variables d'effort disponibles : {available_effort}")
        logger.debug(f"Variables de popularité disponibles : {available_popularity}")

        # Sélection des variables
        effort_var = st.sidebar.selectbox(
            "Variable d'effort culinaire:",
            available_effort,
            index=0 if available_effort else None
        )

        popularity_var = st.sidebar.selectbox(
            "Variable de popularité:",
            available_popularity,
            index=0 if available_popularity else None
        )

        if effort_var and popularity_var:
            _generate_correlation_plots(data, effort_var, popularity_var)
    
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de corrélation : {e}", exc_info=True)
        st.error(f"Erreur lors de l'analyse de corrélation : {e}")


def _generate_correlation_plots(data: pd.DataFrame, effort_var: str, popularity_var: str) -> None:
    """
    Génère les graphiques de corrélation entre effort culinaire et popularité.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset d'analyse
    effort_var : str
        Variable d'effort culinaire sélectionnée
    popularity_var : str
        Variable de popularité sélectionnée
    """
    logger.debug(f"Génération du graphique de corrélation : {effort_var} vs {popularity_var}")
    
    # Graphique de corrélation principal
    sample_data = data.sample(min(5000, len(data))) if len(data) > 5000 else data
    
    fig = px.scatter(
        sample_data,
        x=effort_var,
        y=popularity_var,
        size='n_interactions' if 'n_interactions' in data.columns else None,
        color='bayes_mean' if 'bayes_mean' in data.columns else None,
        hover_data=['id'] + (['age_months'] if 'age_months' in data.columns else []),
        title=f"Relation entre {effort_var} et {popularity_var}",
        color_continuous_scale='viridis',
        opacity=0.6
    )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Graphiques complémentaires
    _generate_complementary_plots(data, effort_var, popularity_var)


def _generate_complementary_plots(data: pd.DataFrame, effort_var: str, popularity_var: str) -> None:
    """
    Génère les graphiques complémentaires (histogramme et boxplot).
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset d'analyse
    effort_var : str
        Variable d'effort culinaire
    popularity_var : str
        Variable de popularité
    """
    col1, col2 = st.columns(2)

    with col1:
        # Distribution de l'effort culinaire
        fig_hist_effort = px.histogram(
            data,
            x=effort_var,
            title=f"Distribution de {effort_var}",
            nbins=50
        )
        st.plotly_chart(fig_hist_effort, use_container_width=True)

    with col2:
        # Boxplot par catégories d'effort si disponible
        if 'effort_category' in data.columns:
            fig_box = px.box(
                data,
                x='effort_category',
                y=popularity_var,
                title=f"{popularity_var} par catégorie d'effort"
            )
            fig_box.update_xaxes(tickangle=45)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            # Fallback : quartiles calculés
            _generate_quartile_boxplot(data, effort_var, popularity_var)


def _generate_quartile_boxplot(data: pd.DataFrame, effort_var: str, popularity_var: str) -> None:
    """
    Génère un boxplot basé sur les quartiles d'effort.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset d'analyse
    effort_var : str
        Variable d'effort culinaire
    popularity_var : str
        Variable de popularité
    """
    try:
        data_temp = data.dropna(subset=[effort_var, popularity_var])
        data_temp['effort_quartile'] = pd.qcut(
            data_temp[effort_var], 
            4, 
            labels=['Q1', 'Q2', 'Q3', 'Q4']
        )
        fig_box = px.box(
            data_temp,
            x='effort_quartile',
            y=popularity_var,
            title=f"{popularity_var} par quartile d'effort"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    except Exception as e:
        logger.error(f"Erreur lors de la création du boxplot : {e}")
        st.error(f"Erreur lors de la création du boxplot : {e}")


def display_correlation_matrix(data: pd.DataFrame, has_real_data: bool) -> None:
    """
    Affiche une matrice de corrélation interactive.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset d'analyse
    has_real_data : bool
        True si les données réelles sont utilisées
        
    Notes
    -----
    Permet la sélection des variables et génère une heatmap de corrélation
    """
    st.subheader("Matrice de corrélation")
    
    try:
        # Variables par défaut selon le type de données
        if has_real_data:
            default_corr_vars = [
                'log_minutes', 'n_steps', 'n_ingredients', 'effort_score', 
                'bayes_mean', 'wilson_lb', 'log1p_interactions_per_month_w'
            ]
        else:
            default_corr_vars = [
                'log_minutes', 'n_steps', 'n_ingredients', 'effort_score',
                'bayes_mean', 'wilson_lb', 'n_interactions'
            ]
        
        # Filtrer les variables disponibles
        available_corr_vars = [var for var in default_corr_vars if var in data.columns]
        
        correlation_options = st.multiselect(
            "Sélectionnez les variables pour la matrice de corrélation:",
            options=[col for col in data.columns if data[col].dtype in ['int64', 'float64']],
            default=available_corr_vars[:8]  # Limiter à 8 variables par défaut
        )
        
        if len(correlation_options) >= 2:
            _generate_correlation_heatmap(data, correlation_options)
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la matrice de corrélation : {e}", exc_info=True)
        st.error(f"Erreur lors de la génération de la matrice de corrélation : {e}")


def _generate_correlation_heatmap(data: pd.DataFrame, variables: list[str]) -> None:
    """
    Génère une heatmap de corrélation pour les variables sélectionnées.
    Les données sont vérifiées afin de garantir qu'elles soient numériques.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset d'analyse
    variables : list[str]
        Liste des variables pour la matrice de corrélation
    """
    logger.debug(f"Calcul de la matrice de corrélation pour : {variables}")
    
    try:
        # Calculer la matrice de corrélation
        corr_data = data[variables].select_dtypes(include=[np.number])
        
        # Vérifier qu'il y a des données numériques
        if corr_data.empty:
            logger.warning("Aucune donnée numérique disponible pour la corrélation")
            st.warning("Aucune donnée numérique disponible pour générer la matrice de corrélation.")
            return
        
        corr_matrix = corr_data.corr()
        
        # Créer la heatmap avec Plotly
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            title="Matrice de corrélation",
            color_continuous_scale='RdBu_r',
            aspect="auto",
            zmin=-1,
            zmax=1
        )
        
        fig_corr.update_layout(width=700, height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        logger.debug("Matrice de corrélation générée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la heatmap : {e}")
        st.error(f"Erreur lors de la génération de la matrice de corrélation : {e}")


def display_data_table(data: pd.DataFrame) -> None:
    """
    Affiche un tableau interactif des données.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset à afficher
        
    Notes
    -----
    Affiche un échantillon des données avec les colonnes les plus importantes
    """
    if st.checkbox("Afficher les données"):
        try:
            st.subheader("Données détaillées")
            sample_size = min(1000, len(data))
            st.write(f"Affichage d'un échantillon de {sample_size} recettes sur {len(data)} total")
            
            # Colonnes importantes à afficher en priorité
            priority_cols = ['id', 'effort_score', 'bayes_mean', 'wilson_lb', 'n_interactions', 'age_months']
            available_priority = [col for col in priority_cols if col in data.columns]
            other_cols = [col for col in data.columns if col not in available_priority]
            display_cols = available_priority + other_cols[:10]  # Limiter le nombre de colonnes
            
            st.dataframe(
                data[display_cols].sample(sample_size).reset_index(drop=True), 
                use_container_width=True
            )
            logger.debug(f"Tableau de données affiché avec {sample_size} échantillons")
        
        except Exception as e:
            logger.error(f"Erreur lors de l'affichage du tableau : {e}", exc_info=True)
            st.error(f"Erreur lors de l'affichage du tableau : {e}")


def display_model_configuration(data: pd.DataFrame) -> tuple[list[str], str]:
    """
    Affiche l'interface de configuration des modèles prédictifs.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset d'analyse
        
    Returns
    -------
    tuple[list[str], str]
        - selected_features : Liste des variables prédictives sélectionnées
        - selected_target : Variable cible sélectionnée
        
    Notes
    -----
    Permet la sélection des variables prédictives et de la variable cible
    """
    with st.expander("Configuration des modèles", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Variables prédictives:**")
            default_features = ['log_minutes', 'n_steps', 'n_ingredients', 'effort_score', 'age_months']
            available_features = [f for f in default_features if f in data.columns]
            
            selected_features = st.multiselect(
                "Sélectionnez les variables prédictives:",
                options=available_features,
                default=available_features[:3]
            )
        
        with col2:
            st.write("**Variables cibles:**")
            default_targets = ['bayes_mean', 'wilson_lb', 'log1p_interactions_per_month_w']
            available_targets = [t for t in default_targets if t in data.columns]
            
            selected_target = st.selectbox(
                "Variable à prédire:",
                options=available_targets,
                index=0 if available_targets else None
            )
    
    return selected_features, selected_target


def display_model_performance(model_results: dict) -> None:
    """
    Affiche les métriques de performance des modèles.
    
    Parameters
    ----------
    model_results : dict
        Dictionnaire contenant les résultats des différents modèles
        
    Notes
    -----
    Affiche R², RMSE et autres métriques pour chaque modèle
    """
    st.subheader("Performance des modèles")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        #Gestion des différentes clés possibles
        lr_key = 'linear_regression' if 'linear_regression' in model_results else 'lr'
        if lr_key in model_results:
            st.metric("Régression Linéaire - R²", f"{model_results[lr_key]['r2']:.3f}")
            st.metric("RMSE", f"{model_results[lr_key]['rmse']:.3f}")
        else:
            st.metric("Régression Linéaire - R²", "N/A")
            st.metric("RMSE", "N/A")
    
    with col2:
        rf_key = 'random_forest' if 'random_forest' in model_results else 'rf'
        if rf_key in model_results:
            st.metric("Forêt Aléatoire - R²", f"{model_results[rf_key]['r2']:.3f}")
            st.metric("RMSE", f"{model_results[rf_key]['rmse']:.3f}")
        else:
            st.metric("Forêt Aléatoire - R²", "N/A")
            st.metric("RMSE", "N/A")
    
    with col3:
        if 'ols' in model_results:
            r2_key = 'adj_r2' if 'adj_r2' in model_results['ols'] else 'r2'
            nobs_key = 'nobs' if 'nobs' in model_results['ols'] else 'n_obs'
            
            st.metric("OLS - R² ajusté", f"{model_results['ols'].get(r2_key, 0):.3f}")
            st.metric("Observations", f"{int(model_results['ols'].get(nobs_key, 0))}")
        else:
            st.metric("OLS - R² ajusté", "N/A")
            st.metric("Observations", "N/A")


def display_prediction_visualization(enriched_data: pd.DataFrame, 
                                   model_features: list[str],
                                   selected_features: list[str], 
                                   selected_target: str) -> None:
    """
    Affiche la visualisation des prédictions avec courbes de régression.
    
    Parameters
    ----------
    enriched_data : pd.DataFrame
        Dataset enrichi avec features standardisées
    model_features : list[str]
        Liste des features utilisées pour l'entraînement
    selected_features : list[str]
        Variables prédictives sélectionnées par l'utilisateur
    selected_target : str
        Variable cible sélectionnée
        
    Notes
    -----
    Génère un graphique interactif avec points réels et courbes de prédiction
    """
    st.subheader("Visualisation des prédictions")
    
    try:
        # Préparer les données pour la visualisation
        plot_data = enriched_data.dropna(subset=model_features + [selected_target])
        
        if len(plot_data) > 0:
            display_feature, sample_display_size = _configure_visualization_parameters(
                selected_features, plot_data
            )
            
            # Générer le graphique de prédictions
            _generate_prediction_plot(
                plot_data, model_features, selected_features, selected_target,
                display_feature, sample_display_size
            )
        else:
            st.error("Pas assez de données pour générer les visualisations.")
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération des visualisations : {e}", exc_info=True)
        st.error(f"Erreur lors de la génération des visualisations : {e}")


def _configure_visualization_parameters(selected_features: list[str], 
                                      plot_data: pd.DataFrame) -> tuple[str, int]:
    """
    Configure les paramètres de visualisation des prédictions.
    
    Parameters
    ----------
    selected_features : list[str]
        Variables prédictives sélectionnées
    plot_data : pd.DataFrame
        Dataset pour la visualisation
        
    Returns
    -------
    tuple[str, int]
        - display_feature : Variable à afficher en X
        - sample_display_size : Nombre de points à afficher
    """
    #Gestion du cas où selected_features est vide après avoir créé l'expander
    with st.expander("Paramètres de visualisation", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            if not selected_features:
                logger.warning("Aucune variable sélectionnée pour la visualisation")
                st.warning("Aucune variable sélectionnée")
                display_feature = None
            else:
                display_feature = st.selectbox(
                    "Variable d'effort à afficher en X:",
                    options=selected_features,
                    index=0,
                    help="Sélectionnez la variable d'effort culinaire à utiliser pour l'axe X du graphique"
                )
        
        with col2:
            sample_display_size = st.slider(
                "Nombre de points à afficher", 
                min_value=500, 
                max_value=min(5000, len(plot_data)), 
                value=min(2000, len(plot_data))
            )
    
    return display_feature, sample_display_size


def _generate_prediction_plot(plot_data: pd.DataFrame, 
                            model_features: list[str],
                            selected_features: list[str],
                            selected_target: str,
                            display_feature: str, 
                            sample_display_size: int) -> None:
    """
    Génère le graphique principal des prédictions.
    
    Parameters
    ----------
    plot_data : pd.DataFrame
        Dataset pour la visualisation
    model_features : list[str]
        Features utilisées pour l'entraînement
    selected_features : list[str]
        Variables prédictives sélectionnées
    selected_target : str
        Variable cible
    display_feature : str
        Variable à afficher en X
    sample_display_size : int
        Nombre de points à afficher
    """
    
    # Gestion du cas où display_feature est None
    if display_feature is None:
        logger.warning("Aucune variable d'affichage sélectionnée")
        st.warning("Veuillez sélectionner une variable pour l'affichage")
        return
    
    logger.debug(f"Feature principale pour visualisation : {display_feature}")
    
    # Créer un échantillon pour la performance
    sample_data = plot_data.sample(sample_display_size)
    
    # Préparer les données d'entraînement
    #X_plot = sample_data[model_features].values
    y_true = sample_data[selected_target].values
    X_train = plot_data[model_features].values
    y_train = plot_data[selected_target].values
    
    # Entraîner les modèles
    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    



    
    # Créer le graphique
    fig_pred = _create_prediction_figure(
        sample_data, y_true, display_feature, selected_target, selected_features
    )
    
    # Ajouter les courbes de prédiction
    _add_prediction_curves(
        fig_pred, plot_data, model_features, display_feature, 
        lr_model, rf_model, sample_data
    )
    
    # Finaliser et afficher le graphique
    _finalize_prediction_plot(
        fig_pred, selected_target, display_feature, selected_features, plot_data
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)


def _create_prediction_figure(sample_data: pd.DataFrame, 
                            y_true: np.ndarray,
                            display_feature: str, 
                            selected_target: str,
                            selected_features: list[str]) -> go.Figure:
    """
    Crée la figure de base pour les prédictions.
    
    Parameters
    ----------
    sample_data : pd.DataFrame
        Échantillon des données
    y_true : np.ndarray
        Valeurs réelles de la variable cible
    display_feature : str
        Variable à afficher en X
    selected_target : str
        Variable cible
    selected_features : list[str]
        Variables prédictives sélectionnées
        
    Returns
    -------
    go.Figure
        Figure Plotly initialisée
    """
    fig_pred = go.Figure()
    
    # Points réels avec couleur selon la target
    fig_pred.add_trace(go.Scatter(
        x=sample_data[display_feature],
        y=y_true,
        mode='markers',
        name='Valeurs réelles',
        opacity=0.6,
        marker=dict(
            size=6,
            color=y_true,
            colorscale='viridis',
            colorbar=dict(title=selected_target),
            line=dict(width=0.5, color='darkblue')
        ),
        hovertemplate=(
            f'{display_feature}: %{{x:.3f}}<br>{selected_target}: %{{y:.3f}}<br>'
            f'Autres variables: {", ".join([f for f in selected_features if f != display_feature])}'
            '<extra></extra>'
        )
    ))
    
    return fig_pred


def _add_prediction_curves(fig_pred: go.Figure, 
                          plot_data: pd.DataFrame,
                          model_features: list[str], 
                          display_feature: str,
                          lr_model, rf_model, sample_data: pd.DataFrame) -> None:
    """
    Ajoute les courbes de prédiction au graphique.
    
    Parameters
    ----------
    fig_pred : go.Figure
        Figure Plotly
    plot_data : pd.DataFrame
        Dataset complet
    model_features : list[str]
        Features du modèle
    display_feature : str
        Variable affichée en X
    lr_model : sklearn model
        Modèle de régression linéaire entraîné
    rf_model : sklearn model
        Modèle de forêt aléatoire entraîné
    sample_data : pd.DataFrame
        Échantillon des données
    """
    # Créer une grille pour les courbes de prédiction
    feature_range = np.linspace(
        sample_data[display_feature].min(),
        sample_data[display_feature].max(),
        100
    )
    
    # Créer une matrice pour les prédictions
    X_curve = np.zeros((len(feature_range), len(model_features)))
    
    # Remplir la matrice de prédiction
    for i, feature in enumerate(model_features):
        feature_clean = feature.replace('_std', '')
        
        if feature_clean == display_feature:
            # Variable principale : utiliser la grille
            if feature.endswith('_std'):
                # Si standardisée, convertir la grille
                original_mean = plot_data[display_feature].mean()
                original_std = plot_data[display_feature].std()
                standardized_range = (feature_range - original_mean) / original_std
                X_curve[:, i] = standardized_range
            else:
                X_curve[:, i] = feature_range
        else:
            # Autres features : fixer à leur moyenne
            if feature.endswith('_std'):
                X_curve[:, i] = 0  # Moyenne standardisée
            else:
                X_curve[:, i] = plot_data[feature].mean()
    
    # Prédictions pour les courbes
    y_curve_lr = lr_model.predict(X_curve)
    y_curve_rf = rf_model.predict(X_curve)
    
    # Ajouter les courbes
    fig_pred.add_trace(go.Scatter(
        x=feature_range,
        y=y_curve_lr,
        mode='lines',
        name='Régression Linéaire',
        line=dict(color='red', width=3),
        hovertemplate=f'{display_feature}: %{{x:.3f}}<br>Prédiction LR: %{{y:.3f}}<extra></extra>'
    ))
    
    fig_pred.add_trace(go.Scatter(
        x=feature_range,
        y=y_curve_rf,
        mode='lines',
        name='Forêt Aléatoire',
        line=dict(color='green', width=3, dash='dash'),
        hovertemplate=f'{display_feature}: %{{x:.3f}}<br>Prédiction RF: %{{y:.3f}}<extra></extra>'
    ))


def _finalize_prediction_plot(fig_pred: go.Figure, 
                            selected_target: str,
                            display_feature: str, 
                            selected_features: list[str],
                            plot_data: pd.DataFrame) -> None:
    """
    Finalise le graphique de prédictions et affiche les informations contextuelles.
    
    Parameters
    ----------
    fig_pred : go.Figure
        Figure Plotly
    selected_target : str
        Variable cible
    display_feature : str
        Variable affichée en X
    selected_features : list[str]
        Variables prédictives sélectionnées
    plot_data : pd.DataFrame
        Dataset complet
    """
    # Informations sur les autres variables dans le titre
    other_features = [f for f in selected_features if f != display_feature]
    other_features_info = (
        f" (autres variables fixées à leur moyenne: {', '.join(other_features)})" 
        if other_features else ""
    )
    
    fig_pred.update_layout(
        title=f"Prédictions de {selected_target} en fonction de {display_feature}{other_features_info}",
        xaxis_title=f"{display_feature}",
        yaxis_title=f"{selected_target}",
        height=600,
        hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Afficher les valeurs des autres variables utilisées
    if len(other_features) > 0:
        st.info(
            " **Variables fixées pour la prédiction:** " + 
            ", ".join([f"{feat}: {plot_data[feat].mean():.3f}" 
                      for feat in other_features if feat in plot_data.columns])
        )


def display_ols_coefficients(model_results: dict) -> None:
    """
    Affiche les coefficients OLS avec tests de significativité.
    
    Parameters
    ----------
    model_results : dict
        Dictionnaire contenant les résultats du modèle OLS
        
    Notes
    -----
    Affiche un tableau avec coefficients, p-values et significativité statistique
    """
    if 'ols' in model_results:
        st.subheader("Coefficients du modèle OLS (avec significativité)")
        
        ols_data = []
        for var, coef in model_results['ols']['coefficients'].items():
            p_val = model_results['ols']['p_values'][var]
            std_err = model_results['ols']['std_err'][var]
            significance = ("***" if p_val < 0.001 else 
                          "**" if p_val < 0.01 else 
                          "*" if p_val < 0.05 else "")
            
            ols_data.append({
                'Variable': var.replace('_std', ''),
                'Coefficient': coef,
                'P-value': p_val,
                'Std Error': std_err,
                'Significativité': significance
            })
        
        ols_df = pd.DataFrame(ols_data)
        st.dataframe(ols_df.round(4), use_container_width=True)
        st.caption("Significativité : *** p<0.001, ** p<0.01, * p<0.05")







def display_about_tab() -> None:
    """ Affiche le contenu de l'onglet "À propos".
    Notes
    -----
    Charge le fichier rapport_analyse_effort_popularite.md ou affiche des informations par défaut
    """
    logger.debug("Chargement de l'onglet À propos")

    rapport_path = Path(__file__).resolve().parents[1] / "docs" / "rapport_analyse_effort_popularite.md"
    images_dir = rapport_path.parent / "images"

    try:
        rapport_content = rapport_path.read_text(encoding="utf-8")

        #Remplace les liens d'images locales par des URLs encodées en base64
        def replace_local_images(match):
            img_path = images_dir / match.group(1)
            if img_path.exists():
                data = base64.b64encode(img_path.read_bytes()).decode()
                suffix = img_path.suffix.lower().lstrip(".")
                return f"![](data:image/{suffix};base64,{data})"
            else:
                logger.warning(f"Image introuvable : {img_path}")
                return match.group(0)

        # Cherche les patterns Markdown du type ![](images/xxx.png)
        rapport_content = re.sub(r"!\[[^\]]*\]\((?:\.\/)?images\/([^)]+)\)", replace_local_images, rapport_content)

        st.markdown(rapport_content, unsafe_allow_html=True)
        logger.debug("rapport_analyse_effort_popularite chargé avec succès")

    except FileNotFoundError:
        logger.warning("Fichier rapport_analyse_effort_popularite.md non trouvé")
        st.error("Impossible de charger le contenu du rapport_analyse_effort_popularite.")
        st.markdown("""
        ## À propos du projet
        ...
        """)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du rapport_analyse_effort_popularite : {e}")
        st.error(f"Erreur : {e}")


def main() -> None:
    """
    Fonction principale de l'application Streamlit.
    
    Notes
    -----
    Orchestre l'initialisation, le chargement des données et l'affichage des onglets
    """
    global USE_REAL_DATA, build_analysis_dataset, utils
    
    # Initialisation
    logger = _setup_logging()
    USE_REAL_DATA, build_analysis_dataset, utils = _import_analysis_modules()
    _configure_streamlit()
    
    logger.info("Application Streamlit démarrée")
    
    # Création des onglets
    tab_analyse, tab_models, tab_about = st.tabs(["Étude", "Modèles Prédictifs", "À propos de l'EDA"])
    
    # Chargement des données
    logger.debug("Début du chargement des données...")
    
    if USE_REAL_DATA:
        recipes_df, interactions_df, analysis_df, has_real_data = load_real_datasets(build_analysis_dataset)
        if has_real_data:
            data = analysis_df
            logger.info("Utilisation des données réelles")
            st.success("Données réelles chargées avec succès")
            st.info(f"{len(recipes_df):,} recettes | {len(interactions_df):,} interactions | "
                   f"{len(analysis_df):,} recettes analysables")
        else:
            logger.info("Fallback vers les données simulées")
            data = generate_sample_data()
            has_real_data = False
            recipes_df = interactions_df = None
            st.info("Utilisation de données simulées pour la démonstration")
    else:
        logger.info("Utilisation de données simulées")
        data = generate_sample_data()
        has_real_data = False
        recipes_df = interactions_df = None
        st.info("Utilisation de données simulées pour la démonstration")
    
    # Vérification des données
    if data.empty:
        logger.critical("Aucune donnée disponible - arrêt de l'application")
        st.error("Aucune donnée disponible. Impossible de continuer.")
        st.stop()
    
    # Affichage des onglets
    with tab_analyse:
        logger.debug("Chargement de l'onglet Analyse")
        st.sidebar.header("Paramètres")
        
        # Sections de l'onglet analyse
        display_variable_definitions()
        display_data_overview(data, has_real_data, recipes_df, interactions_df)
        display_variable_statistics(data)
        display_correlation_analysis(data, has_real_data)
        display_correlation_matrix(data, has_real_data)
        display_data_table(data)
    
    with tab_models:
        logger.debug("Chargement de l'onglet Modèles")
        st.header("Modèles Prédictifs")
        
        if not USE_REAL_DATA:
            logger.warning("Modules d'analyse non disponibles pour les modèles prédictifs")
            st.warning("Les modèles prédictifs nécessitent le module d'analyse. "
                      "Fonctionnalité non disponible.")
        elif not has_real_data:
            logger.warning("Données réelles non disponibles pour les modèles prédictifs")
            st.warning("Les modèles prédictifs nécessitent les données réelles. "
                      "Utilisez les données simulées dans l'onglet Étude.")
        else:
            st.markdown("""
            Cette section utilise des modèles de machine learning pour prédire la popularité des recettes 
            en fonction de leur effort culinaire.
            """)
            
            try:
                # Configuration des modèles
                selected_features, selected_target = display_model_configuration(data)
                
                if (selected_features and selected_target and 
                    st.button("Entraîner les modèles", type="primary")):
                    
                    logger.info(f"Début de l'entraînement des modèles - "
                               f"Features: {selected_features}, Target: {selected_target}")
                    
                    with st.spinner("Entraînement des modèles en cours..."):
                        try:
                            # Préparation et entraînement
                            enriched_data, meta = utils.add_feature_columns(data)
                            
                            # Utiliser les versions standardisées si disponibles
                            model_features = []
                            for feature in selected_features:
                                std_feature = f"{feature}_std"
                                if std_feature in enriched_data.columns:
                                    model_features.append(std_feature)
                                else:
                                    model_features.append(feature)
                            
                            logger.debug(f"Features utilisées pour l'entraînement : {model_features}")
                            
                            # Entraîner les modèles
                            results = utils.run_models(
                                enriched_data,
                                features=model_features,
                                targets=[selected_target]
                            )
                            
                            logger.info("Modèles entraînés avec succès")
                            st.success("Modèles entraînés avec succès!")
                            
                            # Affichage des résultats
                            if selected_target in results['models']:
                                model_results = results['models'][selected_target]
                                
                                # Sections d'affichage
                                display_model_performance(model_results)
                                display_prediction_visualization(
                                    enriched_data, model_features, selected_features, selected_target
                                )
                                display_ols_coefficients(model_results)
                                
                                logger.info(f"Résultats affichés pour le modèle {selected_target}")
                        
                        except Exception as e:
                            logger.error(f"Erreur lors de l'entraînement des modèles : {e}", exc_info=True)
                            st.error(f"Erreur lors de l'entraînement des modèles : {e}")
            
            except Exception as e:
                logger.error(f"Erreur dans l'onglet modèles : {e}", exc_info=True)
                st.error(f"Erreur dans l'onglet modèles : {e}")
    
    with tab_about:
        display_about_tab()
    
    logger.info("Application Streamlit terminée")


if __name__ == "__main__":
    main()