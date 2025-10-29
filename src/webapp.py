"""
Webapp Streamlit orientée storytelling et exploration pour analyser
le lien entre effort culinaire et popularité des recettes.
"""

from __future__ import annotations

import base64
import inspect
import logging
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from PIL import Image

# Ajouter le répertoire parent au path pour importer les modules locaux.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from webapp_utils import (  # noqa: E402
    compute_effort_pattern,
    compute_histogram_figure,
    compute_quartile_pattern,
    fit_simple_regression,
    infer_filter_options,
)

logger: logging.Logger | None = None


def _setup_logging() -> logging.Logger:
    """
    Initialise le logger de l'application.

    :returns: Logger configuré (ou logger générique si l'import échoue).
    :rtype: logging.Logger
    """
    global logger
    try:
        from src.logger import logger as custom_logger

        logger = custom_logger
    except ImportError:
        logger = logging.getLogger(__name__)
    return logger


logger = _setup_logging()

try:
    _PLOTLY_CHART_PARAMS = inspect.signature(st.plotly_chart).parameters  # type: ignore[arg-type]
except (ValueError, TypeError):
    _PLOTLY_CHART_PARAMS = {}

_PLOTLY_SUPPORTS_WIDTH = "width" in _PLOTLY_CHART_PARAMS
_PLOTLY_SUPPORTS_USE_CONTAINER = "use_container_width" in _PLOTLY_CHART_PARAMS


def _plotly_config(width: str) -> dict:
    """
    Mappe la notion de largeur souhaitée vers la configuration Plotly appropriée.

    :param width: Largeur souhaitée (`'stretch'` ou `'content'`).
    :type width: str
    :returns: Paramètres de configuration Plotly correspondant.
    :rtype: dict
    """
    responsive = width == "stretch"
    return {
        "responsive": responsive,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }


def _plotly_display(fig: go.Figure, *, width: str = "stretch") -> None:
    """
    Affiche un graphique Plotly en respectant la convention de largeur demandée.

    :param fig: Figure Plotly à représenter.
    :type fig: go.Figure
    :param width: Largeur d'affichage (`'stretch'` ou `'content'`).
    :type width: str
    :returns: ``None``.
    :rtype: None
    :raises ValueError: Si ``width`` n'est pas reconnu.
    """
    if width not in {"stretch", "content"}:
        raise ValueError("La largeur doit être 'stretch' ou 'content'.")
    config = _plotly_config(width)
    if _PLOTLY_SUPPORTS_WIDTH:
        st.plotly_chart(fig, config=config, width=width)
    elif _PLOTLY_SUPPORTS_USE_CONTAINER:
        st.plotly_chart(
            fig,
            config=config,
            use_container_width=width == "stretch",
        )
    else:
        st.plotly_chart(fig, config=config)


def _import_analysis_modules() -> tuple[bool, object | None]:
    """
    Tente d'importer les modules d'analyse réels.

    :returns: Tuple ``(disponible, build_analysis_dataset)``.
    :rtype: tuple[bool, object | None]
    """
    try:
        from dataset_analysis.dataset_preprocessing import build_analysis_dataset

        logger.info("Modules d'analyse chargés avec succès.")
        return True, build_analysis_dataset
    except ImportError as exc:
        logger.warning(
            "Modules d'analyse indisponibles, bascule sur des données simulées : %s", exc
        )
        return False, None


@st.cache_data
def load_real_datasets(_build_analysis_dataset_func):
    """
    Charge les datasets nettoyés lorsqu'ils sont disponibles.

    :param _build_analysis_dataset_func: Fonction permettant de construire les datasets préparés.
    :type _build_analysis_dataset_func: Callable[..., tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]
    :returns: Tuple ``(recipes_df, interactions_df, analysis_df, succès)`` où chaque DataFrame peut être ``None``.
    :rtype: tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, bool]
    """
    try:
        recipes_df, interactions_df, analysis_df = _build_analysis_dataset_func(save=False)
        logger.info(
            "Flux réel chargé (%d recettes, %d interactions).",
            len(recipes_df),
            len(interactions_df),
        )
        return recipes_df, interactions_df, analysis_df, True
    except FileNotFoundError as exc:
        logger.error("Fichiers de données introuvables : %s", exc)
        st.error(
            "Impossible de charger les données réelles. "
            "Exécutez d'abord le pipeline de préparation."
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Erreur inattendue lors du chargement des données réelles : %s", exc)
        st.error(f"Erreur lors du chargement des données réelles : {exc}")
    return None, None, None, False


@st.cache_data
def generate_sample_data(n_recipes: int = 1000) -> pd.DataFrame:
    """
    Génère un dataset simulé pour illustrer la narration lorsqu'aucune donnée réelle
    n'est disponible localement.

    :param n_recipes: Nombre de recettes synthétiques à créer.
    :type n_recipes: int
    :returns: DataFrame simulé reprenant les principales variables d'analyse.
    :rtype: pd.DataFrame
    """
    logger.info("Génération de %d recettes simulées...", n_recipes)
    np.random.seed(42)

    n_ingredients = np.random.poisson(8, n_recipes) + 3
    n_steps = np.random.poisson(6, n_recipes) + 2
    minutes = np.random.lognormal(3.5, 0.8, n_recipes)
    log_minutes = np.log(minutes)
    avg_words_per_step = np.random.normal(15, 5, n_recipes).clip(5, 30)

    effort_score = (
        (n_ingredients - 3) / 15 * 0.25
        + (n_steps - 2) / 20 * 0.25
        + (log_minutes - 2) / 4 * 0.30
        + (avg_words_per_step - 5) / 25 * 0.20
    ) * 100

    base_rating = 4.2 - 0.3 * (effort_score / 100) + np.random.normal(0, 0.3, n_recipes)
    avg_rating = np.clip(base_rating, 1, 5)
    bayes_mean = np.clip(avg_rating + np.random.normal(0, 0.1, n_recipes), 1, 5)

    wilson_lb = np.random.beta(2, 1, n_recipes) * 0.8 + 0.1
    n_interactions = np.random.poisson(
        20 * np.exp(-0.5 * (effort_score / 100)), n_recipes
    ) + 1
    age_months = np.random.exponential(24, n_recipes)
    interactions_per_month = n_interactions / np.maximum(1, age_months)
    log1p_interactions_per_month_w = np.log1p(interactions_per_month)

    def categorize_effort(score: float) -> str:
        if score <= 15:
            return "Très Facile"
        if score <= 20:
            return "Facile"
        if score <= 25:
            return "Modéré"
        if score <= 30:
            return "Difficile"
        return "Très Difficile"

    effort_category = [categorize_effort(score) for score in effort_score]

    simulated_df = pd.DataFrame(
        {
            "id": range(1, n_recipes + 1),
            "n_ingredients": n_ingredients,
            "n_steps": n_steps,
            "minutes": minutes,
            "log_minutes": log_minutes,
            "avg_words_per_step": avg_words_per_step,
            "effort_score": effort_score,
            "effort_category": effort_category,
            "avg_rating": avg_rating,
            "bayes_mean": bayes_mean,
            "wilson_lb": wilson_lb,
            "age_months": age_months,
            "n_interactions": n_interactions,
            "interactions_per_month": interactions_per_month,
            "log1p_interactions_per_month_w": log1p_interactions_per_month_w,
        }
    )
    simulated_df["log_n_ingredients"] = np.log(simulated_df["n_ingredients"].clip(lower=1))

    try:
        simulated_df["effort_quartile"] = pd.qcut(
            simulated_df["effort_score"],
            4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        )
    except ValueError:
        simulated_df["effort_quartile"] = pd.Series(["Q1"] * len(simulated_df))

    return simulated_df


def _configure_streamlit() -> None:
    """
    Applique les réglages généraux de la page Streamlit.

    :returns: ``None``.
    :rtype: None
    """
    st.set_page_config(
        page_title="Effort culinaire & Popularité",
        page_icon="assets/logo_MTM.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _prepare_analysis_data() -> tuple[pd.DataFrame, str]:
    """
    Charge les données d'analyse réelles ou bascule sur un échantillon simulé.

    :returns: Tuple ``(dataset, origine)`` où ``origine`` vaut ``"réelles"`` ou ``"simulées"``.
    :rtype: tuple[pd.DataFrame, str]
    """
    has_real_data, build_analysis_dataset = _import_analysis_modules()
    if has_real_data and build_analysis_dataset:
        _, _, analysis_df, success = load_real_datasets(build_analysis_dataset)
        if success and analysis_df is not None and not analysis_df.empty:
            return analysis_df, "réelles"

    fallback_df = generate_sample_data()
    return fallback_df, "simulées"


def _render_sidebar(data_origin: str, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Affiche les filtres latéraux et retourne le dataset filtré.

    :param data_origin: Provenance des données (``"réelles"`` ou ``"simulées"``).
    :type data_origin: str
    :param dataset: Jeu de données initial à filtrer.
    :type dataset: pd.DataFrame
    :returns: DataFrame filtré selon les choix de l'utilisateur.
    :rtype: pd.DataFrame
    """
    # Charger l’image logo
    logo = Image.open("assets/logo_MTM.png")
    st.sidebar.image(logo)         # ajuste la taille de l'image
    
    st.sidebar.header("Filtres")
    filtered = dataset.copy()
    options = infer_filter_options(dataset)

    if "minutes" in dataset.columns:
        minutes_series = dataset["minutes"].dropna()
        if not minutes_series.empty:
            min_minutes = int(np.floor(minutes_series.quantile(0.01)))
            max_minutes = int(np.ceil(minutes_series.quantile(0.99)))
            if min_minutes < max_minutes:
                default_high = int(np.ceil(minutes_series.quantile(0.75)))
                default_range = (
                    min_minutes,
                    min(max_minutes, max(min_minutes + 1, default_high)),
                )
                step_minutes = max(1, int((max_minutes - min_minutes) // 12))
                step_minutes = min(step_minutes, max_minutes - min_minutes)
                step_minutes = max(1, step_minutes)
                selected_minutes = st.sidebar.slider(
                    "Temps de préparation (minutes)",
                    min_value=min_minutes,
                    max_value=max_minutes,
                    value=default_range,
                    step=step_minutes,
                )
                filtered = filtered[
                    (filtered["minutes"] >= selected_minutes[0])
                    & (filtered["minutes"] <= selected_minutes[1])
                ]

    if options.age_range and options.age_range[0] < options.age_range[1]:
        selected_age = st.sidebar.slider(
            "Âge des recettes (mois) à partir d'aujourd'hui",
            min_value=float(options.age_range[0]),
            max_value=float(options.age_range[1]),
            value=options.age_range,
            step=1.0,
        )
        filtered = filtered[
            (filtered["age_months"] >= selected_age[0])
            & (filtered["age_months"] <= selected_age[1])
        ]

    if options.interactions_range and options.interactions_range[0] < options.interactions_range[1]:
        min_inter, max_inter = options.interactions_range
        threshold = st.sidebar.slider(
            "Interactions minimales",
            min_value=int(min_inter),
            max_value=int(max_inter),
            value=int(min_inter),
            step=1,
        )
        filtered = filtered[filtered["n_interactions"] >= threshold]

    if options.effort_categories:
        selected_categories = st.sidebar.multiselect(
            "Catégories d'effort",
            options=options.effort_categories,
            default=options.effort_categories,
        )
        if selected_categories:
            filtered = filtered[filtered["effort_category"].isin(selected_categories)]
        else:
            filtered = filtered.iloc[0:0]

    if "bayes_mean" in filtered.columns:
        bayes_min = float(filtered["bayes_mean"].min())
        bayes_max = float(filtered["bayes_mean"].max())
        if bayes_max > bayes_min:
            min_rating = st.sidebar.slider(
                "Note bayésienne minimale",
                min_value=bayes_min,
                max_value=bayes_max,
                value=bayes_min,
                step=0.1,
            )
            filtered = filtered[filtered["bayes_mean"] >= min_rating]

    return filtered


def _is_valid_number(value: float | None) -> bool:
    """
    Vérifie qu'une valeur numérique est définie et non ``NaN``.

    :param value: Valeur à tester.
    :type value: float | None
    :returns: ``True`` si la valeur est exploitable, ``False`` sinon.
    :rtype: bool
    """
    return value is not None and not np.isnan(value)


def _compute_story_indicators(data: pd.DataFrame, quartile_pattern) -> dict:
    """
    Calcule les indicateurs chiffrés affichés en tête de page.

    :param data: Dataset filtré actuellement visible.
    :type data: pd.DataFrame
    :param quartile_pattern: Résultat du calcul de pattern en quartiles (ou ``None``).
    :type quartile_pattern: webapp_utils.QuartilePatternData | None
    :returns: Dictionnaire des indicateurs (recettes, efforts, corrélations, etc.).
    :rtype: dict
    """
    indicators: dict = {"recipes": len(data)}

    indicators["avg_popularity"] = (
        float(data["bayes_mean"].mean()) if "bayes_mean" in data.columns else np.nan
    )
    indicators["avg_effort"] = (
        float(data["effort_score"].mean()) if "effort_score" in data.columns else np.nan
    )

    if "effort_category" in data.columns:
        distribution = data["effort_category"].value_counts(normalize=True)
        indicators["share_very_easy"] = float(distribution.get("Très Facile", np.nan))
        indicators["share_very_hard"] = float(distribution.get("Très Difficile", np.nan))
    else:
        indicators["share_very_easy"] = np.nan
        indicators["share_very_hard"] = np.nan

    if {"effort_score", "bayes_mean"}.issubset(data.columns):
        subset = data[["effort_score", "bayes_mean"]].dropna()
        if len(subset) > 2:
            indicators["spearman"] = float(
                subset["effort_score"].corr(subset["bayes_mean"], method="spearman")
            )
        else:
            indicators["spearman"] = np.nan
    else:
        indicators["spearman"] = np.nan

    if quartile_pattern and not quartile_pattern.means.empty:
        amplitude = float(quartile_pattern.means.max() - quartile_pattern.means.min())
        indicators["quartile_amplitude"] = amplitude
        indicators["quartile_means"] = quartile_pattern.means
    else:
        indicators["quartile_amplitude"] = np.nan
        indicators["quartile_means"] = None

    return indicators


def _render_metrics(indicators: dict, total_count: int | None = None) -> None:
    """
    Affiche les indicateurs clés sous forme de compteurs Streamlit.

    :param indicators: Indicateurs calculés via ``_compute_story_indicators``.
    :type indicators: dict
    :param total_count: Taille totale du dataset avant filtrage.
    :type total_count: int | None
    :returns: ``None``.
    :rtype: None
    """
    col1, col2, col3 = st.columns(3)

    col1.metric("Recettes analysées", f"{indicators['recipes']:,}")

    if _is_valid_number(indicators["avg_popularity"]):
        col2.metric(
            "Popularité moyenne",
            f"{indicators['avg_popularity']:.2f} / 5",
        )
    else:
        col2.metric("Popularité moyenne", "—")

    if _is_valid_number(indicators["avg_effort"]):
        col3.metric(
            "Effort moyen",
            f"{indicators['avg_effort']:.0f} / 100",
        )
    else:
        col3.metric("Effort moyen", "—")

    if total_count is not None and total_count != indicators["recipes"]:
        st.caption(
            f"Filtres actifs : {indicators['recipes']:,} recettes visibles sur "
            f"{total_count:,}."
        )

    if all(
        _is_valid_number(indicators[key]) for key in ("share_very_easy", "share_very_hard")
    ):
        st.caption(
            f"Répartition : {indicators['share_very_easy']:.0%} très faciles · "
            f"{indicators['share_very_hard']:.0%} très difficiles."
        )


def _render_patterns(pattern, quartile_pattern) -> None:
    """
    Affiche les visualisations principales du storytelling.

    :param pattern: Données agrégées par catégorie d'effort (ou ``None``).
    :type pattern: webapp_utils.EffortPatternData | None
    :param quartile_pattern: Données agrégées par quartile d'effort (ou ``None``).
    :type quartile_pattern: webapp_utils.QuartilePatternData | None
    :returns: ``None``.
    :rtype: None
    """
    if pattern is None and quartile_pattern is None:
        st.info(
            "Les colonnes nécessaires aux visualisations (effort_category, bayes_mean) "
            "sont absentes du dataset courant."
        )
        return

    if pattern is not None:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pattern.categories,
                y=pattern.expected,
                mode="lines+markers",
                name="Intuition : plus d'effort, moins de popularité",
                line=dict(color="#A0AEC0", dash="dash"),
                marker=dict(size=8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pattern.categories,
                y=pattern.observed,
                mode="lines+markers",
                name="Données observées",
                line=dict(color="#EF553B", width=3),
                marker=dict(size=9),
            )
        )
        fig.update_layout(
            title="Popularité moyenne par niveau d'effort perçu",
            yaxis_title="Note moyenne (bayes_mean)",
            xaxis_title="Niveau d'effort",
            height=420,
            legend_orientation="h",
            template="plotly_white",
        )
        _plotly_display(fig, width="stretch")

        # Encart d'observations figées pour le graphique d'effort
        with st.container():
            st.markdown(
                """
                <div style="
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 4px solid #FF6B6B;
                    margin: 20px 0;
                ">
                    <h5 style="color: #2c3e50; margin-top: 0;">
                        Contexte - Graphique d'effort
                    </h5>
                    <p style="color: #34495e; line-height: 1.6;">
                        Ce graphique compare la
                        <strong>popularité moyenne des recettes</strong> (axe vertical)
                        en fonction du <strong>niveau d'effort perçu</strong>
                        (axe horizontal).
                    </p>
                    <p style="color: #34495e; line-height: 1.6;">
                        Deux lignes sont présentées :
                    </p>
                    <ul style="color: #34495e; line-height: 1.6;">
                        <li>
                            <strong>Ligne grise pointillée</strong> : une intuition
                            courante — <em>plus une recette demande d'effort, moins
                            elle est populaire</em>.
                        </li>
                        <li>
                            <strong>Ligne rouge</strong> :
                            les <strong>données observées</strong>.
                        </li>
                    </ul>
                    <h5 style="color: #2c3e50; margin-top: 15px;">Observations</h5>
                    <ul style="color: #34495e; line-height: 1.6;">
                        <li>
                            <strong>Stabilité inattendue :</strong>
                            contrairement à l'intuition, la popularité observée reste
                            <strong>stable</strong> quel que soit le niveau de
                            difficulté perçue.
                        </li>
                        <li>
                            <strong>Niveau de popularité :</strong>
                            les notes moyennes sont proches, y compris pour les
                            recettes jugées « très difficiles ».
                        </li>
                        <li>
                            <strong>Pas de pénalité pour la complexité :</strong>
                            les recettes exigeantes ne sont
                            <strong>pas moins appréciées</strong> que les recettes
                            faciles.
                        </li>
                    </ul>
                    <h5 style="color: #2c3e50; margin-top: 15px;">Interprétations</h5>
                    <p style="color: #34495e; line-height: 1.6;">
                        Il semblerait que la
                        <strong>perception de l'effort</strong> n'influence pas
                        directement la popularité des recettes — du moins
                        <strong>pas de manière linéaire ou négative</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


    if quartile_pattern is not None and not quartile_pattern.means.empty:
        fig_quartile = go.Figure()
        fig_quartile.add_trace(
            go.Scatter(
                x=list(quartile_pattern.labels),
                y=list(quartile_pattern.means.values),
                mode="lines+markers",
                line=dict(color="#636EFA", width=3),
                marker=dict(size=9),
                name="Popularité moyenne",
            )
        )
        fig_quartile.update_layout(
            title="Pattern en U : popularité par quartile d'effort",
            yaxis_title="Note bayésienne",
            xaxis_title="Quartile d'effort",
            height=380,
            showlegend=False,
            template="plotly_white",
        )
        _plotly_display(fig_quartile, width="stretch")

        # Encart d'observations figées pour le pattern en U
        with st.container():
            st.markdown(
                """
                <div style="
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 4px solid #FF6B6B;
                    margin: 20px 0;
                ">
                    <h5 style="color: #2c3e50; margin-top: 15px;">
                        Contexte - Pattern en U
                    </h5>
                    <ul style="color: #34495e; line-height: 1.6;">
                        <li>
                            L’axe horizontal découpe les recettes en 4 groupes d’effort
                            (quartiles) : Q1 = les plus simples … Q4 = les plus
                            complexes.
                        </li>
                        <li>
                            L’axe vertical indique la note moyenne de popularité des
                            recettes.
                        </li>
                    </ul>
                    <h5 style="color: #2c3e50; margin-top: 15px;">Observations</h5>
                    <ul style="color: #34495e; line-height: 1.6;">
                        <li>
                            La courbe forme un petit “U” : les recettes très simples
                            (Q1) et très complexes (Q4) ont une popularité légèrement
                            plus élevée que celles d’effort moyen (Q2–Q3).
                        </li>
                        <li>
                            L’écart est minuscule entre le minimum (Q3) et le maximum
                            (Q1/Q4). Autrement dit, toutes les catégories ont
                            quasiment la même note.
                        </li>
                    </ul>
                    <h5 style="color: #2c3e50; margin-top: 15px;">Interprétations</h5>
                    <ul style="color: #34495e; line-height: 1.6;">
                        <li>
                            On peut parler d’un “pattern en U” (statistiquement
                            détectable), mais l’amplitude est tellement faible qu’elle
                            est sans impact concret pour l'utilisateur.
                        </li>
                        <p>
                            Ainsi, même si la forme en U existe, la popularité reste
                            quasi identique quel que soit le niveau d’effort.
                        </p>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )



def _render_methodology_section() -> None:
    """
    Présente les encarts méthodologiques et contextuels dans des volets repliables.

    :returns: ``None``.
    :rtype: None
    """
    with st.expander("Comment avons-nous préparé les données ?", expanded=False):
        st.markdown(
            "- Nettoyage des temps extrêmes : suppression des recettes < 1 minute "
            "ou > 6 heures, transformation logarithmique sur `minutes`.\n"
            "- Construction d'un score d'effort combinant durée, étapes, ingrédients "
            "et complexité textuelle (`avg_words_per_step`).\n"
            "- Calcul d'indicateurs de popularité robustes (`bayes_mean`, "
            "`wilson_lb`, interactions mensuelles) avec transformations log / winsorisation.\n"
            "- Stratification par quartiles d'effort pour tester statistiquement les "
            "différences (ANOVA + Kruskal-Wallis)."
        )
    with st.expander("Liens avec le cahier des charges", expanded=False):
        st.markdown(
            "- Storytelling clair et accessible pour le grand public.\n"
            "- Visualisations dynamiques (Plotly) et widgets interactifs Streamlit.\n"
            "- Mise en avant des hypothèses, résultats et limites directement dans "
            "l'interface.\n"
            "- Possibilité de filtrer les données selon l'effort, les interactions et les "
            "notes pour s'approprier l'analyse."
        )


def _render_correlation_matrix(data: pd.DataFrame) -> None:
    """
    Affiche la matrice de corrélation sur un sous-ensemble de variables cibles.

    :param data: Dataset filtré contenant les colonnes numériques.
    :type data: pd.DataFrame
    :returns: ``None``.
    :rtype: None
    """
    target_vars = [
        "log_minutes",
        "n_steps",
        "n_ingredients",
        "effort_score",
        "bayes_mean",
        "wilson_lb",
    ]
    numeric_df = data.select_dtypes(include=[np.number])
    available_vars = [col for col in target_vars if col in numeric_df.columns]

    if len(available_vars) < 2:
        st.info(
            "La matrice de corrélation nécessite au moins deux variables parmi : "
            "`log_minutes`, `n_steps`, `n_ingredients`, `effort_score`, "
            "`bayes_mean`, `wilson_lb`."
        )
        return

    numeric_df = numeric_df[available_vars].dropna(how="all", axis=1)

    if numeric_df.shape[1] < 2:
        st.info(
            "Après nettoyage des colonnes vides, il ne reste pas assez de variables pour "
            "calculer une matrice de corrélation."
        )
        return

    method_map = {
        "Spearman": "spearman",
        "Pearson": "pearson",
        "Kendall": "kendall",
    }
    method_choice = st.radio(
        "Méthode de calcul",
        options=list(method_map.keys()),
        index=0,
        horizontal=True,
        key="corr_method",
    )
    st.caption(
        """
        **Pearson** permet de mesurer la force d’une **relation linéaire entre
        deux variables numériques**. Il est conseillé de l’appliquer sur des
        **valeurs brutes**, qui contiennent **peu d’outliers** ou de fortes
        asymétries.

        **Spearman** permet de mesurer la force d’une **relation monotone
        croissante ou décroissante**, même si elle n’est pas linéaire. La méthode
        de calcul de Spearman est plus **robuste aux outliers** et **adaptée aux
        distributions non normales** notamment.

        **Kendall** permet d’évaluer la monotonicité, mais à partir de la
        **proportion de paires concordantes/discordantes**. Cette méthode de
        calcul est **très robuste sur petits échantillons** et en **présence de
        nombreux ex-æquo**.
        """
    )
    corr = numeric_df.corr(method=method_map[method_choice]).replace([np.inf, -np.inf], np.nan)
    corr = corr.fillna(0.0)

    heatmap_height = min(1200, 120 + 32 * corr.shape[0])
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=f"Matrice de corrélation ({method_choice})",
    )
    fig.update_layout(
        height=heatmap_height,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=0, t=80, b=0),
    )
    _plotly_display(fig, width="stretch")

    with st.container():
        st.markdown(
            """
            <div style="
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #FF6B6B;
                margin: 20px 0;
            ">
                <h4 style="color: #2c3e50; margin-top: 0;">
                    Contexte - Matrice de corrélation
                </h4>
                <ul style="color: #34495e; line-height: 1.6;">
                    <p>
                        Chaque case mesure à quel point deux variables évoluent
                        ensemble (de –1 à +1).
                    </p>
                    <li>
                        <strong>+1</strong> : elles montent/descendent
                        <strong>ensemble</strong> (corrélation positive parfaite).
                    </li>
                    <li>
                        <strong>0</strong> :
                        <strong>aucun lien linéaire</strong> détecté.
                    </li>
                    <li>
                        <strong>–1</strong> : quand l’une monte, l’autre
                        <strong>baisse</strong> (corrélation négative parfaite).
                    </li>
                    <p>
                        La <strong>couleur</strong> et le <strong>nombre</strong>
                        indiquent la force du lien.
                    </p>
                </ul>
                <h4 style="color: #2c3e50; margin-top: 0;">
                    Observations avec interprétations
                </h4>
                <ul style="color: #34495e; line-height: 1.6;">
                    <li>
                        <strong>Les indicateurs d’effort</strong>
                        <p>
                            <code>log_minutes</code> (durée) ↔
                            <code>effort_score</code> → très forte association.<br>
                            <code>n_ingredients</code> ↔ <code>effort_score</code> →
                            forte association.<br>
                            <code>log_minutes</code> ↔ <code>n_ingredients</code> →
                            association modérée.<br>
                            Une déduction possible serait que plus il y a
                            d’ingrédients et plus c’est long, plus le
                            <strong>score d’effort</strong> est élevé (cohérent).
                            Mais le coefficient de corrélation reste trop faible
                            pour valider cette déduction.
                        </p>
                    </li>
                </ul>
                <ul style="color: #34495e; line-height: 1.6;">
                    <li>
                        <strong>La popularité et l’effort</strong>
                        <p>
                            <code>bayes_mean</code> (note moyenne “robuste”) et
                            <code>wilson_lb</code> (borne de confiance) ont des
                            corrélations <strong>qui tendent vers 0</strong> avec
                            <code>effort_score</code>, <code>log_minutes</code> et
                            <code>n_ingredients</code>.<br>
                            <strong>
                                Il n'y a donc pas de corrélation entre l'effort et
                                la popularité des recettes.
                            </strong>
                        </p>
                    </li>
                </ul>
                <ul style="color: #34495e; line-height: 1.6;">
                    <li>
                        <strong>Activité des utilisateurs et note donnée</strong>
                        <p>
                            <code>n_interactions</code> (nombre d’avis/notations) ↔
                            <code>wilson_lb</code> (modéré).<br>
                            Plus il y a d’interactions, plus l’estimation de la
                            popularité est <strong>faiblement</strong>
                            meilleure/plus fiable.<br>
                            <em>
                                Cependant les notes données par les utilisateurs
                                étant majoritairement hautes, nos interprétations
                                sont donc biaisées.
                            </em>
                        </p>
                    </li>
                </ul>
                <ul style="color: #34495e; line-height: 1.6;">
                    <li>
                        <strong>Texte et longueur des étapes</strong>
                        <p>
                            <code>avg_words_per_step</code> a des corrélations
                            <strong>faibles</strong> avec le reste des variables
                            (proches de zéro).<br>
                            La <strong>verbosité</strong> des instructions
                            n’explique donc pas la popularité.
                        </p>
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

def _render_explorer(data: pd.DataFrame) -> None:
    """
    Affiche la section d'exploration libre effort/popularité.

    :param data: Dataset filtré courant.
    :type data: pd.DataFrame
    :returns: ``None``.
    :rtype: None
    """
    if data.empty:
        st.info("Aucune donnée disponible avec les filtres actuels.")
        return

    st.write(
        "Choisissez vos axes pour visualiser la relation effort/popularité et ajustez "
        "les distributions en direct."
    )

    effort_candidates = [
        "effort_score",
        "log_minutes",
        "n_steps",
        "n_ingredients",
        "avg_words_per_step",
    ]
    popularity_candidates = [
        "bayes_mean",
        "wilson_lb",
        "avg_rating",
        "log1p_interactions_per_month_w",
        "interactions_per_month",
        "n_interactions",
    ]

    available_effort = [col for col in effort_candidates if col in data.columns]
    available_popularity = [col for col in popularity_candidates if col in data.columns]

    if not available_effort or not available_popularity:
        st.info("Les colonnes nécessaires à l'exploration ne sont pas présentes.")
        return

    col_controls = st.columns(3)
    effort_axis = col_controls[0].selectbox(
        "Axe effort", available_effort, index=0, key="explorer_effort"
    )
    popularity_axis = col_controls[1].selectbox(
        "Axe popularité", available_popularity, index=0, key="explorer_popularity"
    )

    color_options = ["Aucune"]
    color_options.extend(
        [col for col in ("effort_category", "effort_quartile") if col in data.columns]
    )
    color_choice = col_controls[2].selectbox(
        "Colorer par", color_options, index=0, key="explorer_color"
    )

    sample_data = data.sample(min(5000, len(data)), random_state=42) if len(data) > 5000 else data
    fig = px.scatter(
        sample_data,
        x=effort_axis,
        y=popularity_axis,
        color=color_choice if color_choice != "Aucune" else None,
        size="n_interactions" if "n_interactions" in data.columns else None,
        hover_data=[col for col in ["id", "minutes", "n_ingredients"] if col in data.columns],
        opacity=0.7,
        template="plotly_white",
        title="Explorer librement effort et popularité",
    )

    if st.checkbox("Afficher la tendance linéaire", value=True, key="explorer_trend"):
        regression = fit_simple_regression(sample_data, effort_axis, popularity_axis)
        if regression:
            fig.add_trace(
                go.Scatter(
                    x=regression.x_curve,
                    y=regression.y_curve,
                    mode="lines",
                    name="Régression linéaire",
                    line=dict(color="#EB6F92", width=2),
                )
            )
            fig.add_annotation(
                x=0.99,
                y=0.02,
                xref="paper",
                yref="paper",
                text=f"R² = {regression.r_squared:.3f}",
                showarrow=False,
                font=dict(color="#EB6F92"),
            )

    _plotly_display(fig, width="stretch")

    # Encart d'descriptif
    with st.container():
        st.markdown(
            """
            <div style="
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #FF6B6B;
                margin: 20px 0;
            ">
                <ul style="color: #34495e; line-height: 1.6;">
                    <p>
                        Exploration utile pour visualiser la régression linéaire
                        entre deux variables.
                    </p>
                    <p>
                        Une régression linéaire est une méthode statistique qui
                        modélise la relation entre une variable cible y et une ou
                        plusieurs variables explicatives x (le nuage de points),
                        par une droite (ici rouge).
                    </p>
                    <p>
                        Le fait d'observer une droite quasiment horizontale permet
                        de comprendre qu'il n'y a aucun effet linéaire de x sur y.
                        En d'autres termes, les variations de x n'impliquent pas
                        de variations sur y.
                    </p>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    
    hist_var = st.selectbox(
        "Distribution à explorer",
        available_effort + available_popularity,
        index=0,
        key="hist_var",
    )
    hist_fig = compute_histogram_figure(data[hist_var], var_label=hist_var)
    _plotly_display(hist_fig, width="stretch")

    # Encart d'descriptif
    with st.container():
        st.markdown(
            """
            <div style="
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #FF6B6B;
                margin: 20px 0;
            ">
                <ul style="color: #34495e; line-height: 1.6;">
                    <p>
                        Exploration utile pour visualiser la distribution des
                        valeurs. Pour certaines valeurs, la distribution est
                        difficilement interprétable.
                    </p>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "effort_category" in data.columns:
            effort_order = ["Très Facile", "Facile", "Modéré", "Difficile", "Très Difficile"]
            cat_counts = (
                data["effort_category"]
                .value_counts(normalize=True)
                .rename_axis("Effort")
                .reset_index(name="Part")
            )
            order_mapping = {label: idx for idx, label in enumerate(effort_order)}
            cat_counts = cat_counts.sort_values(
                by="Effort",
                key=lambda series: series.map(order_mapping).fillna(len(order_mapping)),
            )
            fig_categories = px.bar(
                cat_counts,
                x="Effort",
                y="Part",
                text="Part",
                title="Répartition des recettes par niveau d'effort",
                template="plotly_white",
            )
            fig_categories.update_traces(texttemplate="%{text:.0%}", textposition="outside")
            fig_categories.update_yaxes(tickformat=".0%")
            _plotly_display(fig_categories, width="stretch")

    # Encart d'descriptif
    with st.container():
        st.markdown(
            """
            <div style="
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #FF6B6B;
                margin: 20px 0;
            ">
                <ul style="color: #34495e; line-height: 1.6;">
                    <p>
                        Exploration utile pour comprendre la répartition des
                        recettes par niveau d'effort. Cette répartition n'est
                        pas homogène.
                    </p>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_scenario_planner(data: pd.DataFrame) -> None:
    """
    Propose un assistant interactif pour identifier des recettes respectant des contraintes.

    :param data: Dataset filtré contenant au minimum ``minutes``, ``n_ingredients`` et ``bayes_mean``.
    :type data: pd.DataFrame
    :returns: ``None``.
    :rtype: None
    """
    required_cols = {"minutes", "n_ingredients", "bayes_mean"}
    st.write(
        "Définissez vos contraintes et découvrez les recettes qui tiennent la promesse "
        "popularité vs effort."
    )

    if not required_cols.issubset(data.columns):
        st.info(
            "Certaines colonnes indispensables à cet assistant ne sont pas disponibles "
            "dans le dataset courant."
        )
        return

    minutes_series = data["minutes"].dropna()
    ingredients_series = data["n_ingredients"].dropna()
    if minutes_series.empty or ingredients_series.empty:
        st.info("Impossible d'estimer le temps ou les ingrédients avec les filtres actuels.")
        return

    max_minutes = int(np.ceil(minutes_series.quantile(0.95)))
    max_minutes = max(max_minutes, 15)
    max_ingredients = int(np.ceil(ingredients_series.quantile(0.95)))
    max_ingredients = max(max_ingredients, 5)

    col1, col2, col3 = st.columns(3)
    minutes_cap = col1.slider(
        "Temps maximum disponible (minutes)",
        min_value=0,
        max_value=max_minutes,
        value=min(45, max_minutes),
        step=5,
        key="scenario_minutes",
    )
    ingredient_cap = col2.slider(
        "Nombre max d'ingrédients",
        min_value=1,
        max_value=max_ingredients,
        value=min(10, max_ingredients),
        step=1,
        key="scenario_ingredients",
    )
    min_rating = col3.slider(
        "Note minimale souhaitée",
        min_value=1.0,
        max_value=5.0,
        value=4.0,
        step=0.1,
        key="scenario_rating",
    )

    effort_pref = "Peu importe"
    if "effort_category" in data.columns:
        effort_options = ["Peu importe"] + sorted(data["effort_category"].dropna().unique())
        effort_pref = st.radio(
            "Niveau d'effort recherché",
            options=effort_options,
            horizontal=True,
            key="scenario_effort",
        )

    scenario_df = data.copy()
    scenario_df = scenario_df[scenario_df["minutes"] <= minutes_cap]
    scenario_df = scenario_df[scenario_df["n_ingredients"] <= ingredient_cap]
    scenario_df = scenario_df[scenario_df["bayes_mean"] >= min_rating]

    if effort_pref != "Peu importe" and "effort_category" in scenario_df.columns:
        scenario_df = scenario_df[scenario_df["effort_category"] == effort_pref]

    st.markdown("---")

    if scenario_df.empty:
        st.warning(
            "Aucune recette ne correspond à ces critères. Desserrez légèrement les curseurs."
        )
        return

    scenario_mean_effort = (
        scenario_df["effort_score"].mean() if "effort_score" in scenario_df.columns else np.nan
    )

    metrics_cols = st.columns(3)
    metrics_cols[0].metric("Recettes compatibles", f"{len(scenario_df):,}")
    metrics_cols[1].metric("Note moyenne", f"{scenario_df['bayes_mean'].mean():.2f} / 5")
    if _is_valid_number(scenario_mean_effort):
        metrics_cols[2].metric("Effort moyen", f"{scenario_mean_effort:.0f} / 100")
    elif "n_steps" in scenario_df.columns:
        metrics_cols[2].metric("Étapes moyennes", f"{scenario_df['n_steps'].mean():.1f}")
    else:
        metrics_cols[2].metric("Effort moyen", "—")

    display_cols = [
        col
        for col in (
            "id",
            "minutes",
            "n_ingredients",
            "effort_score",
            "effort_category",
            "bayes_mean",
            "wilson_lb",
            "n_interactions",
        )
        if col in scenario_df.columns
    ]
    top_candidates = (
        scenario_df.sort_values(["bayes_mean", "n_interactions"], ascending=[False, False])
        .head(5)
        .reset_index(drop=True)
    )
    st.dataframe(top_candidates[display_cols], width="stretch")

    if "effort_category" in scenario_df.columns:
        category_view = (
            scenario_df["effort_category"]
            .value_counts(normalize=True)
            .rename_axis("Effort")
            .reset_index(name="Part")
        )
        fig_reco = px.bar(
            category_view,
            x="Effort",
            y="Part",
            text="Part",
            title="Répartition effort des suggestions retenues",
            template="plotly_white",
        )
        fig_reco.update_traces(texttemplate="%{text:.0%}", textposition="outside")
        fig_reco.update_yaxes(tickformat=".0%")
        _plotly_display(fig_reco, width="stretch")

    st.caption(
        "Astuce : augmentez légèrement la note minimale pour des recettes premium, "
        "ou détendez le temps pour trouver plus d'options."
    )


def display_about_tab() -> None:
    """
    Affiche le contenu de l'onglet "À propos".

    :returns: ``None``.
    :rtype: None

    .. note::
       Charge le fichier ``rapport_analyse_effort_popularite.md`` ou affiche des informations par défaut.
    """
    logger.debug("Chargement de l'onglet À propos")

    rapport_path = Path(__file__).resolve().parents[1] / "docs" / "rapport_analyse_effort_popularite.md"
    images_dir = rapport_path.parent / "images"

    try:
        rapport_content = rapport_path.read_text(encoding="utf-8")

        def replace_local_images(match: re.Match[str]) -> str:
            """
            Remplace un lien d'image locale par une version encodée en base64.

            :param match: Résultat de la correspondance regex sur un lien d'image.
            :type match: re.Match[str]
            :returns: Lien Markdown mis à jour (ou original si l'image est introuvable).
            :rtype: str
            """
            img_path = images_dir / match.group(1)
            if img_path.exists():
                data = base64.b64encode(img_path.read_bytes()).decode()
                suffix = img_path.suffix.lower().lstrip(".")
                return f"![](data:image/{suffix};base64,{data})"
            logger.warning("Image introuvable : %s", img_path)
            return match.group(0)

        pattern = r"!\[[^\]]*\]\((?:\.\/)?images\/([^)]+)\)"
        rapport_content = re.sub(pattern, replace_local_images, rapport_content)

        st.markdown(rapport_content, unsafe_allow_html=True)
        logger.debug("rapport_analyse_effort_popularite chargé avec succès")

    except FileNotFoundError:
        logger.warning("Fichier rapport_analyse_effort_popularite.md non trouvé")
        st.error("Impossible de charger `docs/rapport_analyse_effort_popularite.md`.")
        st.markdown(
            """
            ## À propos du projet
            Cette application raconte l'analyse de la relation entre effort culinaire et popularité.
            Consultez le dépôt GitHub pour accéder au rapport détaillé et aux scripts d'analyse.
            """
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Erreur lors du chargement du rapport_analyse_effort_popularite : %s", exc)
        st.error(f"Erreur : {exc}")


def render_storytelling(data: pd.DataFrame, data_origin: str, total_recipes: int) -> None:
    """
    Orchestre le storytelling principal destiné à un public non spécialiste.

    :param data: Jeu de données filtré actuel.
    :type data: pd.DataFrame
    :param data_origin: Indique si les données sont ``"réelles"`` ou ``"simulées"``.
    :type data_origin: str
    :param total_recipes: Nombre total de recettes disponibles avant filtrage.
    :type total_recipes: int
    :returns: ``None``.
    :rtype: None
    """
    if data.empty:
        st.warning(
            "Aucune recette ne correspond aux filtres sélectionnés dans la barre latérale."
        )
        return

    pattern = compute_effort_pattern(data)
    quartile_pattern = compute_quartile_pattern(data)
    indicators = _compute_story_indicators(data, quartile_pattern)

    st.title("En quoi l'effort culinaire influence-t-il la popularité des recettes ?")
    st.write(
        "On part de l'intuition qu'une recette très exigeante décourage les cuisiniers, clients du site web. "
        "Voyons si l'analyse des données confirment ce ressenti, et explorons les à l'aide "
        "de divers filtres."
    )

    story_tab, explorer_tab, scenario_tab, about_tab = st.tabs(
        ["Storytelling", "Explorer les données", "Trouver ma recette idéale", "À propos"]
    )

    with story_tab:
        st.subheader("1. Vue d'ensemble de l'étude")
        st.write(
            "Chaque recette combine un score d'effort (temps, étapes, ingrédients, complexité) "
            "et des indicateurs de popularité (notes, confiance, interactions). "
            "Les compteurs ci-dessous reflètent la sélection actuelle."
        )
        _render_metrics(indicators, total_recipes)

        st.subheader("2. Intuition vs réalité")

        _render_patterns(pattern, quartile_pattern)


        st.write(
            "Valeurs précises des affichages"
        )
        if _is_valid_number(indicators["spearman"]):
            st.caption(
                f"Corrélation effort/popularité (Spearman) : {indicators['spearman']:.3f} "
                "(tend vers zéro : il n'y a donc pas de lien linéaire direct)."
            )

        if _is_valid_number(indicators["quartile_amplitude"]):
            st.caption(
                f"Écart maximal entre quartiles : {indicators['quartile_amplitude']:.3f} point "
                "sur une échelle de 5. La variation est faible ⇒ l'effort seul ne suffit pas à expliquer "
                "la popularité."
            )

        st.subheader("3. Matrice de corrélation complète")
        st.write(
            "Retrouvez ci-dessous toutes les corrélations entre variables numériques, pour "
            "repérer les couples qui évoluent de concert ou au contraire s'opposent."
        )
        _render_correlation_matrix(data)

        st.subheader("4. Notre regard critique")

        with st.container():
            st.markdown(
                """
                <div style="
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 4px solid #FF6B6B;
                    margin: 20px 0;
                ">
                    <ul style="color: #34495e; line-height: 1.6;">
                        <p>
                            Dans cette partie, nous complétons notre constat de
                            non-corrélation entre l’effort culinaire et la popularité
                            des recettes, par un regard critique quant aux limites
                            de cette étude, dont nous sommes conscients.
                        </p>
                        <p>Les différentes limites abordées :</p>
                    </ul>
                    <h5 style="color: #2c3e50; margin-top: 15px;">
                        1. Contexte de la plateforme food.com (GeniusKitchen)
                    </h5>
                    <ul style="color: #34495e; line-height: 1.6;">
                        <p>
                            Les interactions et les notes attribuées aux recettes
                            dépendent du comportement des utilisateurs et de la
                            construction de la plateforme elle-même. Dans notre analyse,
                            nous n’avons pas accès au contexte d’établissement des notes
                            et du référencement des recettes mis en place sur la
                            plateforme. Ainsi, nous constatons que l’uniformité des notes
                            et les biais d’engagement observés confortent le rejet de
                            notre hypothèse de départ.
                        </p>
                    </ul>
                    <h5 style="color: #2c3e50; margin-top: 15px;">
                        2. Méthodologies liées au choix des variables
                    </h5>
                    <ul style="color: #34495e; line-height: 1.6;">
                        <p>
                            Notre définition de popularité se base sur les variables
                            <code>bayes_mean</code> et <code>wilson_lb</code>, qui
                            reposent sur des modèles statistiques atténuant la variance
                            (sorte de lissage). Ainsi, des différences fines entre
                            recettes ont pu être masquées. À l’avenir, des variables
                            telles que le temps passé sur la page ou une pondération
                            accrue pour les utilisateurs très actifs pourraient mieux
                            contextualiser la définition de popularité.
                        </p>
                        <p>
                            Notre définition d’effort culinaire se base sur une
                            pondération arbitraire, définie par le groupe. Cependant,
                            elle constitue une hypothèse subjective quant au rapport
                            entre temps, complexité et ingrédients.
                        </p>
                    </ul>
                    <h5 style="color: #2c3e50; margin-top: 15px;">
                        3. Contexte du jeu de données
                    </h5>
                    <ul style="color: #34495e; line-height: 1.6;">
                        <p>
                            Comme étudié lors de nos TPs, le contexte dans lequel
                            les données sont extraites joue un rôle important, car
                            il influence leurs résultats (accessibilité du site,
                            facteurs de popularité supplémentaires tels que la mise
                            en forme personnalisée des recettes, etc.).
                        </p>
                    </ul>
                    <h5 style="color: #2c3e50; margin-top: 15px;">
                        Ouvertures
                    </h5>
                    <ul style="color: #34495e; line-height: 1.6;">
                        <p>
                            En conséquence des limites énoncées, nous souhaiterions
                            intégrer à notre future étude de nouvelles variables qui
                            traduiraient au mieux l’expérience utilisateur vis-à-vis
                            de la plateforme, afin de mieux contextualiser les données
                            extraites : temps passé sur la page, mise en avant de la
                            recette, etc. Nous pourrions également la compléter par
                            une analyse sémantique des commentaires (notamment en
                            étudiant la polarisation de ceux-ci), afin de consolider
                            notre définition de la popularité d’une recette.
                        </p>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            conclusions = [
                (
                    "Pour vous convaincre et vous approprier l'analyse, interagissez "
                    "avec les données dans le second onglet ! Notre rapport d'analyse"
                    "est présent dans le dernier onglet 'À propos'."
                )
            ]
            st.markdown("\n".join(f"- {item}" for item in conclusions))

        _render_methodology_section()

        st.info(
            f"Histoire construite à partir de données {data_origin}. "
            "Consultez le rapport détaillé (`docs/rapport_analyse_effort_popularite.md`) "
            "pour approfondir la méthodologie et les tests statistiques."
        )

    with explorer_tab:
        _render_explorer(data)

    with scenario_tab:
        _render_scenario_planner(data)

    with about_tab:
        display_about_tab()


def main() -> None:
    """
    Point d'entrée de la webapp Streamlit.

    :returns: ``None``.
    :rtype: None
    """
    _configure_streamlit()
    analysis_df, data_origin = _prepare_analysis_data()

    if analysis_df.empty:
        st.error("Aucune donnée disponible pour raconter l'histoire. Vérifiez le pipeline.")
        return

    filtered_df = _render_sidebar(data_origin, analysis_df)
    render_storytelling(filtered_df, data_origin, total_recipes=len(analysis_df))


if __name__ == "__main__":
    main()
