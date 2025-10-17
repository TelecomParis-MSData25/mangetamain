"""
Préparation des jeux de données Recipes / Interactions pour les analyses.

Ce module regroupe la logique appliquée dans les notebooks d'exploration afin
de produire un jeu de données unique, prêt pour les analyses univariées et
multivariées. Il expose des fonctions réutilisables et un point d'entrée
exécutable pour générer les fichiers nettoyés sur disque.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Détection de la racine du dépôt et chemins utiles
# ---------------------------------------------------------------------------


def _detect_repo_root() -> Path:
    """Retourne la racine du dépôt quel que soit le contexte d'exécution."""
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1]

    # Fallback pour un appel depuis un notebook
    try:
        import ipynbname  # type: ignore

        return ipynbname.path().parents[2]
    except Exception:
        return Path.cwd()


REPO_ROOT = _detect_repo_root()
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "dataset_analysis" / "data"


# ---------------------------------------------------------------------------
# Chargement des données brutes
# ---------------------------------------------------------------------------


def _ensure_raw_data_available() -> None:
    """Télécharge les fichiers Kaggle si nécessaire."""
    raw_files = ["RAW_recipes.csv", "RAW_interactions.csv"]
    if all((DATA_DIR / name).exists() for name in raw_files):
        return

    script_path = REPO_ROOT / "scripts" / "download_data.py"
    if not script_path.exists():
        raise FileNotFoundError(
            "Impossible de télécharger les données : scripts/download_data.py introuvable."
        )

    spec = importlib.util.spec_from_file_location("download_data", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None  # pour mypy
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    module.download_and_extract()  # type: ignore[attr-defined]


def load_raw_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les deux fichiers bruts Kaggle."""
    _ensure_raw_data_available()
    recipes = pd.read_csv(DATA_DIR / "RAW_recipes.csv")
    interactions = pd.read_csv(DATA_DIR / "RAW_interactions.csv")
    return recipes, interactions


# ---------------------------------------------------------------------------
# Pré-traitement des recettes
# ---------------------------------------------------------------------------


def _parse_steps(value: object) -> list[str]:
    """Transforme la colonne `steps` (str/list) en liste de chaînes."""
    if isinstance(value, list):
        return [str(step) for step in value]
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(step) for step in parsed]
        except (SyntaxError, ValueError):
            return []
    return []


def categorize_prep_time(minutes: float) -> str:
    if minutes < 30:
        return "Rapide"
    if minutes <= 90:
        return "Moyenne"
    return "Longue"


def categorize_complexity(n_steps: int) -> str:
    if n_steps <= 5:
        return "Simple"
    if n_steps <= 10:
        return "Modéré"
    if n_steps <= 20:
        return "Complexe"
    return "Très complexe"


def categorize_step_length(avg_words: float) -> str:
    if avg_words < 10:
        return "Étapes courtes"
    if avg_words < 20:
        return "Étapes moyennes"
    return "Étapes longues"


def categorize_n_ingredients(n_ingredients: int) -> str:
    if n_ingredients <= 5:
        return "Peu d'ingrédients"
    if n_ingredients <= 10:
        return "Ingrédients modérés"
    return "Beaucoup d'ingrédients"


def calculate_avg_words_per_step(steps_value: object) -> float:
    """Calcule la longueur moyenne (en mots) des étapes d'une recette."""
    steps = _parse_steps(steps_value)
    if not steps:
        return 0.0
    word_counts = [len(step.split()) for step in steps if step]
    return float(np.mean(word_counts)) if word_counts else 0.0


def _minmax_scale(series: pd.Series) -> pd.Series:
    """Ramène la série sur [0, 1] (retourne 0 si variance nulle)."""
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def prepare_recipes(recipes: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et enrichit le jeu de données des recettes."""
    recipes_clean = recipes.copy()

    recipes_clean["minutes"] = recipes_clean["minutes"].replace({0: np.nan})
    recipes_clean = recipes_clean.dropna(subset=["minutes"])

    # Filtrer les recettes improbables : > 6h, aucune étape, aucune instruction
    recipes_clean = recipes_clean[
        (recipes_clean["minutes"] > 0) & (recipes_clean["minutes"] <= 360)
    ]
    recipes_clean = recipes_clean[recipes_clean["n_steps"].fillna(0) > 0]
    recipes_clean = recipes_clean[recipes_clean["steps"].notna()]
    parsed_steps = recipes_clean["steps"].apply(_parse_steps)
    recipes_clean = recipes_clean[parsed_steps.map(len) > 0].copy()

    # Nettoyage des dates
    recipes_clean["submitted"] = pd.to_datetime(
        recipes_clean["submitted"], errors="coerce"
    )

    recipes_clean["log_minutes"] = np.log(recipes_clean["minutes"])
    recipes_clean["category_minutes"] = recipes_clean["minutes"].apply(
        categorize_prep_time
    )
    recipes_clean["avg_words_per_step"] = parsed_steps.loc[
        recipes_clean.index
    ].apply(calculate_avg_words_per_step)
    recipes_clean["step_length_category"] = recipes_clean["avg_words_per_step"].apply(
        categorize_step_length
    )
    recipes_clean["complexity"] = recipes_clean["n_steps"].apply(categorize_complexity)
    recipes_clean["category_n_ingredients"] = recipes_clean["n_ingredients"].apply(
        categorize_n_ingredients
    )
    recipes_clean["log_n_ingredients"] = np.log(
        recipes_clean["n_ingredients"].clip(lower=1)
    )

    # Calcul du score d'effort culinaire (normalisation + pondération)
    to_scale = {
        "log_minutes": 0.30,
        "n_steps": 0.25,
        "avg_words_per_step": 0.20,
        "log_n_ingredients": 0.25,
    }

    for column in to_scale:
        recipes_clean[f"{column}_scaled"] = _minmax_scale(recipes_clean[column])

    recipes_clean["effort_score"] = (
        sum(
            weight * recipes_clean[f"{column}_scaled"]
            for column, weight in to_scale.items()
        )
        * 100
    )

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

    recipes_clean["effort_category"] = recipes_clean["effort_score"].apply(
        categorize_effort
    )

    return recipes_clean


# ---------------------------------------------------------------------------
# Pré-traitement des interactions
# ---------------------------------------------------------------------------


def prepare_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Applique les traitements décrits dans le notebook sur les interactions."""
    interactions_clean = interactions.copy()
    interactions_clean["date"] = pd.to_datetime(
        interactions_clean["date"], errors="coerce"
    )

    mask_zero = interactions_clean["rating"] == 0
    interactions_clean.loc[mask_zero, "rating"] = pd.NA
    interactions_clean["review_length"] = interactions_clean["review"].fillna("").str.len()

    return interactions_clean


def _aggregate_recipe_metrics(interactions: pd.DataFrame) -> pd.DataFrame:
    """Calcule les métriques d'engagement et de satisfaction par recette."""
    recipe_metrics = (
        interactions.groupby("recipe_id")
        .agg(
            rating_count=("rating", "count"),
            avg_rating=("rating", "mean"),
            median_rating=("rating", "median"),
            std_rating=("rating", "std"),
            n_unique_users=("user_id", "nunique"),
            n_reviews_text=("review", lambda x: x.notna().sum()),
        )
        .rename(columns={"rating_count": "n_interactions"})
        .round(3)
    )

    # Moyenne bayésienne
    global_mean = interactions["rating"].dropna().mean()
    prior_count = 10
    recipe_metrics["bayes_mean"] = (
        (recipe_metrics["avg_rating"] * recipe_metrics["n_interactions"])
        + global_mean * prior_count
    ) / (recipe_metrics["n_interactions"] + prior_count)

    # Wilson Lower Bound
    def wilson_lower_bound(positives: int, total: int, confidence: float = 0.95) -> float:
        if total == 0:
            return 0.0
        z = 1.96 if confidence == 0.95 else 1.64
        p = positives / total
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
        return max(0.0, center - margin)

    wilson_scores: dict[int, float] = {}
    grouped_ratings = interactions.groupby("recipe_id")["rating"]
    for recipe_id, values in grouped_ratings:
        ratings = values.dropna()
        if ratings.empty:
            wilson_scores[recipe_id] = 0.0
            continue
        positives = (ratings >= 4).sum()
        total = len(ratings)
        wilson_scores[recipe_id] = wilson_lower_bound(positives, total)

    recipe_metrics["wilson_lb"] = recipe_metrics.index.map(wilson_scores)

    return recipe_metrics


# ---------------------------------------------------------------------------
# Fusion recettes / interactions et enrichissements complémentaires
# ---------------------------------------------------------------------------


def _compute_age_in_months(submitted: pd.Series) -> pd.Series:
    current_date = pd.Timestamp.now(tz=None)
    age_months = (current_date - submitted).dt.days / 30.44
    return age_months.clip(lower=0).round(1)


def winsorize(series: pd.Series, lower_pct: float = 0.05, upper_pct: float = 0.99) -> pd.Series:
    """Borne la série entre les quantiles choisis sans supprimer de lignes."""
    data = series.dropna()
    if data.empty:
        return series
    lower = data.quantile(lower_pct)
    upper = data.quantile(upper_pct)
    return series.clip(lower=lower, upper=upper)


def enrich_analysis_dataset(recipes: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les recettes et les métriques d'interaction et ajoute les variables dérivées."""
    recipe_metrics = _aggregate_recipe_metrics(interactions)

    df_analysis = recipes.merge(
        recipe_metrics, left_on="id", right_index=True, how="inner"
    )

    df_analysis["age_months"] = _compute_age_in_months(df_analysis["submitted"])
    df_analysis["interactions_per_month"] = df_analysis["n_interactions"] / np.maximum(
        1, df_analysis["age_months"]
    )

    df_analysis = df_analysis[df_analysis["minutes"] > 0]
    df_analysis["log_minutes"] = np.log(df_analysis["minutes"])

    engagement_vars = ["n_interactions", "interactions_per_month", "n_unique_users"]
    for var in engagement_vars:
        df_analysis[f"log1p_{var}"] = np.log1p(df_analysis[var].clip(lower=0))

    for var in ["log_minutes", "log1p_n_interactions", "log1p_interactions_per_month", "log1p_n_unique_users"]:
        df_analysis[f"{var}_w"] = winsorize(df_analysis[var])

    df_analysis["rating_gap"] = 5 - df_analysis["avg_rating"]
    df_analysis["bayes_gap"] = 5 - df_analysis["bayes_mean"]
    df_analysis["wilson_gap"] = 1 - df_analysis["wilson_lb"]

    return df_analysis


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def build_analysis_dataset(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Construit les différents DataFrames nécessaires à l'analyse.

    Returns
    -------
    recipes_clean : DataFrame
        Recettes nettoyées avec les variables d'effort culinaire.
    interactions_clean : DataFrame
        Interactions prêtes pour agrégation (notes à 0 passées en NA).
    df_analysis : DataFrame
        Jeu de données consolidé recettes + métriques d'engagement.
    """
    recipes_raw, interactions_raw = load_raw_datasets()
    recipes_clean = prepare_recipes(recipes_raw)
    interactions_clean = prepare_interactions(interactions_raw)

    # On ne garde que les interactions correspondant aux recettes retenues
    interactions_clean = interactions_clean[
        interactions_clean["recipe_id"].isin(recipes_clean["id"])
    ]

    df_analysis = enrich_analysis_dataset(recipes_clean, interactions_clean)

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        recipes_clean.to_csv(OUTPUT_DIR / "recipes_clean.csv", index=False)
        interactions_clean.to_csv(OUTPUT_DIR / "interactions_clean.csv", index=False)
        df_analysis.to_csv(OUTPUT_DIR / "analysis_dataset.csv", index=False)

    return recipes_clean, interactions_clean, df_analysis


if __name__ == "__main__":
    recipes_df, interactions_df, analysis_df = build_analysis_dataset(save=True)
    print(
        (
            f"{len(recipes_df):,} recettes nettoyées | "
            f"{len(interactions_df):,} interactions retenues | "
            f"{len(analysis_df):,} recettes dans le dataset d'analyse"
        )
    )
