"""
Tests unitaires et fonctionnels du module `dataset_analysis.dataset_preprocessing`.

Ce fichier couvre :
- Les fonctions pures de catégorisation et de parsing,
- Les transformations DataFrame (recettes, interactions),
- La fusion finale et les métriques,
- Un test mocké de `build_analysis_dataset` (sans dépendance avec des éléments extérieurs).

Les données sont simulées via fixtures ; aucun accès disque n’est requis.
"""

import pytest
import pandas as pd
import numpy as np

from dataset_analysis import dataset_preprocessing as dp


# ============================================================================
# Fixtures : petits extraits simulés de RAW_recipes et RAW_interactions
# ============================================================================

@pytest.fixture
def sample_recipes_df() -> pd.DataFrame:
    """Extrait minimal simulé de RAW_recipes.csv."""
    data = {
        "id": [112140, 109439, 39959],
        "minutes": [20, 100, 0],                # la 3e est invalide et sera vérifiée pendant les tests
        "submitted": ["2020-01-01", "2020-06-01", "2019-01-01"],
        "n_steps": [3, 8, 0],                   # la 3e est invalide et sera vérifiée pendant les tests
        "steps": ["['Mix and cut', 'Bake']", "['Prep and separate the ingredients', 'Cook']", None],
        "n_ingredients": [4, 9, 5],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_interactions_df() -> pd.DataFrame:
    """Extrait minimal simulé de RAW_interactions.csv."""
    data = {
        "user_id": [273745, 468945, 202555, 217118],
        "recipe_id": [85009, 134728, 200236, 225241],
        "date": ["2020-01-01", "2020-02-01", "2021-01-01", "2022-01-01"],
        "rating": [5, 0, 4, np.nan],            # un 0 doit devenir NA
        "review": ["MMMMM! This is so good!", "Pork + fruit + crockpot = awesome", None, "À refaire"],
    }
    return pd.DataFrame(data)


# ============================================================================
# Tests unitaires – fonctions pures et déterministes
# ============================================================================

def test_parse_steps_parses_correctly():
    """Vérifie que `_parse_steps` convertit correctement une chaîne de liste."""
    assert dp._parse_steps("['step1', 'step2']") == ["step1", "step2"]
    assert dp._parse_steps(["Mix", "Bake"]) == ["Mix", "Bake"]
    assert dp._parse_steps("invalid") == []
    assert dp._parse_steps(np.nan) == []


@pytest.mark.parametrize(
    "minutes,expected",
    [(10, "Rapide"), (60, "Moyenne"), (120, "Longue")],
)
def test_categorize_prep_time(minutes, expected):
    """Teste les catégories de temps de préparation."""
    assert dp.categorize_prep_time(minutes) == expected


@pytest.mark.parametrize(
    "n_steps,expected",
    [
        (0, "Simple"),
        (5, "Simple"),
        (6, "Modéré"),
        (10, "Modéré"),
        (11, "Complexe"),
        (20, "Complexe"),
        (21, "Très complexe"),
    ],
)
def test_categorize_complexity(n_steps, expected):
    """Vérifie les intervalles de complexité `categorize_complexity`."""
    assert dp.categorize_complexity(n_steps) == expected


@pytest.mark.parametrize(
    "avg_words,expected",
    [
        (0, "Étapes courtes"),
        (9.999, "Étapes courtes"),
        (10.0, "Étapes moyennes"),
        (19.999, "Étapes moyennes"),
        (20.0, "Étapes longues"),
        (37.5, "Étapes longues"),
    ],
)
def test_categorize_step_length(avg_words, expected):
    """Vérifie les seuils stricts en fonction du nombre
     de mots imposés dans `categorize_step_length`."""
    assert dp.categorize_step_length(avg_words) == expected


@pytest.mark.parametrize(
    "n_ing,expected",
    [
        (0, "Peu d'ingrédients"),
        (5, "Peu d'ingrédients"),
        (6, "Ingrédients modérés"),
        (10, "Ingrédients modérés"),
        (11, "Beaucoup d'ingrédients"),
        (17, "Beaucoup d'ingrédients"),
    ],
)
def test_categorize_n_ingredients(n_ing, expected):
    """Teste la catégorisation selon le nombre d’ingrédients."""
    assert dp.categorize_n_ingredients(n_ing) == expected


def test_calculate_avg_words_per_step_various():
    """
    Vérifie le calcul de la moyenne de mots par étape, y compris les cas limites (Nan et sans étapes).
    """
    steps = ["Mix and cut", "Bake", "Prep and separate the ingredients"]  # longueurs: 3, 1, 5 => moyenne = 3
    assert dp.calculate_avg_words_per_step(steps) == pytest.approx(3)

    # Via une chaîne représentant une liste Python
    steps_str = "['Mix and cut', 'Bake']"  # 3 et 1 = moyenne = 2
    assert dp.calculate_avg_words_per_step(steps_str) == pytest.approx(2)

    # Cas où aucune étape n'est spécifiée
    assert dp.calculate_avg_words_per_step([]) == 0.0
    assert dp.calculate_avg_words_per_step(np.nan) == 0.0


def test_minmax_scale_with_constant_values():
    """
    Vérifie que `_minmax_scale` renvoie 0 si tous les éléments sont identiques,
    puisque la variance est nulle.
    """
    s = pd.Series([5, 5, 5])
    scaled = dp._minmax_scale(s)
    assert (scaled == 0).all()

# ============================================================================
# Tests fonctionnels – transformations de DataFrames
# ============================================================================

def test_prepare_recipes_filters_and_adds_columns(sample_recipes_df):
    """
    Vérifie que `prepare_recipes` réalise :
    - Le filtrage des recettes invalides (minutes<=0, steps vides, n_steps=0 ...).
    - L'ajout des colonnes dérivées (log, catégories, scores).
    """
    cleaned = dp.prepare_recipes(sample_recipes_df)

    # Seules les deux premières lignes doivent passer les filtres
    assert len(cleaned) == 2

    expected_cols = {
        "log_minutes",
        "category_minutes",
        "avg_words_per_step",
        "step_length_category",
        "complexity",
        "category_n_ingredients",
        "log_n_ingredients",
        "effort_score",
        "effort_category",
    }
    assert expected_cols.issubset(cleaned.columns)
    assert cleaned["effort_score"].between(0, 100).all()


def test_prepare_interactions_converts_and_cleans(sample_interactions_df):
    """
    Vérifie que `prepare_interactions` :
    - Converti `date` en datetime.
    - Remplace les `rating == 0` par NaN.
    - Ajoute correctement `review_length` dont les valeurs doivent être >= 0.
    """
    cleaned = dp.prepare_interactions(sample_interactions_df)
    assert np.issubdtype(cleaned["date"].dtype, np.datetime64)
    assert pd.isna(cleaned.loc[1, "rating"])
    assert (cleaned["review_length"] >= 0).all()


def test_aggregate_recipe_metrics_computes_expected_columns(sample_interactions_df):
    """
    Vérifie que `_aggregate_recipe_metrics`produit les métriques clés
    d'engagement et de satisfaction par recette :
    - `n_interactions`, `avg_rating`, `bayes_mean`, `wilson_lb`, ...
    """
    metrics = dp._aggregate_recipe_metrics(sample_interactions_df)
    expected = {"n_interactions", "avg_rating", "median_rating",
                "std_rating", "n_unique_users", "n_reviews_text",
                "bayes_mean", "wilson_lb"}
    assert expected.issubset(metrics.columns)
    assert (metrics["wilson_lb"] >= 0).all()


def test_enrich_analysis_dataset_merges_and_computes_fields(sample_recipes_df, sample_interactions_df):
    """
    Vérifie que `enrich_analysis_dataset` :
    - Fusionne les recettes et ajoutes les métriques d’interactions,
    - Calcule les variables dérivées (`age_months`, `log1p_*`, `*_gap`, `winsorize`).
    """
    recipes = dp.prepare_recipes(sample_recipes_df)
    interactions = dp.prepare_interactions(sample_interactions_df)

    enriched = dp.enrich_analysis_dataset(recipes, interactions)
    expected_cols = {
        "avg_rating", "bayes_mean", "wilson_lb",
        "age_months", "interactions_per_month",
        "log_minutes", "log1p_n_interactions", "log1p_interactions_per_month",
        "log1p_n_unique_users", "rating_gap", "bayes_gap", "wilson_gap",
        "log_minutes_w", "log1p_n_interactions_w",
        "log1p_interactions_per_month_w", "log1p_n_unique_users_w",
    }
    assert expected_cols.issubset(enriched.columns)
    assert (enriched["age_months"] >= 0).all()


# ============================================================================
# Test mocké – build_analysis_dataset (sans dépendance avec des éléments extérieurs)
# ============================================================================

def test_build_analysis_dataset_with_mock(monkeypatch, sample_recipes_df, sample_interactions_df):
    """
    Remplace `load_raw_datasets` pour isoler `build_analysis_dataset` de dépendance avec des éléments extérieurs.
    Vérifie que la fonction renvoie trois DataFrames cohérents.
    """
    def _mock_load_raw_datasets():
        return sample_recipes_df, sample_interactions_df

    monkeypatch.setattr(dp, "load_raw_datasets", _mock_load_raw_datasets)
    recipes_clean, interactions_clean, df_analysis = dp.build_analysis_dataset(save=False)

    assert isinstance(recipes_clean, pd.DataFrame)
    assert isinstance(interactions_clean, pd.DataFrame)
    assert isinstance(df_analysis, pd.DataFrame)
    assert "effort_score" in recipes_clean.columns
    assert "avg_rating" in df_analysis.columns