"""
Tests unitaires pour le module `dataset_analysis.utils`.

Ce module valide la cohérence fonctionnelle des principales fonctions utilitaires
utilisées pour l'analyse statistique et la modélisation du dataset de recettes.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import dataset_analysis.utils as utils


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Génère un DataFrame simulé similaire à `analysis_dataset.csv`, avec
    les colonnes utilisées dans la suite des traitements.

    Contient des variables d'effort culinaire, de popularité et de temporalité
    permettant de tester les fonctions, sans échange avec le fichier réel.
    """
    rng = np.random.default_rng(42)
    n = 30
    return pd.DataFrame({
        "log_minutes": rng.normal(4, 0.5, n),
        "n_steps": rng.integers(5, 15, n),
        "n_ingredients": rng.integers(3, 12, n),
        "effort_score": rng.uniform(20, 80, n),
        "bayes_mean": rng.uniform(3.0, 5.0, n),
        "wilson_lb": rng.uniform(0.2, 0.9, n),
        "interactions_per_month": rng.uniform(0, 1, n),
        "log1p_interactions_per_month_w": rng.uniform(0, 0.02, n),
        "avg_words_per_step": rng.uniform(5, 20, n),
        "age_months": rng.uniform(50, 300, n),
        "effort_category": rng.choice(["Facile", "Moyen", "Difficile"], n),
    })


# ---------------------------------------------------------------------
# Tests de base
# ---------------------------------------------------------------------


def test_resolve_dataset_path_default(monkeypatch):
    """
    Vérifie que `resolve_dataset_path` retourne le chemin par défaut
    lorsque aucun argument n’est fourni.
    """
    default_path = utils.resolve_dataset_path()
    assert default_path.name == "analysis_dataset.csv"
    assert default_path.exists() or "dataset_analysis" in str(default_path)


def test_resolve_dataset_path_with_dir(tmp_path):
    """
    Vérifie que `resolve_dataset_path` résolve le chemin absolu vers le fichier d'analyse.
    """
    resolved = utils.resolve_dataset_path(tmp_path)
    assert resolved.name == "analysis_dataset.csv"
    assert resolved.parent == tmp_path


def test_load_analysis_dataset_filters_missing(tmp_path):
    """
    Simule un CSV et vérifie que `load_analysis_dataset` filtre les lignes
    contenant des valeurs manquantes dans les colonnes de popularité.
    """
    df = pd.DataFrame({
        "bayes_mean": [4.5, np.nan],
        "rating_gap": [0.5, 0.3],
        "bayes_gap": [0.4, 0.2],
    })
    csv_path = tmp_path / "analysis_dataset.csv"
    df.to_csv(csv_path, index=False)

    loaded = utils.load_analysis_dataset(csv_path)
    assert len(loaded) == 1
    assert "bayes_mean" in loaded.columns


# ---------------------------------------------------------------------
# Tests d'enrichissement des variables
# ---------------------------------------------------------------------


def test_add_feature_columns_creates_std_and_interaction(sample_df):
    """
    Vérifie la création des variables standardisées et du terme d’interaction
    soient corrects (`add_feature_columns`), tels que:
    - Les termes d'interaction (produits de variables)
    - Les variables standardisées (z-scores) pour la régression linéaire
    En vérifiant la création de colonnes contenant le suffixe "_std", ainsi que la
    standardisation soit correcte (moyenne très proche de zéro avec une tolérance de 1e-6).
    """
    enriched, meta = utils.add_feature_columns(sample_df)

    assert "steps_x_ingredients" in enriched.columns
    assert all(f"{col}_std" in enriched.columns for col in meta["standardization"])
    assert all(np.isclose(enriched[col].mean(), 0, atol=1e-6)
               for col in enriched.columns if col.endswith("_std"))


# ---------------------------------------------------------------------
# Tests de corrélations
# ---------------------------------------------------------------------


def test_compute_correlations_spearman(sample_df):
    """
    Vérifie que la matrice de corrélation Spearman générée par
    `compute_correlations` soit complète et cohérente.
    """
    result = utils.compute_correlations(sample_df, method="spearman")
    assert isinstance(result.coefficients, pd.DataFrame)
    assert result.coefficients.shape == (4, 3)
    assert result.p_values.notna().any().any()


def test_compute_correlations_pearson(sample_df):
    """
    Vérifie que la méthode Pearson, générée par `compute_correlations` fonctionne avec
    les mappings par défaut.
    Le cas où la colonne transformée n'existe pas n'est pas traitée, puisque
    cela est réalisé dans un autre test (d'où le dropna() ).
    """
    result = utils.compute_correlations(sample_df, method="pearson")
    assert "log_minutes" in result.coefficients.index
    assert "bayes_mean" in result.coefficients.columns
    assert np.all(result.n_obs.dropna().to_numpy() >= 0)



# ---------------------------------------------------------------------
# Tests d’agrégation et de stratification
# ---------------------------------------------------------------------


def test_summarize_by_effort_quantiles_returns_expected_structure(sample_df):
    """
    Vérifie que `summarize_by_effort_quantiles`effectue correctement la création correcte
    des métriques de popularité par strates d'effort culinaire, et du résumé statistique.
    """
    summary = utils.summarize_by_effort_quantiles(sample_df)
    assert isinstance(summary, dict)
    assert "summary" in summary
    assert isinstance(summary["summary"], pd.DataFrame)
    assert len(summary["quartile_edges"]) == 4


def test_summarize_by_effort_quantiles_invalid_column_raises(sample_df):
    """
    Vérifie que `summarize_by_effort_quantiles` soulève une erreur si la colonne
    de score n’existe pas.
    """
    with pytest.raises(KeyError):
        utils.summarize_by_effort_quantiles(sample_df, score_col="unknown_col")


# ---------------------------------------------------------------------
# Tests de comparaison de groupes (ANOVA / Kruskal)
# ---------------------------------------------------------------------


def test_run_group_tests_returns_dataframe(sample_df):
    """
    Vérifie `run_group_tests`, dont la sortie du test ANOVA / Kruskal et la cohérence des résultats, ici
    la métrique visée est "bayes_mean".
    Ainsi, le résultat doit contenir :
    - Au minimum les colonnes essentielles (metric, methode, ...),
    - Le nombre de catégories d'effort correct,
    La vérification de la cohérence entre l'effectif total et le nombre d'observations valides
    est également réalisée.    
    """
    results = utils.run_group_tests(
        sample_df, group_col="effort_category", metrics=["bayes_mean"]
    )
    assert set(results.columns) >= {"metric", "method", "statistic", "p_value"}
    assert results["n_groups"].iloc[0] >= 2
    assert results["n_total"].iloc[0] == len(sample_df.dropna())


def test_run_group_tests_raises_if_single_group(sample_df):
    """
    Vérifie que `run_group_tests` lève une erreur si un seul groupe est présent.
    """
    df_single = sample_df.copy()
    df_single["effort_category"] = "Unique"
    with pytest.raises(ValueError):
        utils.run_group_tests(df_single, group_col="effort_category",
                              metrics=["bayes_mean"])


# ---------------------------------------------------------------------
# Tests LOWESS
# ---------------------------------------------------------------------


def test_prepare_lowess_series_outputs_smooth_curve(sample_df):
    """
    Vérifie que `prepare_lowess_series` retourne une courbe lissée avec des points triés.
    """
    result = utils.prepare_lowess_series(
        sample_df, x_col="log_minutes", y_col="bayes_mean", frac=0.5
    )
    assert isinstance(result, dict)
    assert len(result["x_smooth"]) == len(result["y_smooth"])
    assert result["n_points"] <= len(sample_df)


def test_prepare_lowess_series_invalid_columns_raise(sample_df):
    """
    Vérifie que `prepare_lowess_series` lève une KeyError si une colonne est manquante.
    """
    with pytest.raises(KeyError):
        utils.prepare_lowess_series(sample_df, x_col="foo", y_col="bar")


# ---------------------------------------------------------------------
# Tests de modélisation
# ---------------------------------------------------------------------


def test_run_models_produces_expected_keys(sample_df):
    """
    Vérifie que `run_models` entraîne correctement les modèles et retourne
    les métriques de performance attendues pour chaque approche.
    """
    enriched, _ = utils.add_feature_columns(sample_df)
    results = utils.run_models(enriched)

    assert "models" in results
    any_target = next(iter(results["models"]))
    model_data = results["models"][any_target]
    assert set(model_data) >= {"linear_regression", "random_forest", "ols"}
    assert all("r2" in model_data[m] for m in ["linear_regression", "random_forest"])