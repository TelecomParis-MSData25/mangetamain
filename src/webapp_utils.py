"""
Fonctions utilitaires pour la webapp Streamlit.

Ce module regroupe les calculs et préparations de données nécessaires aux
visualisations et aux résumés statistiques sans dépendre directement de Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dataset_analysis.utils import summarize_by_effort_quantiles


@dataclass(frozen=True)
class DescriptiveStats:
    """Résumé statistique d'une variable numérique."""

    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    q1: float
    q3: float
    count: int

    def to_metric_dict(self) -> dict[str, float]:
        """Retourne un mapping simple, utile pour l'affichage."""
        return {
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.minimum,
            "q1": self.q1,
            "q3": self.q3,
            "max": self.maximum,
            "count": float(self.count),
        }


@dataclass(frozen=True)
class RegressionResult:
    """Informations relatives à une régression linéaire simple."""

    slope: float
    intercept: float
    r_squared: float
    x_curve: np.ndarray
    y_curve: np.ndarray


@dataclass(frozen=True)
class EffortPatternData:
    """Données permettant de tracer l'hypothèse et la réalité observée."""

    categories: list[str]
    expected: np.ndarray
    observed: np.ndarray


@dataclass(frozen=True)
class QuartilePatternData:
    """Synthèse du pattern en U observé sur les quartiles d'effort."""

    labels: Sequence[str]
    means: pd.Series


@dataclass(frozen=True)
class FilterOptions:
    """Bornes et valeurs possibles pour construire les filtres Streamlit."""

    age_range: tuple[float, float] | None
    interactions_range: tuple[int, int] | None
    effort_categories: list[str]


def compute_histogram_figure(
    series: pd.Series,
    *,
    var_label: str,
    nbins: int | None = None,
) -> go.Figure:
    """
    Génère une figure Plotly pour l'histogramme d'une série numérique.

    Parameters
    ----------
    series : pd.Series
        Série de valeurs numériques (NaN déjà filtrés).
    var_label : str
        Etiquette descriptive à utiliser pour les axes/titres.
    nbins : int | None
        Nombre de bacs. Si None, utilise une heuristique sqrt.
    """
    clean = series.dropna()
    if nbins is None:
        nbins = min(60, max(10, int(np.sqrt(len(clean))))) if len(clean) else 10

    fig = px.histogram(
        x=clean,
        nbins=nbins,
        title=f"Histogramme de {var_label}",
        labels={"x": series.name or var_label, "y": "Nombre de recettes"},
    )
    fig.update_layout(
        xaxis_title=series.name or var_label,
        yaxis_title="Nombre de recettes",
        bargap=0.05,
        height=420,
    )
    return fig


def compute_descriptive_stats(series: pd.Series) -> DescriptiveStats:
    """
    Calcule les statistiques descriptives d'une série numérique.

    Parameters
    ----------
    series : pd.Series
        Série de valeurs numériques.
    """
    clean = series.dropna()
    if clean.empty:
        raise ValueError("Impossible de calculer des statistiques sur une série vide.")

    return DescriptiveStats(
        mean=float(clean.mean()),
        median=float(clean.median()),
        std=float(clean.std()),
        minimum=float(clean.min()),
        maximum=float(clean.max()),
        q1=float(clean.quantile(0.25)),
        q3=float(clean.quantile(0.75)),
        count=int(clean.count()),
    )


def fit_simple_regression(data: pd.DataFrame, x_col: str, y_col: str) -> RegressionResult | None:
    """
    Ajuste une régression linéaire simple et retourne les paramètres principaux.

    Parameters
    ----------
    data : pd.DataFrame
        Données contenant les colonnes à corréler.
    x_col : str
        Nom de la variable explicative.
    y_col : str
        Nom de la variable réponse.
    """
    if x_col not in data.columns or y_col not in data.columns:
        return None

    clean = data[[x_col, y_col]].dropna()
    if len(clean) < 3 or clean[x_col].nunique() < 2:
        return None

    slope, intercept = np.polyfit(clean[x_col], clean[y_col], 1)
    x_curve = np.linspace(clean[x_col].min(), clean[x_col].max(), 100)
    y_curve = slope * x_curve + intercept

    corr = clean[x_col].corr(clean[y_col])
    r_squared = float(corr ** 2) if corr is not None else float("nan")

    return RegressionResult(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=r_squared,
        x_curve=x_curve,
        y_curve=y_curve,
    )


def compute_effort_pattern(data: pd.DataFrame) -> EffortPatternData | None:
    """
    Calcule les valeurs attendues et observées par catégorie d'effort.

    L'ordre des catégories est harmonisé pour garantir un affichage cohérent.
    """
    if "effort_category" not in data.columns or "bayes_mean" not in data.columns:
        return None

    aggregated = (
        data[["effort_category", "bayes_mean"]]
        .dropna()
        .groupby("effort_category", observed=False)["bayes_mean"]
        .mean()
    )
    if aggregated.empty:
        return None

    preferred_order = [
        "Très Facile",
        "Facile",
        "Modéré",
        "Difficile",
        "Très Difficile",
    ]
    ordered_categories: list[str] = [
        cat for cat in preferred_order if cat in aggregated.index
    ]
    ordered_categories.extend(
        [cat for cat in aggregated.index if cat not in ordered_categories]
    )

    observed = aggregated.reindex(ordered_categories)
    expected = np.linspace(5.0, 3.0, len(observed))

    return EffortPatternData(
        categories=ordered_categories,
        expected=expected,
        observed=observed.to_numpy(),
    )


def compute_quartile_pattern(
    data: pd.DataFrame,
    *,
    effort_col_candidates: Iterable[str] = ("effort_quartile", "effort_score", "score_effort"),
) -> QuartilePatternData | None:
    """
    Retourne la moyenne de `bayes_mean` par quartile d'effort.

    Les colonnes candidates sont testées successivement pour déterminer la meilleure option.
    """
    if "bayes_mean" not in data.columns:
        return None

    # Cas 1 : une colonne effort_quartile existe déjà
    if "effort_quartile" in effort_col_candidates and "effort_quartile" in data.columns:
        quartile_series = data["effort_quartile"].dropna()
        if quartile_series.empty:
            return None
        joined = pd.DataFrame({
            "quartile": quartile_series,
            "bayes_mean": data.loc[quartile_series.index, "bayes_mean"]
        }).dropna()
        if joined.empty:
            return None
        quartile_means = joined.groupby("quartile", observed=False)["bayes_mean"].mean().sort_index()
        return QuartilePatternData(labels=list(quartile_means.index), means=quartile_means)

    # Cas 2 : on utilise summarize_by_effort_quantiles pour générer les strates
    score_col = next((col for col in effort_col_candidates if col in data.columns and col != "effort_quartile"), None)
    if score_col is None:
        return None

    try:
        quantile_result = summarize_by_effort_quantiles(
            data.dropna(subset=[score_col, "bayes_mean"]),
            score_col=score_col,
            targets=["bayes_mean"],
            quantiles=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
        )
    except (KeyError, ValueError):
        return None

    summary = quantile_result.get("summary")
    if summary is None or ("bayes_mean", "mean") not in summary.columns:
        return None

    means = summary[("bayes_mean", "mean")].astype(float)
    labels = [str(idx) for idx in summary.index]
    return QuartilePatternData(labels=labels, means=means)


def describe_numeric_columns(
    data: pd.DataFrame, columns: Sequence[str]
) -> dict[str, DescriptiveStats]:
    """
    Calcule les DescriptiveStats pour un ensemble de colonnes numériques.
    """
    summaries: dict[str, DescriptiveStats] = {}
    for column in columns:
        if column in data.columns and np.issubdtype(data[column].dtype, np.number):
            summaries[column] = compute_descriptive_stats(data[column])
    return summaries


def infer_filter_options(data: pd.DataFrame) -> FilterOptions:
    """
    Détermine les bornes utiles pour configurer les filtres latéraux.
    """
    age_range: tuple[float, float] | None = None
    interactions_range: tuple[int, int] | None = None
    effort_categories: list[str] = []

    if "age_months" in data.columns:
        age_series = data["age_months"].dropna()
        if not age_series.empty:
            age_range = (
                float(np.floor(age_series.min())),
                float(np.ceil(age_series.max())),
            )

    if "n_interactions" in data.columns:
        interactions_series = data["n_interactions"].dropna()
        if not interactions_series.empty:
            interactions_range = (
                int(np.floor(interactions_series.min())),
                int(np.ceil(interactions_series.max())),
            )

    if "effort_category" in data.columns:
        effort_categories = sorted(
            {cat for cat in data["effort_category"].dropna()}
        )

    return FilterOptions(
        age_range=age_range,
        interactions_range=interactions_range,
        effort_categories=effort_categories,
    )
