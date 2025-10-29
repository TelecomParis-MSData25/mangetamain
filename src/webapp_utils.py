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
    """
    Résume les statistiques descriptives d'une variable numérique.

    :ivar mean: Moyenne arithmétique.
    :vartype mean: float
    :ivar median: Médiane.
    :vartype median: float
    :ivar std: Écart-type.
    :vartype std: float
    :ivar minimum: Valeur minimale observée.
    :vartype minimum: float
    :ivar maximum: Valeur maximale observée.
    :vartype maximum: float
    :ivar q1: Premier quartile (25 %).
    :vartype q1: float
    :ivar q3: Troisième quartile (75 %).
    :vartype q3: float
    :ivar count: Effectif total des valeurs non manquantes.
    :vartype count: int
    """

    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    q1: float
    q3: float
    count: int

    def to_metric_dict(self) -> dict[str, float]:
        """
        Transforme l'instance en dictionnaire prêt pour l'affichage.

        :returns: Mapping clé/valeur avec les statistiques principales.
        :rtype: dict[str, float]
        """
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
    """
    Informations principales d'une régression linéaire simple.

    :ivar slope: Coefficient directeur de la droite ajustée.
    :vartype slope: float
    :ivar intercept: Ordonnée à l'origine.
    :vartype intercept: float
    :ivar r_squared: Coefficient de détermination R².
    :vartype r_squared: float
    :ivar x_curve: Abscisses de la courbe ajustée.
    :vartype x_curve: np.ndarray
    :ivar y_curve: Ordonnées de la courbe ajustée.
    :vartype y_curve: np.ndarray
    """

    slope: float
    intercept: float
    r_squared: float
    x_curve: np.ndarray
    y_curve: np.ndarray


@dataclass(frozen=True)
class EffortPatternData:
    """
    Données permettant de tracer l'hypothèse et la réalité observée.

    :ivar categories: Ordre des catégories d'effort.
    :vartype categories: list[str]
    :ivar expected: Valeurs attendues selon l'hypothèse.
    :vartype expected: np.ndarray
    :ivar observed: Valeurs observées (moyenne réelle).
    :vartype observed: np.ndarray
    """

    categories: list[str]
    expected: np.ndarray
    observed: np.ndarray


@dataclass(frozen=True)
class QuartilePatternData:
    """
    Synthétise le pattern en U observé sur les quartiles d'effort.

    :ivar labels: Libellés utilisés pour les strates d'effort.
    :vartype labels: Sequence[str]
    :ivar means: Moyennes de popularité par quartile.
    :vartype means: pd.Series
    """

    labels: Sequence[str]
    means: pd.Series


@dataclass(frozen=True)
class FilterOptions:
    """
    Bornes et valeurs possibles pour construire les filtres Streamlit.

    :ivar age_range: Intervalle minimal/maximal d'âge des recettes (en mois).
    :vartype age_range: tuple[float, float] | None
    :ivar interactions_range: Intervalle des interactions utilisateur.
    :vartype interactions_range: tuple[int, int] | None
    :ivar effort_categories: Catégories d'effort disponibles pour filtrage.
    :vartype effort_categories: list[str]
    """

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

    :param series: Série de valeurs numériques (NaN déjà filtrés).
    :type series: pd.Series
    :param var_label: Étiquette descriptive à afficher sur le graphique.
    :type var_label: str
    :param nbins: Nombre de bacs souhaité ; ``None`` applique une heuristique racine carrée.
    :type nbins: int | None
    :returns: Figure Plotly contenant l'histogramme.
    :rtype: go.Figure
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

    :param series: Série de valeurs numériques.
    :type series: pd.Series
    :returns: Objet ``DescriptiveStats`` avec les métriques calculées.
    :rtype: DescriptiveStats
    :raises ValueError: Si la série ne contient aucune valeur exploitable.
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

    :param data: Données contenant les colonnes à mettre en relation.
    :type data: pd.DataFrame
    :param x_col: Nom de la variable explicative.
    :type x_col: str
    :param y_col: Nom de la variable réponse.
    :type y_col: str
    :returns: Résultat de la régression ou ``None`` si l'ajustement est impossible.
    :rtype: RegressionResult | None
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

    :param data: Dataset contenant au minimum ``effort_category`` et ``bayes_mean``.
    :type data: pd.DataFrame
    :returns: Données prêtes pour la visualisation ou ``None`` si les colonnes manquent.
    :rtype: EffortPatternData | None
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

    :param data: Dataset contenant la variable ``bayes_mean``.
    :type data: pd.DataFrame
    :param effort_col_candidates: Liste de colonnes à tester pour la stratification d'effort.
    :type effort_col_candidates: Iterable[str]
    :returns: Structure ``QuartilePatternData`` ou ``None`` si le calcul échoue.
    :rtype: QuartilePatternData | None
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

    :param data: DataFrame contenant les colonnes à résumer.
    :type data: pd.DataFrame
    :param columns: Noms des colonnes à inspecter.
    :type columns: Sequence[str]
    :returns: Mapping ``{colonne: DescriptiveStats}``.
    :rtype: dict[str, DescriptiveStats]
    """
    summaries: dict[str, DescriptiveStats] = {}
    for column in columns:
        if column in data.columns and np.issubdtype(data[column].dtype, np.number):
            summaries[column] = compute_descriptive_stats(data[column])
    return summaries


def infer_filter_options(data: pd.DataFrame) -> FilterOptions:
    """
    Détermine les bornes utiles pour configurer les filtres latéraux.

    :param data: Dataset courant.
    :type data: pd.DataFrame
    :returns: Options de filtrage exploitables dans l'UI.
    :rtype: FilterOptions
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
