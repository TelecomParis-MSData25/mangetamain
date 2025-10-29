"""
Bibliothèque d'analyse statistique pour l'étude de la relation effort culinaire ↔ popularité.

Fournit des fonctions pures et réutilisables pour l'analyse exploratoire et inférentielle
du dataset de recettes, conçues pour l'intégration dans des pipelines, applications web
ou scripts d'automatisation.

Fonctionnalités
---------------
- Chargement et prétraitement avec filtrage configurable
- Analyses de corrélation (Pearson, Spearman) avec transformations automatiques
- Agrégations par quantiles et tests de comparaison de groupes (ANOVA, Kruskal-Wallis)
- Régression non-paramétrique (LOWESS) pour relations non-linéaires
- Modélisation prédictive (linéaire, forêts aléatoires, OLS) avec métriques d'évaluation

Architecture
------------
Fonctions sans effets de bord : pas de modification in-place, pas d'affichage graphique,
retours sérialisables (DataFrames, dictionnaires, dataclasses).

Usage
-----
>>> from dataset_analysis.utils import load_analysis_dataset, compute_correlations
>>> df = load_analysis_dataset()
>>> result = compute_correlations(df, method='spearman')
>>> print(result.coefficients)

Notes
-----
Les transformations (log, winsorisation, standardisation) sont appliquées automatiquement
selon les méthodes statistiques, conformément aux bonnes pratiques d'analyse exploratoire.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, pearsonr, spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# Configuration et constantes
# ==============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "dataset_analysis" / "data" / "analysis_dataset.csv"

# Variables par défaut pour l'analyse bivariée
DEFAULT_EFFORT_VARS = ["log_minutes", "n_steps", "n_ingredients", "effort_score"]
DEFAULT_POPULARITY_VARS = ["bayes_mean", "wilson_lb", "interactions_per_month"]

# Mapping des transformations pour la corrélation de Pearson
# Les variables sont remplacées par leurs versions transformées pour satisfaire
# les hypothèses de normalité et de linéarité
PEARSON_MAP_EFFORT = {
    "log_minutes": "log_minutes",  # transformation logarithmique déjà appliquée
    "n_steps": "n_steps",  # distribution acceptable sans transformation
    "n_ingredients": "log_n_ingredients",  # normalisation via log
    "effort_score": "effort_score",  # score composite normalisé
}
PEARSON_MAP_POPULARITY = {
    "bayes_mean": "bayes_mean",
    "wilson_lb": "wilson_lb",
    "interactions_per_month": "log1p_interactions_per_month_w",  # log1p + winsorisation
}

DEFAULT_POPULARITY_FILTER_COLS = ["bayes_mean", "rating_gap", "bayes_gap"]

DEFAULT_QUANTILE_TARGETS = [
    "bayes_mean",
    "wilson_lb",
    "log1p_interactions_per_month_w",
]

DEFAULT_MODEL_FEATURES = [
    "log_minutes_std",
    "n_steps_std",
    "n_ingredients_std",
    "steps_x_ingredients_std",
    "age_months_std",
]

DEFAULT_MODEL_TARGETS = ["bayes_mean", "wilson_lb", "log1p_interactions_per_month_w"]


@dataclass(frozen=True)
class CorrelationResult:
    """
    Encapsulation des résultats d'analyse de corrélation.

    :ivar coefficients: Matrice des coefficients de corrélation (ρ ou r selon la méthode).
    :vartype coefficients: pd.DataFrame
    :ivar p_values: Matrice des p-values associées (test bilatéral H₀: ρ = 0).
    :vartype p_values: pd.DataFrame
    :ivar n_obs: Matrice des effectifs (paires complètes après suppression des valeurs manquantes).
    :vartype n_obs: pd.DataFrame
    """

    coefficients: pd.DataFrame
    p_values: pd.DataFrame
    n_obs: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """
        Sérialise les matrices en dictionnaires imbriqués.

        :returns: Structure compatible JSON pour export ou usage API.
        :rtype: dict[str, Any]
        """
        return {
            "coefficients": self.coefficients.to_dict(),
            "p_values": self.p_values.to_dict(),
            "n_obs": self.n_obs.to_dict(),
        }


# ==============================================================================
# Chargement et préparation des données
# ==============================================================================


def resolve_dataset_path(path: str | Path | None = None) -> Path:
    """
    Résout le chemin absolu vers le fichier d'analyse.

    :param path: Chemin personnalisé vers le fichier CSV ou son répertoire parent.
        Si ``None``, utilise le chemin par défaut du projet.
    :type path: str | Path | None
    :returns: Chemin absolu résolu vers ``analysis_dataset.csv``.
    :rtype: Path

    .. note::
       Si un répertoire est fourni, le fichier ``analysis_dataset.csv`` est automatiquement ajouté au chemin.
    """
    if path is None:
        return DEFAULT_DATA_PATH

    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "analysis_dataset.csv"
    return resolved


def load_analysis_dataset(
    path: str | Path | None = None,
    *,
    columns: Sequence[str] | None = None,
    drop_missing_popularity: bool = True,
    popularity_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Charge le dataset d'analyse avec filtrage optionnel des observations incomplètes.

    :param path: Chemin vers le fichier CSV ou son répertoire parent.
    :type path: str | Path | None
    :param columns: Sous-ensemble de colonnes à charger. Lorsque ``None``, toutes les colonnes sont conservées.
    :type columns: Sequence[str] | None
    :param drop_missing_popularity: Indique s'il faut exclure les recettes sans métriques de popularité valides.
    :type drop_missing_popularity: bool
    :param popularity_cols: Colonnes utilisées pour identifier les valeurs manquantes de popularité.
        Par défaut ``['bayes_mean', 'rating_gap', 'bayes_gap']``.
    :type popularity_cols: Sequence[str] | None
    :returns: Dataset nettoyé avec index réinitialisé.
    :rtype: pd.DataFrame

    .. note::
       Le filtrage sur la popularité permet d'exclure les recettes sans interactions,
       évitant ainsi les biais dans les analyses de corrélation et de régression.
    """

    dataset_path = resolve_dataset_path(path)
    df = pd.read_csv(dataset_path, usecols=columns)

    if drop_missing_popularity:
        cols = list(popularity_cols or DEFAULT_POPULARITY_FILTER_COLS)
        existing = [col for col in cols if col in df.columns]
        if existing:
            df = df.dropna(subset=existing)

    return df.reset_index(drop=True)


def add_feature_columns(
    df: pd.DataFrame,
    *,
    interaction: bool = True,
    standardize: bool = True,
    columns_to_standardize: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Enrichit le dataset avec des variables dérivées pour la modélisation.

    Cette fonction génère :
    - Termes d'interaction (produits de variables)
    - Variables standardisées (z-scores) pour la régression linéaire

    :param df: Dataset source (non modifié, une copie est créée).
    :type df: pd.DataFrame
    :param interaction: Indique s'il faut créer la variable ``steps_x_ingredients``.
    :type interaction: bool
    :param standardize: Indique s'il faut générer les versions centrées-réduites (suffixe ``_std``).
    :type standardize: bool
    :param columns_to_standardize: Liste personnalisée des colonnes à standardiser.
        Par défaut ``['log_minutes', 'n_steps', 'n_ingredients', 'avg_words_per_step', 'age_months', 'effort_score', 'steps_x_ingredients']``.
    :type columns_to_standardize: Sequence[str] | None
    :returns: Tuple composé du DataFrame enrichi et des métadonnées de standardisation (moyenne, écart-type par variable).
    :rtype: tuple[pd.DataFrame, dict[str, Any]]

    .. note::
       La standardisation utilise l'écart-type de population (``ddof=0``) pour rester
       cohérente avec les pratiques courantes en machine learning. Les métadonnées permettent
       de reproduire la transformation sur de nouvelles données.
    """

    enriched = df.copy()
    meta: dict[str, Any] = {}

    if interaction:
        enriched["steps_x_ingredients"] = enriched["n_steps"] * enriched["n_ingredients"]

    if standardize:
        targets = (
            list(columns_to_standardize)
            if columns_to_standardize is not None
            else [
                "log_minutes",
                "n_steps",
                "n_ingredients",
                "avg_words_per_step",
                "age_months",
                "effort_score",
                "steps_x_ingredients",
            ]
        )

        meta["standardization"] = {}
        for column in targets:
            if column not in enriched.columns:
                continue
            values = enriched[column].astype(float)
            mean = values.mean()
            std = values.std(ddof=0)
            meta["standardization"][column] = {"mean": float(mean), "std": float(std)}
            if std == 0 or np.isnan(std):
                enriched[f"{column}_std"] = 0.0
            else:
                enriched[f"{column}_std"] = (values - mean) / std

    return enriched, meta


# ==============================================================================
# Analyses de corrélation et agrégations statistiques
# ==============================================================================


def compute_correlations(
    df: pd.DataFrame,
    *,
    effort_vars: Sequence[str] | None = None,
    popularity_vars: Sequence[str] | None = None,
    method: Literal["spearman", "pearson"] = "spearman",
    pearson_effort_map: Mapping[str, str] | None = None,
    pearson_popularity_map: Mapping[str, str] | None = None,
) -> CorrelationResult:
    """
    Calcule les corrélations bivariées entre variables d'effort et de popularité.

    Cette fonction génère trois matrices :
    - Coefficients de corrélation (ρ pour Spearman, r pour Pearson)
    - P-values associées (test bilatéral)
    - Effectifs (nombre de paires complètes)

    :param df: Dataset contenant les variables d'intérêt.
    :type df: pd.DataFrame
    :param effort_vars: Variables d'effort culinaire. Par défaut ``['log_minutes', 'n_steps',
        'n_ingredients', 'effort_score']``.
    :type effort_vars: Sequence[str] | None
    :param popularity_vars: Variables de popularité. Par défaut ``['bayes_mean', 'wilson_lb',
        'interactions_per_month']``.
    :type popularity_vars: Sequence[str] | None
    :param method: Méthode de corrélation (``'spearman'`` robuste aux distributions non-normales
        ou ``'pearson'`` pour corrélations linéaires).
    :type method: Literal["spearman", "pearson"]
    :param pearson_effort_map: Correspondance entre variables d'effort et leurs versions transformées.
    :type pearson_effort_map: Mapping[str, str] | None
    :param pearson_popularity_map: Correspondance entre variables de popularité et leurs versions transformées.
    :type pearson_popularity_map: Mapping[str, str] | None
    :returns: Dataclass contenant les trois matrices (coefficients, ``p_values``, ``n_obs``).
    :rtype: CorrelationResult

    .. note::
       Pour la corrélation de Pearson, les transformations logarithmiques et la
       winsorisation sont appliquées automatiquement via les mappings pour satisfaire
       les hypothèses de normalité bivariée. Pour Spearman, les variables brutes
       sont utilisées.
    """

    effort_vars = list(effort_vars or DEFAULT_EFFORT_VARS)
    popularity_vars = list(popularity_vars or DEFAULT_POPULARITY_VARS)

    pearson_effort_map = pearson_effort_map or PEARSON_MAP_EFFORT
    pearson_popularity_map = pearson_popularity_map or PEARSON_MAP_POPULARITY

    coefficients = pd.DataFrame(index=effort_vars, columns=popularity_vars, dtype=float)
    p_values = coefficients.copy()
    n_obs = coefficients.copy()

    for effort in effort_vars:
        effort_col = pearson_effort_map.get(effort, effort) if method == "pearson" else effort
        if effort_col not in df.columns:
            continue

        for popularity in popularity_vars:
            popularity_col = (
                pearson_popularity_map.get(popularity, popularity)
                if method == "pearson"
                else popularity
            )
            if popularity_col not in df.columns:
                continue

            subset = (
                df[[effort_col, popularity_col]].dropna().to_numpy(dtype=float)
                if method == "pearson"
                else df[[effort_col, popularity_col]].dropna()
            )

            if len(subset) < 3:
                coefficients.loc[effort, popularity] = np.nan
                p_values.loc[effort, popularity] = np.nan
                n_obs.loc[effort, popularity] = len(subset)
                continue

            if method == "spearman":
                rho, p_val = spearmanr(subset.iloc[:, 0], subset.iloc[:, 1])
            else:
                arr = subset
                rho, p_val = pearsonr(arr[:, 0], arr[:, 1])

            coefficients.loc[effort, popularity] = float(rho)
            p_values.loc[effort, popularity] = float(p_val)
            n_obs.loc[effort, popularity] = len(subset)

    return CorrelationResult(coefficients=coefficients, p_values=p_values, n_obs=n_obs)


def summarize_by_effort_quantiles(
    df: pd.DataFrame,
    *,
    score_col: str = "effort_score",
    targets: Sequence[str] | None = None,
    quantiles: int = 4,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Agrège les métriques de popularité par strates d'effort culinaire.

    Effectue une stratification du score d'effort en quantiles et calcule
    les statistiques descriptives des variables de popularité pour chaque strate.

    :param df: Dataset d'analyse.
    :type df: pd.DataFrame
    :param score_col: Variable continue utilisée pour la stratification.
    :type score_col: str
    :param targets: Variables de popularité à agréger. Par défaut ``['bayes_mean', 'wilson_lb', 'log1p_interactions_per_month_w']``.
    :type targets: Sequence[str] | None
    :param quantiles: Nombre de quantiles (``4`` = quartiles, ``5`` = quintiles, etc.).
    :type quantiles: int
    :param labels: Étiquettes personnalisées pour les strates.
    :type labels: Sequence[str] | None
    :returns: Dictionnaire contenant les strates calculées, les agrégations et les bornes de quantiles.
    :rtype: dict[str, Any]

    .. note::
       Cette fonction est utile pour détecter des effets de seuil ou des relations
       non linéaires entre effort et popularité via une analyse de variance inter-strates.
    """

    if score_col not in df.columns:
        raise KeyError(f"Colonne '{score_col}' introuvable dans le DataFrame.")

    targets = list(targets or DEFAULT_QUANTILE_TARGETS)
    available_targets = [col for col in targets if col in df.columns]
    if not available_targets:
        raise ValueError("Aucune cible valide pour la synthèse par quantiles.")

    quantile_series = pd.qcut(
        df[score_col].dropna(),
        q=quantiles,
        labels=labels,
        duplicates="drop",
    )

    enriched = df.copy()
    enriched = enriched.loc[quantile_series.index]
    enriched["effort_quantile"] = quantile_series

    agg_dict = {col: ["mean", "std", "count"] for col in available_targets}
    summary = enriched.groupby("effort_quantile", observed=False).agg(agg_dict)

    quartile_edges = list(map(str, quantile_series.cat.categories))

    return {
        "quartile_column": enriched["effort_quantile"].astype(str).to_list(),
        "summary": summary,
        "summary_dict": summary.to_dict(),
        "quartile_edges": quartile_edges,
    }


def run_group_tests(
    df: pd.DataFrame,
    *,
    group_col: str,
    metrics: Sequence[str],
    methods: Sequence[Literal["anova", "kruskal"]] = ("anova", "kruskal"),
) -> pd.DataFrame:
    """
    Effectue des tests de comparaison de moyennes entre groupes.

    Implémente l'ANOVA paramétrique (hypothèse de normalité) et le test de
    Kruskal-Wallis non-paramétrique pour évaluer l'homogénéité des distributions
    de variables continues à travers des groupes catégoriels.

    :param df: Dataset d'analyse.
    :type df: pd.DataFrame
    :param group_col: Variable catégorielle définissant les groupes (ex. ``'effort_category'``).
    :type group_col: str
    :param metrics: Variables continues à tester.
    :type metrics: Sequence[str]
    :param methods: Tests statistiques à appliquer (ANOVA, Kruskal).
    :type methods: Sequence[Literal["anova", "kruskal"]]
    :returns: Tableau de résultats (metric, method, statistic, p_value, n_groups, n_total, group_sizes).
    :rtype: pd.DataFrame
    :raises ValueError: Si moins de deux groupes restent après le nettoyage.

    .. note::
       - ANOVA : test F de Fisher-Snedecor (hypothèses de normalité et homoscédasticité).
       - Kruskal-Wallis : test H non paramétrique (basé sur les rangs, robuste).
    """

    if group_col not in df.columns:
        raise KeyError(f"Colonne '{group_col}' introuvable.")

    clean = df.dropna(subset=[group_col, *metrics]).copy()
    results: list[dict[str, Any]] = []

    grouped = clean.groupby(group_col, observed=False)
    group_sizes = grouped.size()

    n_groups = len(group_sizes)
    if n_groups < 2:
        raise ValueError(
            f"Au moins deux groupes requis pour les tests statistiques, trouvé {n_groups}."
        )

    for metric in metrics:
        if metric not in clean.columns:
            continue

        groups = [group[metric].to_numpy() for _, group in grouped]
        for method in methods:
            if method == "anova":
                stat, p_val = f_oneway(*groups)
            elif method == "kruskal":
                stat, p_val = kruskal(*groups)
            else:
                raise ValueError(f"Test '{method}' non supporté.")

            results.append(
                {
                    "metric": metric,
                    "method": method,
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "n_groups": int(len(groups)),
                    "n_total": int(len(clean)),
                    "group_sizes": group_sizes.to_dict(),
                }
            )

    return pd.DataFrame(results)


# ==============================================================================
# Régression non-paramétrique et analyses non-linéaires
# ==============================================================================


def prepare_lowess_series(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    frac: float = 0.3,
    sample_size: int | None = 20_000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Calcule une régression LOWESS (Locally Weighted Scatterplot Smoothing).

    La régression LOWESS est une méthode non-paramétrique qui ajuste localement
    des polynômes pondérés pour capturer des relations non-linéaires sans spécifier
    de forme fonctionnelle a priori.

    :param df: Dataset contenant les variables d'entrée.
    :type df: pd.DataFrame
    :param x_col: Nom de la variable prédictrice (axe X).
    :type x_col: str
    :param y_col: Nom de la variable réponse (axe Y).
    :type y_col: str
    :param frac: Fraction de points pour chaque ajustement local (bandwidth). Valeur faible → courbe flexible.
    :type frac: float
    :param sample_size: Taille d'échantillon maximale (LOWESS est coûteux). ``None`` pour utiliser toutes les lignes.
    :type sample_size: int | None
    :param random_state: Graine aléatoire utilisée pour l'échantillonnage.
    :type random_state: int
    :returns: Dictionnaire contenant les points bruts, la courbe lissée, le paramètre de lissage et le volume d'échantillon.
    :rtype: dict[str, Any]
    :raises ValueError: Si aucun point valide n'est disponible après nettoyage.

    .. note::
       Cette fonction ne génère pas de graphique mais fournit les données prêtes à tracer.
    """

    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"Colonnes {x_col} / {y_col} introuvables.")

    data = df[[x_col, y_col]].dropna()
    if sample_size and len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=random_state)

    if data.empty:
        raise ValueError("Aucun point disponible après nettoyage pour LOWESS.")

    smoothed = lowess(
        endog=data[y_col],
        exog=data[x_col],
        frac=frac,
        return_sorted=True,
    )

    return {
        "x_raw": data[x_col].to_list(),
        "y_raw": data[y_col].to_list(),
        "x_smooth": smoothed[:, 0].tolist(),
        "y_smooth": smoothed[:, 1].tolist(),
        "frac": frac,
        "n_points": int(len(data)),
    }


# ==============================================================================
# Modélisation prédictive et évaluation
# ==============================================================================


def run_models(
    df: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    targets: Sequence[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    rf_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Entraîne et évalue plusieurs modèles de régression pour prédire la popularité.

    Cette fonction implémente un pipeline complet d'entraînement avec validation
    hold-out, incluant trois approches :
    - Régression linéaire (scikit-learn) : modèle paramétrique simple
    - Forêt aléatoire (scikit-learn) : modèle non-linéaire avec interactions
    - Régression OLS (statsmodels) : inférence statistique avec p-values

    :param df: Dataset contenant les prédicteurs et les variables cibles.
    :type df: pd.DataFrame
    :param features: Variables prédictives. Par défaut ``['log_minutes_std', 'n_steps_std',
        'n_ingredients_std', 'steps_x_ingredients_std', 'age_months_std']``.
    :type features: Sequence[str] | None
    :param targets: Variables à prédire. Par défaut ``['bayes_mean', 'wilson_lb', 'log1p_interactions_per_month_w']``.
    :type targets: Sequence[str] | None
    :param test_size: Fraction du dataset réservée à la validation (hold-out).
    :type test_size: float
    :param random_state: Graine aléatoire pour la reproductibilité.
    :type random_state: int
    :param rf_params: Hyperparamètres passés à ``RandomForestRegressor``.
    :type rf_params: Mapping[str, Any] | None
    :returns: Structure contenant la configuration utilisée et les métriques des modèles entraînés.
    :rtype: dict[str, Any]

    .. note::
       - Les prédicteurs doivent être pré-standardisés (suffixe ``_std``).
       - La régression linéaire applique une seconde standardisation via ``StandardScaler``.
       - OLS est ajusté sur l'ensemble complet pour l'inférence statistique.
       - Les métriques reportées proviennent de l'ensemble de test.
    """

    features = list(features or DEFAULT_MODEL_FEATURES)
    targets = list(targets or DEFAULT_MODEL_TARGETS)
    rf_params = dict(rf_params or {})

    results: dict[str, Any] = {
        "settings": {
            "features": features,
            "targets": targets,
            "test_size": test_size,
            "random_state": random_state,
            "rf_params": rf_params,
        },
        "models": {},
    }

    for target in targets:
        available_cols = [col for col in features if col in df.columns]
        if target not in df.columns or not available_cols:
            continue

        modelling_df = df[available_cols + [target]].dropna()
        if modelling_df.empty:
            continue

        X = modelling_df[available_cols]
        y = modelling_df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)

        y_pred_lr = lr.predict(X_test_scaled)
        lr_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_lr))),
            "mae": float(mean_absolute_error(y_test, y_pred_lr)),
            "r2": float(r2_score(y_test, y_pred_lr)),
            "intercept": float(lr.intercept_),
            "coefficients": {
                feature: float(coeff) for feature, coeff in zip(available_cols, lr.coef_)
            },
        }

        rf = RandomForestRegressor(random_state=random_state, **rf_params)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_rf))),
            "mae": float(mean_absolute_error(y_test, y_pred_rf)),
            "r2": float(r2_score(y_test, y_pred_rf)),
            "feature_importances": {
                feature: float(importance)
                for feature, importance in zip(available_cols, rf.feature_importances_)
            },
        }

        X_ols = sm.add_constant(X, prepend=True)
        ols_model = sm.OLS(y, X_ols).fit()
        ols_metrics = {
            "r2": float(ols_model.rsquared),
            "adj_r2": float(ols_model.rsquared_adj),
            "nobs": float(ols_model.nobs),
            "coefficients": {col: float(val) for col, val in ols_model.params.items()},
            "p_values": {col: float(val) for col, val in ols_model.pvalues.items()},
            "std_err": {col: float(val) for col, val in ols_model.bse.items()},
        }

        results["models"][target] = {
            "linear_regression": lr_metrics,
            "random_forest": rf_metrics,
            "ols": ols_metrics,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }

    return results


__all__ = [
    "CorrelationResult",
    "resolve_dataset_path",
    "load_analysis_dataset",
    "add_feature_columns",
    "compute_correlations",
    "summarize_by_effort_quantiles",
    "run_group_tests",
    "prepare_lowess_series",
    "run_models",
]
