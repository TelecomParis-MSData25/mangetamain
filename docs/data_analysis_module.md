# Module d'Analyse des Données - MangetaMain

## Description

Ce module fournit une classe `RecipeDataAnalyzer` pour analyser le dataset de recettes du projet MangetaMain. Il est basé sur l'analyse exploratoire du notebook `lab-recipe-corr-student.ipynb` et propose une version modulaire et testée des fonctionnalités d'analyse.

## Structure du projet

```
src/
├── __init__.py
├── data_analysis.py      # Module principal d'analyse
└── webapp.py            # Application Streamlit

tests/
├── __init__.py
├── test_data_analysis.py   # Tests unitaires
└── test_integration.py     # Tests d'intégration
```

## Fonctionnalités

### Classe `RecipeDataAnalyzer`

La classe principale offre les fonctionnalités suivantes :

#### Chargement et informations de base
- `load_data()` : Charge les données depuis un fichier CSV
- `get_basic_info()` : Retourne des informations de base (nombre de recettes, variables, valeurs manquantes)

#### Analyse des variables temporelles
- `analyze_minutes()` : Analyse le temps de préparation des recettes
- `remove_outliers_minutes()` : Supprime les outliers temporels

#### Analyse des contributeurs
- `analyze_contributors()` : Analyse les statistiques des contributeurs

#### Analyse des ingrédients et tags
- `analyze_ingredients()` : Analyse les ingrédients utilisés
- `analyze_tags()` : Analyse les tags des recettes
- `parse_list_column()` : Parse les colonnes contenant des listes

#### Analyse nutritionnelle
- `process_nutrition_scores()` : Traite les scores nutritionnels
- `analyze_nutrition()` : Analyse les données nutritionnelles

#### Analyse complète
- `get_complete_analysis()` : Effectue une analyse complète du dataset

## Installation et Usage

### Prérequis

```bash
# Installer les dépendances avec uv
uv sync
```

Puisque le pyproject.toml a été mis à jour pour inclure pandas, numpy, matplotlib, plotly, streamlit, pytest et pytest-cov, la commande `uv sync` installera toutes les dépendances nécessaires.

### Utilisation de base

```python
from src.data_analysis import RecipeDataAnalyzer

# Initialiser l'analyseur
analyzer = RecipeDataAnalyzer('data/RAW_recipes.csv')

# Charger les données
analyzer.load_data()

# Obtenir des informations de base
info = analyzer.get_basic_info()
print(f"Nombre de recettes: {info['n_recipes']:,}")

# Analyser les temps de préparation
minutes_stats = analyzer.analyze_minutes()
print(f"Temps moyen: {minutes_stats['mean']:.1f} minutes")

# Nettoyer les outliers
analyzer.remove_outliers_minutes()

# Analyser les ingrédients
ingredients_stats = analyzer.analyze_ingredients()
print(f"Ingrédients uniques: {ingredients_stats['n_unique_ingredients']:,}")
```

### Exemple complet

Exécutez l'exemple d'analyse complète :

```bash
cd /home/bnj/mangetamain
uv run python examples/analyze_dataset.py
```

## Tests

### Tests unitaires

Les tests unitaires utilisent des données simulées pour tester chaque fonction :

```bash
# Lancer les tests unitaires
uv run pytest tests/test_data_analysis.py -v

# Avec couverture de code
uv run pytest tests/test_data_analysis.py --cov=src --cov-report=term-missing
```

### Tests d'intégration

Les tests d'intégration utilisent le vrai dataset pour valider le comportement :

```bash
# Lancer les tests d'intégration
uv run pytest tests/test_integration.py -v

# Tests marqués comme lents (peuvent être exclus)
uv run pytest -m "not slow"
```

### Tous les tests

```bash
# Lancer tous les tests avec couverture
uv run pytest -v --cov=src --cov-report=term-missing
```

## Analyses Disponibles

### 1. Informations de base
- Nombre de recettes et variables
- Valeurs manquantes par colonne

### 2. Analyse temporelle
- Statistiques des temps de préparation (moyenne, médiane, quartiles)
- Détection et suppression d'outliers
- Conversion en heures/minutes

### 3. Analyse des contributeurs
- Nombre de contributeurs uniques
- Top contributeur et sa part de marché
- Distribution des contributions

### 4. Analyse des ingrédients
- Ingrédients uniques et mentions totales
- Top 10 des ingrédients les plus utilisés
- Parsing des listes d'ingrédients

### 5. Analyse des tags
- Tags uniques et mentions totales
- Tags les plus populaires
- Recherche de tags spécifiques (ex: végétarien)

### 6. Analyse nutritionnelle
- Traitement des scores nutritionnels
- Statistiques sur les calories, graisses, sucres, etc.

### 7. Analyse des étapes et complexité
- Nombre d'étapes par recette
- Nombre d'ingrédients par recette
- Statistiques de complexité

## Résultats Typiques

Avec le dataset complet (~231k recettes) :

- **Recettes** : 231,637
- **Contributeurs uniques** : ~27,923
- **Ingrédients uniques** : ~14,935
- **Tags uniques** : ~552
- **Top ingrédients** : salt, butter, sugar, onion, water
- **Temps médian** : 40 minutes
- **Complexité moyenne** : 9.8 étapes, 9.1 ingrédients
