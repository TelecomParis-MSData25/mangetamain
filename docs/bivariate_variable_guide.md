# Guide des Variables pour l'Analyse Bivariée

**Analyse de la relation effort culinaire ↔ popularité**

Source : `analysis_dataset.csv` généré par `dataset_preprocessing.py`

---

## Variables Clés

### Métriques de Satisfaction et Qualité

| Variable     | Interprétation        | Description Technique                                                                        |
| ------------ | --------------------- | -------------------------------------------------------------------------------------------- |
| `bayes_mean` | Satisfaction générale | Moyenne bayésienne avec prior empirique (prior_count=10, global_mean≈3.73)                   |
| `wilson_lb`  | Qualité fiable        | Borne inférieure de l'intervalle de confiance de Wilson (95%, notes≥4 considérées positives) |
| `avg_rating` | Note moyenne brute    | Moyenne arithmétique simple sans correction de biais                                         |

### Métriques d'Engagement et Popularité

| Variable                         | Interprétation             | Description Technique                                                 |
| -------------------------------- | -------------------------- | --------------------------------------------------------------------- |
| `log1p_interactions_per_month_w` | Engagement normalisé       | log(1+interactions/age_months), winsorisé aux percentiles 5%-99%      |
| `n_interactions`                 | Volume d'interactions      | Nombre total d'évaluations (ratings non nuls)                         |
| `n_unique_users`                 | Diversité de l'audience    | Cardinalité des utilisateurs ayant interagi                           |
| `interactions_per_month`         | Taux d'engagement temporel | Interactions normalisées par l'ancienneté (n_interactions/age_months) |

### Métriques d'Effort Culinaire

| Variable             | Interprétation           | Description Technique                                                                                                     |
| -------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `effort_score`       | Score d'effort composite | Indice pondéré (0-100) : combinaison linéaire de log_minutes, n_steps, avg_words_per_step, log_n_ingredients (normalisés) |
| `log_minutes`        | Temps de préparation     | Transformation logarithmique de la durée (minutes)                                                                        |
| `n_steps`            | Complexité procédurale   | Nombre d'étapes de la recette                                                                                             |
| `n_ingredients`      | Diversité des composants | Nombre total d'ingrédients                                                                                                |
| `avg_words_per_step` | Complexité descriptive   | Longueur moyenne des instructions par étape                                                                               |

---

## 1. Variables d'Effort Culinaire

### 1.1 Temps de préparation

| Variable             | Type         | Description                                                        |
| -------------------- | ------------ | ------------------------------------------------------------------ |
| `minutes`            | Continue     | Durée brute de préparation (plage valide : [1, 360] minutes)       |
| `log_minutes`        | Continue     | Transformation logarithmique pour normalisation de la distribution |
| `log_minutes_w`      | Continue     | Version winsorisée (écrêtage aux percentiles 5%-99%)               |
| `log_minutes_scaled` | Continue     | Normalisation min-max [0,1] utilisée dans le score composite       |
| `category_minutes`   | Catégorielle | Discrétisation heuristique : {Rapide, Moyenne, Longue}             |

### 1.2 Processus de préparation

| Variable                    | Type         | Description                                                     |
| --------------------------- | ------------ | --------------------------------------------------------------- |
| `n_steps`                   | Discrète     | Nombre d'étapes de la recette                                   |
| `n_steps_scaled`            | Continue     | Version normalisée [0,1] pour le score composite                |
| `complexity`                | Catégorielle | Stratification : {Simple, Modéré, Complexe, Très complexe}      |
| `avg_words_per_step`        | Continue     | Longueur moyenne des instructions (parsing de la liste `steps`) |
| `avg_words_per_step_scaled` | Continue     | Version normalisée [0,1]                                        |
| `step_length_category`      | Catégorielle | Classification : {Étapes courtes, moyennes, longues}            |

### 1.3 Composition

| Variable                   | Type         | Description                                           |
| -------------------------- | ------------ | ----------------------------------------------------- |
| `n_ingredients`            | Discrète     | Nombre total d'ingrédients                            |
| `log_n_ingredients`        | Continue     | Transformation logarithmique                          |
| `log_n_ingredients_scaled` | Continue     | Version normalisée [0,1]                              |
| `category_n_ingredients`   | Catégorielle | Stratification : {Peu, Ingrédients modérés, Beaucoup} |

### 1.4 Indice d'effort composite

| Variable          | Type         | Description                                                                                                                              |
| ----------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `effort_score`    | Continue     | Score synthétique [0,100] : combinaison pondérée des variables normalisées (log_minutes, n_steps, avg_words_per_step, log_n_ingredients) |
| `effort_category` | Catégorielle | Stratification du score : {Très Facile, Facile, Moyen, Difficile, Très Difficile}                                                        |
| `effort_quartile` | Catégorielle | Découpage en quartiles : {Q1, Q2, Q3, Q4}                                                                                                |

---

## 2. Variables de Popularité, Satisfaction et Engagement

### 2.1 Métriques de satisfaction

| Variable         | Type     | Description                                                                                                           |
| ---------------- | -------- | --------------------------------------------------------------------------------------------------------------------- |
| `n_interactions` | Discrète | Cardinalité des interactions (ratings non nuls)                                                                       |
| `avg_rating`     | Continue | Moyenne arithmétique simple des notes                                                                                 |
| `median_rating`  | Continue | Médiane des notes (robuste aux valeurs extrêmes)                                                                      |
| `std_rating`     | Continue | Écart-type des notes (mesure de dispersion)                                                                           |
| `bayes_mean`     | Continue | Moyenne bayésienne : estimateur régularisé avec prior empirique (prior_count=10, global_mean≈3.73)                    |
| `wilson_lb`      | Continue | Borne inférieure de l'intervalle de Wilson (α=0.05) : mesure conservatrice de la proportion de notes positives (≥4/5) |

**Justification statistique** :

- `bayes_mean` corrige le biais d'échantillonnage (shrinkage vers la moyenne globale)
- `wilson_lb` fournit un intervalle de confiance exact pour les proportions binomiales

### 2.2 Métriques d'écart

| Variable     | Type     | Description                                   |
| ------------ | -------- | --------------------------------------------- |
| `rating_gap` | Continue | Écart à la note maximale : 5 − avg_rating     |
| `bayes_gap`  | Continue | Écart bayésien : 5 − bayes_mean               |
| `wilson_gap` | Continue | Écart à la certitude maximale : 1 − wilson_lb |

### 2.3 Métriques d'engagement

| Variable                 | Type       | Description                                                      |
| ------------------------ | ---------- | ---------------------------------------------------------------- |
| `submitted`              | Temporelle | Date de soumission de la recette                                 |
| `age_months`             | Continue   | Ancienneté (mois écoulés depuis submission)                      |
| `interactions_per_month` | Continue   | Taux d'engagement temporel : n_interactions / max(1, age_months) |
| `n_unique_users`         | Discrète   | Cardinalité des utilisateurs ayant interagi                      |
| `n_reviews_text`         | Discrète   | Nombre de reviews avec contenu textuel                           |

### 2.4 Transformations pour la modélisation

| Variable                       | Type     | Transformation        | Justification                                             |
| ------------------------------ | -------- | --------------------- | --------------------------------------------------------- |
| `log1p_n_interactions`         | Continue | log(1 + x)            | Atténuation de l'asymétrie (skewness)                     |
| `log1p_interactions_per_month` | Continue | log(1 + x)            | Stabilisation de la variance                              |
| `log1p_n_unique_users`         | Continue | log(1 + x)            | Normalisation de la distribution                          |
| `log1p_*_w`                    | Continue | Winsorisation + log1p | Robustesse aux outliers (écrêtage aux percentiles 5%-99%) |

**Note** : Les versions winsorisées (`_w`) sont recommandées pour les analyses statistiques afin de limiter l'influence des valeurs extrêmes.

---

## 3. Sélection de Variables pour l'Analyse Bivariée

### 3.1 Variables d'effort culinaire (prédicteurs)

**Variables continues** :

- `log_minutes` : Temps de préparation (transformation log)
- `n_steps` : Nombre d'étapes
- `n_ingredients` : Nombre d'ingrédients
- `avg_words_per_step` : Complexité descriptive
- `effort_score` : Indice composite

**Variables catégorielles** :

- `effort_category` : Stratification en 5 niveaux
- `effort_quartile` : Découpage en quartiles
- `complexity` : Complexité procédurale
- `category_minutes` : Catégories de durée

### 3.2 Variables de popularité (variables réponse)

**Satisfaction** :

- `bayes_mean` : Estimateur régularisé (recommandé pour classement)
- `wilson_lb` : Mesure conservatrice de qualité
- `avg_rating` : Moyenne simple (baseline)
- `median_rating` : Mesure robuste

**Engagement** :

- `log1p_interactions_per_month_w` : Métrique principale (normalisée et robuste)
- `n_interactions` : Volume brut
- `n_unique_users` : Diversité de l'audience
- `log1p_n_interactions_w` : Volume transformé

**Variables de contrôle** :

- `age_months` : Effet temporel (obligatoire)
- `n_interactions` : Pondération par le volume

### 3.3 Stratégies d'analyse selon la distribution

| Méthode                       | Variables Recommandées                                                                  | Justification                                                 |
| ----------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| **Corrélation de Pearson**    | Variables transformées : `log_minutes`, `log1p_interactions_per_month_w`, versions `_w` | Hypothèse de linéarité et normalité                           |
| **Corrélation de Spearman**   | Variables brutes ou ordinales : `n_steps`, `n_ingredients`, `bayes_mean`                | Robuste aux distributions non-normales et relations monotones |
| **ANOVA / Kruskal-Wallis**    | Catégorielles : `effort_category`, `effort_quartile` vs variables continues             | Comparaison de groupes                                        |
| **Régression linéaire (OLS)** | Variables standardisées (suffixe `_std`) + interactions                                 | Modélisation paramétrique                                     |

### 3.4 Recommandations pour les visualisations

**Scatter plots** :

- `effort_score` × `bayes_mean` : Relation effort-satisfaction
- `log_minutes` × `wilson_lb` : Temps vs qualité perçue
- `n_steps` × `log1p_interactions_per_month_w` : Complexité vs engagement

**Box plots / Violin plots** :

- `effort_category` → distribution de `bayes_mean`
- `complexity` → distribution de `log1p_interactions_per_month_w`

**Heatmaps** :

- Matrice de corrélation : sous-ensemble de variables transformées
- Éviter les variables redondantes (ex: `log_minutes` et `log_minutes_w`)

---

## 4. Variables pour la Modélisation

### 4.1 Termes d'interaction

| Variable                  | Type     | Construction                                                       |
| ------------------------- | -------- | ------------------------------------------------------------------ |
| `steps_x_ingredients`     | Continue | Produit simple : n_steps × n_ingredients                           |
| `steps_x_ingredients_std` | Continue | Version standardisée (z-score) pour inclusion dans les modèles OLS |

**Justification** : Capture l'effet non-linéaire de la complexité combinée (étapes + ingrédients).

### 4.2 Variables standardisées

Les variables suivantes sont disponibles en version centrée-réduite (suffixe `_std`) pour la modélisation linéaire :

```
log_minutes_std
n_steps_std
n_ingredients_std
avg_words_per_step_std
age_months_std
effort_score_std
steps_x_ingredients_std
```

**Transformation appliquée** : z-score = (x - μ) / σ

**Avantages** :

- Interprétation en termes d'écarts-types
- Comparabilité directe des coefficients
- Stabilité numérique des algorithmes d'optimisation

### 4.3 Structure du dataset final

Le fichier `analysis_dataset.csv` contient :

- **Variables brutes** : `minutes`, `n_steps`, `n_ingredients`, etc.
- **Variables transformées** : `log_*`, `log1p_*`
- **Variables robustes** : `*_w` (winsorisées)
- **Variables normalisées** : `*_scaled` (min-max)
- **Variables standardisées** : `*_std` (z-score)
- **Variables catégorielles** : `*_category`, `effort_quartile`

---

## 5. Recommandations Méthodologiques

### 5.1 Choix des métriques selon l'objectif

| Objectif d'analyse           | Variables recommandées           | Justification                                                     |
| ---------------------------- | -------------------------------- | ----------------------------------------------------------------- |
| **Satisfaction utilisateur** | `bayes_mean`                     | Correction du biais d'échantillonnage via shrinkage bayésien      |
| **Qualité robuste**          | `wilson_lb`                      | Borne inférieure conservatrice avec garantie statistique (α=0.05) |
| **Engagement normalisé**     | `log1p_interactions_per_month_w` | Neutralise l'effet d'ancienneté, robuste aux outliers             |
| **Volume brut**              | `n_interactions`                 | Métrique simple sans transformation                               |

### 5.2 Précautions méthodologiques

**Variables de contrôle obligatoires** :

- `age_months` : Effet de cohorte temporelle (recettes anciennes vs récentes)
- `n_interactions` : Pondération ou filtrage par volume minimum (ex: n ≥ 10)

**Choix du test de corrélation** :

- **Pearson** : Variables transformées (`log`, `log1p`, `_w`) sous hypothèse de linéarité
- **Spearman** : Variables ordinales ou distributions non-gaussiennes (robust rank-based test)

**Gestion des outliers** :

- Privilégier les versions winsorisées (`_w`) pour les analyses statistiques
- Conserver les versions brutes pour l'exploration descriptive

### 5.3 Considérations pour la modélisation

**Modèle linéaire (OLS)** :

```python
y = β₀ + β₁·log_minutes_std + β₂·n_steps_std + β₃·n_ingredients_std
    + β₄·steps_x_ingredients_std + β₅·age_months_std + ε
```

**Spécifications alternatives** :

- Modèle polynomial : ajouter `effort_score²` pour capturer la non-linéarité
- Modèle avec effets catégoriels : remplacer les continues par `effort_category` (dummy encoding)
- Modèle avec pondération : utiliser `n_interactions` comme poids analytiques

---

## 6. Correspondance Variables Techniques ↔ Interface Utilisateur

Pour l'implémentation dans une application web, mapping recommandé :

| Variable Technique               | Label Interface          | Type d'affichage                    |
| -------------------------------- | ------------------------ | ----------------------------------- |
| `bayes_mean`                     | Satisfaction générale    | Jauge (0-5)                         |
| `wilson_lb`                      | Qualité fiable           | Barre de confiance (0-1)            |
| `log1p_interactions_per_month_w` | Engagement               | Score relatif (percentile)          |
| `effort_score`                   | Score d'effort culinaire | Jauge (0-100)                       |
| `log_minutes`                    | Temps de préparation     | Affichage en minutes (détransformé) |
| `n_steps`                        | Nombre d'étapes          | Entier                              |
| `n_ingredients`                  | Nombre d'ingrédients     | Entier                              |
| `n_interactions`                 | Nombre d'évaluations     | Compteur                            |
| `interactions_per_month`         | Interactions mensuelles  | Taux (décimales)                    |
| `n_unique_users`                 | Utilisateurs uniques     | Compteur                            |
| `age_months`                     | Ancienneté               | Mois (arrondi)                      |
| `complexity`                     | Niveau de complexité     | Badge catégoriel                    |
| `category_minutes`               | Catégorie de durée       | Badge catégoriel                    |

---

## Synthèse

### Variables clés pour l'analyse bivariée effort ↔ popularité

**Prédicteurs (effort)** :

- Continues : `log_minutes`, `n_steps`, `n_ingredients`, `effort_score`
- Catégorielles : `effort_category`, `effort_quartile`

**Variables réponse (popularité)** :

- Satisfaction : `bayes_mean` (principale), `wilson_lb` (robuste)
- Engagement : `log1p_interactions_per_month_w` (principale), `n_interactions` (volume)

**Variables de contrôle** :

- `age_months` (obligatoire)
- `n_interactions` (pondération/filtrage)

### Workflow d'analyse type

1. **Exploration descriptive** : Variables brutes + catégorielles
2. **Analyse de corrélation** : Variables transformées (`log`, `log1p_*_w`)
3. **Tests d'hypothèses** : Comparaisons de groupes (`effort_category`)
4. **Modélisation** : Variables standardisées (`*_std`) + interactions
