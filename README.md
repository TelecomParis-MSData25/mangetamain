# mangetamain

## Problématique

**Comment l’effort culinaire influence-t-il la popularité des recettes ?**

---

## A. Définitions

### 1. Popularité

La popularité d'une recette peut être mesurée selon deux axes :

- **Satisfaction** :  
    - Définie par la note moyenne attribuée (`avg_rating`), la médiane (`median_rating`), et l'écart-type (`rating_std`).
    - À considérer uniquement si le nombre de notes (`n_ratings`) est suffisant par rapport au nombre de reviews (définir un ratio minimal).
    - Si `rating = 0`, aucune note n’a été donnée, mais il peut y avoir des reviews.

- **Engagement** :  
    - Mesuré par le nombre de reviews (`reviews`), variable permettant de définir `n_interactions` en comptant le nombre de lignes existantes `reviews`/ recette.
    - Plus il y a de reviews, plus la recette est considérée comme populaire (dcp indépendamment de la note).

### 2. Effort culinaire

L’effort requis pour une recette est estimé à partir de :

- Le nombre d’étapes (`n_steps`) : plus il est élevé, plus l’effort est grand.
- Le nombre d’ingrédients (`n_ingredients`) : possibilité d’exclure les ingrédients considérés comme accessoires (condiments, etc.).
- Le temps de préparation (`log_minutes`) : temps total -transformé en logarithme pour réduire l’asymétrie-.

---

## B. Variables

### A. Effort (X)

- `log_minutes` : temps de préparation (log-transformé)
- `n_steps` : nombre d’étapes
- `n_ingredients` : nombre d’ingrédients

### B. Popularité

Deux dimensions complémentaires :

#### 1. Satisfaction (Y₁)

- `n_ratings` : nombre de notes
- `avg_rating` : moyenne des notes
- `median_rating` : médiane des notes
- `rating_std` : écart-type des notes

**Variables à créer :**
- `bayes_mean` : moyenne bayésienne (corrige les petits échantillons)
- `wilson_lb` : borne inférieure de l’intervalle de Wilson (notes positives ≥4/5)

#### 2. Engagement (Y₂)

- `n_interactions` : total des interactions
- `n_reviews_text` : nombre de reviews avec texte
- `n_users` : utilisateurs uniques
- `age_months` : âge de la recette (en mois)
- `interactions_per_month` : interactions normalisées par l’âge (pour comparer recettes anciennes et récentes)

---