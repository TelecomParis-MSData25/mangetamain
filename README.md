# mangetamain

## Problématique

**Comment l’effort culinaire influence-t-il la popularité des recettes ?**

---

## Développement et déploiement

### 🐳 Utilisation avec Docker

Le projet peut être containerisé et exécuté avec Docker pour faciliter le déploiement et assurer la portabilité.

#### Construction de l'image Docker

```bash
docker build -t mangetamain .
```

Cette commande :

- Lit le `Dockerfile` à la racine du projet
- Installe toutes les dépendances nécessaires
- Configure l'environnement Python
- Prépare l'application Streamlit

#### Lancement du conteneur

```bash
docker run -p 8501:8501 mangetamain
```

Cette commande :

- Démarre un conteneur basé sur l'image `mangetamain`
- Map le port 8501 du conteneur vers le port 8501 de l'hôte
- Rend l'application accessible à l'adresse : `http://localhost:8501`

#### Options avancées

```bash
# Lancer en arrière-plan (mode détaché)
docker run -d -p 8501:8501 --name mangetamain-app mangetamain

# Monter un volume pour les données
docker run -p 8501:8501 -v $(pwd)/data:/app/data mangetamain

# Arrêter le conteneur
docker stop mangetamain-app

# Supprimer le conteneur
docker rm mangetamain-app
```

### 🧪 Tests unitaires

Le projet inclut une suite complète de tests unitaires et d'intégration pour valider le fonctionnement du module d'analyse des données.

#### Lancement des tests

**Tests unitaires uniquement :**

```bash
uv run pytest tests/test_data_analysis.py -v
```

**Tests d'intégration :**

```bash
uv run pytest tests/test_integration.py -v
```

**Tous les tests :**

```bash
uv run pytest -v
```

#### Tests avec couverture de code

```bash
# Rapport de couverture détaillé
uv run pytest --cov=src --cov-report=term-missing

# Générer un rapport HTML
uv run pytest --cov=src --cov-report=html
```

#### Types de tests disponibles

- **Tests unitaires** (`test_data_analysis.py`) : 17 tests avec données simulées
- **Tests d'intégration** (`test_integration.py`) : 11 tests avec le vrai dataset
- **Tests marqués** :
  - `@pytest.mark.unit` : Tests unitaires rapides
  - `@pytest.mark.integration` : Tests d'intégration
  - `@pytest.mark.slow` : Tests plus longs

#### Exécution sélective des tests

```bash
# Exclure les tests lents
uv run pytest -m "not slow"

# Seulement les tests unitaires
uv run pytest -m "unit"

# Seulement les tests d'intégration
uv run pytest -m "integration"
```

#### Structure des tests

```
tests/
├── __init__.py
├── test_data_analysis.py      # Tests unitaires avec données simulées
└── test_integration.py        # Tests d'intégration avec vrai dataset
```

Les tests couvrent :

- ✅ Chargement et validation des données
- ✅ Analyse des temps de préparation et gestion des outliers
- ✅ Analyse des contributeurs et leur productivité
- ✅ Parsing et analyse des ingrédients et tags
- ✅ Traitement des scores nutritionnels
- ✅ Gestion des cas limites et erreurs

### 📚 Documentation avec Sphinx

Le projet utilise Sphinx pour générer une documentation API complète et professionnelle du code source.

#### Structure de la documentation

```
docs/
├── build/           # Documentation générée
│   └── html/        # Version HTML de la documentation
├── source/          # Sources de la documentation
│   ├── conf.py      # Configuration Sphinx
│   ├── index.rst    # Page d'accueil
│   └── modules.rst  # Documentation des modules
├── Makefile         # Commandes de build (Unix)
└── make.bat         # Commandes de build (Windows)
```

#### Génération de la documentation

**Générer automatiquement la documentation API :**

```bash
cd docs
uv run sphinx-apidoc -o source ../src --force
```

**Construire la documentation HTML :**

```bash
cd docs
uv run sphinx-build -b html source build/html
```

**Ou utiliser le Makefile :**

```bash
cd docs
make html
```

#### Consultation de la documentation

Une fois générée, la documentation est accessible via :

- **Fichier local** : `docs/build/html/index.html`
- **Serveur local** : Ouvrir le fichier dans un navigateur

#### Fonctionnalités de la documentation

- **API complète** : Documentation automatique de toutes les classes et fonctions
- **Docstrings** : Extraction automatique des docstrings Python
- **Navigation** : Index des modules, classes et fonctions
- **Recherche** : Moteur de recherche intégré
- **Thème professionnel** : Interface claire et responsive

#### Mise à jour de la documentation

```bash
# Regénérer complètement la documentation
cd docs
uv run sphinx-apidoc -o source ../src --force
make clean
make html
```

La documentation Sphinx est particulièrement utile pour :

- Comprendre l'architecture du code
- Explorer les APIs disponibles
- Intégrer le module dans d'autres projets
- Maintenir une documentation à jour automatiquement

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
