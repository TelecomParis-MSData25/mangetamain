# Chaîne CI/CD GitHub Actions

Ce document décrit l’architecture d’intégration et déploiement continus du projet **mangetamain**. Les workflows sont fortement inspirés du dépôt pédagogique de Martin Prillard (`try_python`) et adaptés à notre stack (Streamlit + Sphinx + uv).

## Aperçu rapide

- **Automatisation** : GitHub Actions orchestrent tests, documentation, image Docker et audit sécurité.
- **Gestion des dépendances** : [uv](https://github.com/astral-sh/uv) synchronise l’environnement à partir de `pyproject.toml` et `uv.lock`.
- **Documentation** : la doc Sphinx est compilée depuis `docs/source` et publiée sur GitHub Pages.
- **Conteneurisation** : l’image Streamlit est construite à partir du `Dockerfile` et poussée sur GitHub Container Registry (GHCR).
- **Données** : le dataset Food.com est téléchargé à la volée via le Kaggle CLI (`scripts/download_dataset.sh`) avant les tests d’intégration.

## Workflows disponibles

| Fichier | Rôle principal |
|---------|----------------|
| `.github/workflows/ci.yml` | Pipeline complet (PR/push, doc, Docker, scan Trivy). |
| `.github/workflows/dependencies.yml` | Mise à jour hebdomadaire du lockfile + audit sécurité (Safety, Bandit). |
| `.github/environments/production.yml` | Règles de protection pour l’environnement `production`. |

### `ci.yml`

- **PR Checks** : `uv sync --dev`, exécution de `pytest` (rapport XML + couverture) et compilation Sphinx.
- **Dataset Kaggle** : installation du Kaggle CLI, création de `~/.kaggle/kaggle.json` (à partir des secrets) puis exécution de `scripts/download_dataset.sh` qui télécharge `RAW_recipes.csv` dans `data/`.
- **Push Pipeline** : tests avec couverture minimale (80 %), artefacts (`htmlcov/`, `coverage.xml`, `pytest-report.xml`).
- **Documentation** : build Sphinx (HTML) puis déploiement via `peaceiris/actions-gh-pages@v3`.
- **Docker** : build multi-arch (`linux/amd64`, `linux/arm64`) avec `docker/build-push-action@v5` et publication sur GHCR (`ghcr.io/<owner>/<repo>`).
- **Sécurité** : scan Trivy du workspace et export en SARIF dans l’onglet **Security** de GitHub.
- **Dockerfile** : pendant la construction de l’image, `uv run pip install kaggle && bash scripts/download_dataset.sh` télécharge les données si les build args `KAGGLE_USERNAME` / `KAGGLE_KEY` sont fournis (`docker build --build-arg KAGGLE_USERNAME=... --build-arg KAGGLE_KEY=...`).

### `dependencies.yml`

- Job planifié (cron lundi 09h UTC) ou manuel.
- Met à jour l’environnement (`uv sync --dev`), régénère `uv.lock`, ouvre une PR automatique en cas de diff.
- Lance `safety` et `bandit` (répertoires `src/`), stocke les rapports en artefacts.

### Environnement `production`

- Défini dans `.github/environments/production.yml`.
- Implique un reviewer obligatoire et cible la branche `main`.
- Permet d’ajouter ultérieurement des secrets (ex. clés Render) ou des URL de production visibles dans les déploiements GitHub.

## Pré-requis GitHub

1. **Créer le dossier `.github/`** dans le dépôt (déjà versionné avec ce template).
2. **Activer GitHub Pages** : dans `Settings > Pages`, sélectionner “GitHub Actions” comme source.
3. **Autoriser `GITHUB_TOKEN`** : `Settings > Actions > General > Workflow permissions` → activer “Read and write permissions”.
4. **Autoriser GHCR** (si besoin) : `Settings > Packages` → vérifier que l’organisation accepte la publication de packages.
5. **Déployer les workflows** : pousser les fichiers sur la branche `main` (ou ouvrir une PR) pour déclencher la première exécution.

## Variables & secrets

| Usage | Clé | Emplacement | Commentaire |
|-------|-----|-------------|-------------|
| GHCR (auth par défaut) | `GITHUB_TOKEN` | Automatique | Permissions “read/write” nécessaires. |
| Déploiement externe (ex. Render) | `RENDER_SERVICE_ID`, `RENDER_API_KEY` | `Settings > Secrets and variables > Actions` | Ajouter lors de l’intégration d’un job de déploiement. |
| Accès Kaggle | `KAGGLE_USERNAME`, `KAGGLE_KEY` | `Settings > Secrets and variables > Actions` | Requis pour télécharger les données Food.com. |

## Vérifications locales recommandées

```bash
uv sync --dev
uv run pytest
uv run sphinx-build -b html docs/source docs/build/html
uv run streamlit run src/webapp.py  # Optionnel pour test manuel
```

## Dépannage courant

- **`docker buildx` manquant en local** : installer le plugin (`docker buildx install`). GitHub Actions gère l’installation via `docker/setup-buildx-action`.
- **Build Sphinx lent ou incomplet** : nettoyer `docs/build/` (`make clean` ou suppression manuelle) avant un nouveau build.
- **Erreurs Trivy/Bandit** : les actions sont configurées avec `|| true`, elles n’échouent pas le pipeline mais produisent un rapport à consulter.

## Évolutions possibles

- Ajouter un job de déploiement Render ou Streamlit Cloud (réutiliser la structure du dépôt de référence).
- Activer Codecov/org-mode pour exploiter `coverage.xml`.
- Étendre les scans sécurité (dependabot, `pip-audit`, etc.).

---

Pour toute modification de pipeline, mettre à jour les workflows correspondants et ajuster ce document afin de conserver la documentation alignée avec l’infrastructure réelle.
Pour un téléchargement manuel des données :

```bash
export KAGGLE_USERNAME=...
export KAGGLE_KEY=...
uv run pip install kaggle  # ou pip install kaggle
bash scripts/download_dataset.sh
```
- Pour inclure les données dans l’image Docker : `docker build --build-arg KAGGLE_USERNAME=... --build-arg KAGGLE_KEY=... -t mangetamain .`
