from __future__ import annotations

import argparse
import os
import sys
import errno
from pathlib import Path
from typing import Optional, Sequence

from kaggle.api.kaggle_api_extended import KaggleApi

MARKERS = {"pyproject.toml", ".git", "README.md", "Dockerfile"}
KNOWN_FILES = (
    "RAW_recipes.csv",
    "RAW_interactions.csv",
    "PP_recipes.csv",
    "PP_users.csv",
    "ingr_map.pkl",
    "interactions_train.csv",
    "interactions_validation.csv",
    "interactions_test.csv",
)
DATASET_REF = "shuyangli94/food-com-recipes-and-user-interactions"


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Remonte l'arborescence pour trouver la racine du dépôt."""
    p = (start or Path.cwd()).resolve()
    for parent in [p] + list(p.parents):
        if any((parent / m).exists() for m in MARKERS):
            return parent
    return p  # fallback: courant si rien trouvé


def resolve_target_directory(target: str) -> Path:
    """Résout le dossier cible relativement à la racine du dépôt."""
    path = Path(target)
    if not path.is_absolute():
        path = find_repo_root() / path
    return safe_mkdir(path)


def safe_mkdir(path: Path) -> Path:
    """Crée un dossier si nécessaire, avec fallback en lecture seule."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError as e:
        if e.errno == errno.EROFS:
            home_fallback = Path.home() / "mangetamain_data"
            home_fallback.mkdir(parents=True, exist_ok=True)
            print(
                "⚠️  Système de fichiers en lecture seule. "
                f"Basculé vers: {home_fallback}"
            )
            return home_fallback
        raise


def has_kaggle_credentials() -> bool:
    """Vérifie la présence des identifiants Kaggle."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def check_expected_files(data_dir: Path) -> tuple[bool, list[str]]:
    """Retourne l'état de présence des fichiers attendus."""
    missing = [name for name in KNOWN_FILES if not (data_dir / name).exists()]
    return (len(missing) == 0, missing)


def download_and_extract(target_dir: Path, force: bool = False) -> None:
    """Télécharge et extrait le dataset si nécessaire."""
    present, missing = check_expected_files(target_dir)

    print(f"Téléchargement vers : {target_dir}")

    if present and not force:
        print("Datasets déjà présents, téléchargement ignoré.")
        return

    if not has_kaggle_credentials():
        print("Identifiants Kaggle manquants, impossible de télécharger le dataset.")
        print("Définir KAGGLE_USERNAME et KAGGLE_KEY pour activer le téléchargement.")
        return

    if missing:
        print("Fichiers manquants détectés :")
        for name in missing:
            print(f" - {name}")

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        DATASET_REF,
        path=str(target_dir),
        unzip=True,
        force=force or bool(missing),
    )

    present, missing = check_expected_files(target_dir)
    if present:
        print(f"✅ Tous les datasets sont disponibles dans {target_dir}")
    else:
        print(
            "Téléchargement terminé mais certains fichiers sont introuvables. "
            f"Contenu du dossier {target_dir}:"
        )
        for child in sorted(target_dir.iterdir()):
            print(f" - {child.name}")
        missing_list = ', '.join(missing)
        raise FileNotFoundError(f"Fichiers manquants après téléchargement: {missing_list}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Télécharge l'ensemble des datasets Food.com.")
    parser.add_argument(
        "--target",
        default="data",
        help="Dossier cible relatif à la racine du dépôt (ou chemin absolu).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force le téléchargement même si les fichiers sont présents.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    target_dir = resolve_target_directory(args.target)
    download_and_extract(target_dir, force=args.force)


if __name__ == "__main__":
    main(sys.argv[1:])
