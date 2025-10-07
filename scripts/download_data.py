from pathlib import Path
import os
import errno
from kaggle.api.kaggle_api_extended import KaggleApi

MARKERS = {"pyproject.toml", ".git", "README.md", "Dockerfile"}

def find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for parent in [p] + list(p.parents):
        if any((parent / m).exists() for m in MARKERS):
            return parent
    return p  # fallback: courant si rien trouvé

def safe_mkdir(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError as e:
        if e.errno == errno.EROFS:  # read-only FS
            home_fallback = Path.home() / "mangetamain_data"
            home_fallback.mkdir(parents=True, exist_ok=True)
            print(f"⚠️  Système de fichiers en lecture seule. "
                  f"Basculé vers: {home_fallback}")
            return home_fallback
        raise

def download_and_extract():
    dataset_ref = "shuyangli94/food-com-recipes-and-user-interactions"

    repo_root = find_repo_root()
    data_dir = safe_mkdir(repo_root / "data")

    print(f"Racine projet : {repo_root}")
    print(f"Téléchargement vers : {data_dir}")

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(dataset_ref, path=str(data_dir), unzip=True)
    print("✅ Téléchargement terminé.")

if __name__ == "__main__":
    download_and_extract()
