#!/usr/bin/env bash
set -euo pipefail

# Télécharge et décompresse le dataset Food.com depuis Kaggle.
# Utilise les variables d'environnement KAGGLE_USERNAME et KAGGLE_KEY.

DATA_DIR=${1:-data}
TARGET_FILE="$DATA_DIR/RAW_recipes.csv"
DATASET_SLUG="shuyangli94/food-com-recipes-and-user-interactions"

if [[ -f "$TARGET_FILE" ]]; then
  echo "Dataset déjà présent à $TARGET_FILE, téléchargement ignoré."
  exit 0
fi

if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
  echo "Identifiants Kaggle manquants, impossible de télécharger le dataset."
  echo "Définir KAGGLE_USERNAME et KAGGLE_KEY pour activer le téléchargement."
  exit 0
fi

if ! command -v kaggle >/dev/null 2>&1; then
  echo "La commande 'kaggle' est introuvable. Installez le Kaggle CLI avant d'exécuter ce script."
  exit 1
fi

mkdir -p "$DATA_DIR"
echo "Téléchargement du dataset $DATASET_SLUG vers $DATA_DIR..."
kaggle datasets download "$DATASET_SLUG" --path "$DATA_DIR" --force

ZIP_FILE=$(find "$DATA_DIR" -maxdepth 1 -type f -name "*.zip" -print -quit)

if [[ -z "$ZIP_FILE" ]]; then
  echo "Aucun fichier ZIP téléchargé. Vérifiez vos identifiants Kaggle."
  exit 1
fi

echo "Décompression de $ZIP_FILE..."
unzip -o "$ZIP_FILE" -d "$DATA_DIR" >/dev/null
rm -f "$ZIP_FILE"

if [[ -f "$TARGET_FILE" ]]; then
  echo "Dataset disponible : $TARGET_FILE"
else
  echo "La décompression n'a pas produit $TARGET_FILE. Contenu du dossier $DATA_DIR :"
  ls -al "$DATA_DIR"
  exit 1
fi
