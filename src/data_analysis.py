"""
Module d'analyse des données pour le projet MangesTaMain.
Basé sur l'analyse exploratoire du notebook lab-recipe-corr-student.ipynb
"""

import pandas as pd
import numpy as np
import ast
from collections import Counter
from typing import List, Tuple, Dict, Optional


class RecipeDataAnalyzer:
    """Classe pour analyser les données de recettes."""
    
    def __init__(self, csv_path: str) -> None:
        """
        Initialise l'analyseur avec le chemin vers le fichier CSV.
        
        Args:
            csv_path (str): Chemin vers le fichier CSV des recettes
        """
        self.csv_path = csv_path
        self.recipe_data = None
        self.cleaned_data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Charge les données depuis le fichier CSV.
        
        Returns:
            pd.DataFrame: Les données chargées
        """
        self.recipe_data = pd.read_csv(self.csv_path)
        return self.recipe_data
    
    def get_basic_info(self) -> Dict[str, any]:
        """
        Retourne des informations de base sur le dataset.
        
        Returns:
            Dict: Informations de base (nombre de recettes, variables, valeurs manquantes)
        """
        if self.recipe_data is None:
            raise ValueError("Les données doivent être chargées d'abord avec load_data()")
            
        info = {
            'n_recipes': self.recipe_data.shape[0],
            'n_variables': self.recipe_data.shape[1],
            'total_missing_values': self.recipe_data.isnull().sum().sum(),
            'missing_by_column': self.recipe_data.isnull().sum().to_dict(),
            'columns': list(self.recipe_data.columns)
        }
        return info
    
    def analyze_minutes(self) -> Dict[str, any]:
        """
        Analyse la variable 'minutes' (temps de préparation).
        
        Returns:
            Dict: Statistiques sur les minutes
        """
        if self.recipe_data is None:
            raise ValueError("Les données doivent être chargées d'abord avec load_data()")
            
        minutes_stats = {
            'mean': self.recipe_data['minutes'].mean(),
            'median': self.recipe_data['minutes'].median(),
            'min': self.recipe_data['minutes'].min(),
            'max': self.recipe_data['minutes'].max(),
            'q1': np.percentile(self.recipe_data['minutes'], 25),
            'q3': np.percentile(self.recipe_data['minutes'], 75),
            'std': self.recipe_data['minutes'].std()
        }
        return minutes_stats
    
    def remove_outliers_minutes(self, max_minutes: Optional[int] = None) -> pd.DataFrame:
        """
        Supprime les outliers de la variable minutes.
        
        Args:
            max_minutes (int, optional): Temps maximum en minutes. 
                                       Par défaut: 30*24*60 (1 mois)
        
        Returns:
            pd.DataFrame: Données nettoyées
        """
        if self.recipe_data is None:
            raise ValueError("Les données doivent être chargées d'abord avec load_data()")
            
        if max_minutes is None:
            max_minutes = 30 * 24 * 60  # 1 mois en minutes
            
        # Copie des données
        cleaned = self.recipe_data.copy()
        
        # Nombre initial
        initial_count = len(cleaned)
        
        # Supprimer les valeurs extrêmes (max outliers)
        cleaned = cleaned[cleaned['minutes'] != cleaned['minutes'].max()]
        
        # Supprimer les recettes > 1 mois
        cleaned = cleaned[cleaned['minutes'] <= max_minutes]
        
        # Optionnel: supprimer les 0 minutes
        # cleaned = cleaned[cleaned['minutes'] > 0]
        
        final_count = len(cleaned)
        removed_count = initial_count - final_count
        
        print(f"Supprimé {removed_count} observations considérées comme outliers")
        
        self.cleaned_data = cleaned
        return cleaned
    
    def analyze_contributors(self) -> Dict[str, any]:
        """
        Analyse les contributeurs de recettes.
        
        Returns:
            Dict: Statistiques sur les contributeurs
        """
        data = self.cleaned_data if self.cleaned_data is not None else self.recipe_data
        if data is None:
            raise ValueError("Aucune donnée disponible")
            
        contributor_counts = data['contributor_id'].value_counts()
        
        stats = {
            'n_unique_contributors': len(data['contributor_id'].unique()),
            'top_contributor_id': contributor_counts.idxmax(),
            'top_contributor_recipes': contributor_counts.max(),
            'top_contributor_percentage': round(contributor_counts.max() / len(data) * 100, 2),
            'avg_recipes_per_contributor': round(contributor_counts.mean()),
            'median_recipes_per_contributor': contributor_counts.median()
        }
        return stats
    
    def parse_list_column(self, column_name: str) -> List[str]:
        """
        Parse une colonne contenant des listes sous forme de strings.
        
        Args:
            column_name (str): Nom de la colonne à parser
            
        Returns:
            List[str]: Liste de tous les éléments extraits
        """
        data = self.cleaned_data if self.cleaned_data is not None else self.recipe_data
        if data is None:
            raise ValueError("Aucune donnée disponible")
            
        if column_name not in data.columns:
            raise ValueError(f"Colonne '{column_name}' non trouvée")
            
        all_items = []
        for item_list_str in data[column_name]:
            try:
                item_list = ast.literal_eval(item_list_str)
                if isinstance(item_list, list):
                    all_items.extend(item_list)
            except (ValueError, SyntaxError):
                # Ignorer les valeurs qui ne peuvent pas être parsées
                continue
                
        return all_items
    
    def analyze_ingredients(self) -> Dict[str, any]:
        """
        Analyse les ingrédients des recettes.
        
        Returns:
            Dict: Statistiques sur les ingrédients
        """
        all_ingredients = self.parse_list_column('ingredients')
        ingredient_counts = Counter(all_ingredients)
        
        stats = {
            'n_unique_ingredients': len(set(all_ingredients)),
            'total_ingredient_mentions': len(all_ingredients),
            'top_10_ingredients': ingredient_counts.most_common(10),
            'least_common_ingredients': ingredient_counts.most_common()[-10:]
        }
        return stats
    
    def analyze_tags(self) -> Dict[str, any]:
        """
        Analyse les tags des recettes.
        
        Returns:
            Dict: Statistiques sur les tags
        """
        all_tags = self.parse_list_column('tags')
        tag_counts = Counter(all_tags)
        
        stats = {
            'n_unique_tags': len(set(all_tags)),
            'total_tag_mentions': len(all_tags),
            'top_20_tags': tag_counts.most_common(20),
            'least_common_tags': tag_counts.most_common()[-10:]
        }
        return stats
    
    def process_nutrition_scores(self) -> pd.DataFrame:
        """
        Traite les scores nutritionnels en colonnes séparées.
        
        Returns:
            pd.DataFrame: Données avec colonnes nutritionnelles séparées
        """
        data = self.cleaned_data if self.cleaned_data is not None else self.recipe_data
        if data is None:
            raise ValueError("Aucune donnée disponible")
            
        # Copie des données
        processed_data = data.copy()
        
        # Colonnes nutritionnelles
        nutrition_columns = [
            'calories', 'total_fat_pct', 'sugar_pct', 
            'sodium_pct', 'protein_pct', 'saturated_fat_pct', 'carbohydrates_pct'
        ]
        
        # Séparer la colonne nutrition
        nutrition_split = processed_data['nutrition'].str.split(',', expand=True)
        
        if nutrition_split.shape[1] >= len(nutrition_columns):
            for i, col_name in enumerate(nutrition_columns):
                processed_data[col_name] = nutrition_split[i]
                
            # Nettoyer les crochets
            processed_data['calories'] = processed_data['calories'].str.replace('[', '', regex=False)
            processed_data['carbohydrates_pct'] = processed_data['carbohydrates_pct'].str.replace(']', '', regex=False)
            
            # Convertir en float
            for col in nutrition_columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
        
        return processed_data
    
    def analyze_nutrition(self) -> Dict[str, any]:
        """
        Analyse les données nutritionnelles.
        
        Returns:
            Dict: Statistiques nutritionnelles
        """
        nutrition_data = self.process_nutrition_scores()
        
        nutrition_columns = [
            'calories', 'total_fat_pct', 'sugar_pct', 
            'sodium_pct', 'protein_pct', 'saturated_fat_pct', 'carbohydrates_pct'
        ]
        
        stats = {}
        for col in nutrition_columns:
            if col in nutrition_data.columns:
                stats[col] = {
                    'mean': nutrition_data[col].mean(),
                    'median': nutrition_data[col].median(),
                    'min': nutrition_data[col].min(),
                    'max': nutrition_data[col].max(),
                    'std': nutrition_data[col].std()
                }
        
        return stats
    
    def analyze_steps_and_ingredients_count(self) -> Dict[str, any]:
        """
        Analyse le nombre d'étapes et d'ingrédients.
        
        Returns:
            Dict: Statistiques sur n_steps et n_ingredients
        """
        data = self.cleaned_data if self.cleaned_data is not None else self.recipe_data
        if data is None:
            raise ValueError("Aucune donnée disponible")
            
        stats = {
            'n_steps': {
                'mean': data['n_steps'].mean(),
                'median': data['n_steps'].median(),
                'min': data['n_steps'].min(),
                'max': data['n_steps'].max(),
                'mode': data['n_steps'].mode().iloc[0] if not data['n_steps'].mode().empty else None
            },
            'n_ingredients': {
                'mean': data['n_ingredients'].mean(),
                'median': data['n_ingredients'].median(),
                'min': data['n_ingredients'].min(),
                'max': data['n_ingredients'].max(),
                'mode': data['n_ingredients'].mode().iloc[0] if not data['n_ingredients'].mode().empty else None
            }
        }
        return stats
    
    def get_complete_analysis(self) -> Dict[str, any]:
        """
        Effectue une analyse complète du dataset.
        
        Returns:
            Dict: Analyse complète
        """
        # Charger les données si pas déjà fait
        if self.recipe_data is None:
            self.load_data()
            
        analysis = {
            'basic_info': self.get_basic_info(),
            'minutes_analysis': self.analyze_minutes(),
            'contributors_analysis': self.analyze_contributors(),
            'ingredients_analysis': self.analyze_ingredients(),
            'tags_analysis': self.analyze_tags(),
            'steps_ingredients_count': self.analyze_steps_and_ingredients_count(),
            'nutrition_analysis': self.analyze_nutrition()
        }
        
        return analysis


def main():
    """Fonction principale pour tester l'analyseur."""
    analyzer = RecipeDataAnalyzer('data/RAW_recipes.csv')
    analyzer.load_data()
    
    print("=== ANALYSE COMPLETE DU DATASET ===")
    
    # Informations de base
    basic_info = analyzer.get_basic_info()
    print(f"\nNombre de recettes: {basic_info['n_recipes']:,}")
    print(f"Nombre de variables: {basic_info['n_variables']}")
    print(f"Valeurs manquantes totales: {basic_info['total_missing_values']:,}")
    
    # Analyse des minutes
    print("\n=== ANALYSE DES MINUTES ===")
    minutes_stats = analyzer.analyze_minutes()
    print(f"Temps moyen: {minutes_stats['mean']:.2f} minutes")
    print(f"Temps médian: {minutes_stats['median']:.2f} minutes")
    print(f"Min: {minutes_stats['min']} - Max: {minutes_stats['max']:,}")
    
    # Nettoyage des outliers
    print("\n=== NETTOYAGE DES OUTLIERS ===")
    analyzer.remove_outliers_minutes()
    
    # Analyse des contributeurs
    print("\n=== ANALYSE DES CONTRIBUTEURS ===")
    contributors_stats = analyzer.analyze_contributors()
    print(f"Contributeurs uniques: {contributors_stats['n_unique_contributors']:,}")
    print(f"Top contributeur: {contributors_stats['top_contributor_recipes']} recettes")
    
    # Analyse des ingrédients
    print("\n=== ANALYSE DES INGREDIENTS ===")
    ingredients_stats = analyzer.analyze_ingredients()
    print(f"Ingrédients uniques: {ingredients_stats['n_unique_ingredients']:,}")
    print("Top 5 ingrédients:")
    for ingredient, count in ingredients_stats['top_10_ingredients'][:5]:
        print(f"  - {ingredient}: {count:,}")
    
    # Analyse des tags
    print("\n=== ANALYSE DES TAGS ===")
    tags_stats = analyzer.analyze_tags()
    print(f"Tags uniques: {tags_stats['n_unique_tags']:,}")
    print("Top 5 tags:")
    for tag, count in tags_stats['top_20_tags'][:5]:
        print(f"  - {tag}: {count:,}")


if __name__ == "__main__":
    main()