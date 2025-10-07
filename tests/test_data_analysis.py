"""
Tests unitaires pour le module data_analysis.py
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.data_analysis import RecipeDataAnalyzer


class TestRecipeDataAnalyzer:
    """Tests pour la classe RecipeDataAnalyzer."""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Crée un fichier CSV temporaire avec des données de test."""
        data = {
            'name': ['Recipe1', 'Recipe2', 'Recipe3', 'Recipe4'],
            'id': [1, 2, 3, 4],
            'minutes': [30, 60, 0, 432000],  # Inclut des outliers
            'contributor_id': [101, 102, 101, 103],
            'submitted': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'],
            'tags': [
                "['vegetarian', 'quick']",
                "['meat', 'slow']", 
                "['dessert']",
                "['vegetarian', 'dessert']"
            ],
            'nutrition': [
                '[100.0, 5.0, 10.0, 2.0, 15.0, 3.0, 20.0]',
                '[200.0, 10.0, 15.0, 5.0, 20.0, 5.0, 25.0]',
                '[150.0, 7.0, 20.0, 3.0, 10.0, 4.0, 30.0]',
                '[300.0, 15.0, 25.0, 8.0, 25.0, 8.0, 35.0]'
            ],
            'n_steps': [5, 8, 3, 12],
            'steps': [
                "['step1', 'step2', 'step3', 'step4', 'step5']",
                "['step1', 'step2', 'step3', 'step4', 'step5', 'step6', 'step7', 'step8']",
                "['step1', 'step2', 'step3']",
                "['step1', 'step2', 'step3', 'step4', 'step5', 'step6', 'step7', 'step8', 'step9', 'step10', 'step11', 'step12']"
            ],
            'description': ['Desc1', 'Desc2', None, 'Desc4'],  # Une valeur manquante
            'ingredients': [
                "['salt', 'pepper', 'oil']",
                "['beef', 'onion', 'salt']",
                "['sugar', 'flour', 'butter']",
                "['salt', 'sugar', 'milk']"
            ],
            'n_ingredients': [3, 3, 3, 3]
        }
        
        df = pd.DataFrame(data)
        
        # Créer un fichier temporaire
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        yield temp_file.name
        
        # Nettoyer après le test
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def analyzer(self, sample_csv_data):
        """Crée une instance de RecipeDataAnalyzer avec des données de test."""
        return RecipeDataAnalyzer(sample_csv_data)
    
    def test_init(self, sample_csv_data):
        """Test l'initialisation de l'analyseur."""
        analyzer = RecipeDataAnalyzer(sample_csv_data)
        assert analyzer.csv_path == sample_csv_data
        assert analyzer.recipe_data is None
        assert analyzer.cleaned_data is None
    
    def test_load_data(self, analyzer):
        """Test le chargement des données."""
        data = analyzer.load_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 4
        assert analyzer.recipe_data is not None
        assert 'name' in data.columns
        assert 'minutes' in data.columns
    
    def test_get_basic_info(self, analyzer):
        """Test l'obtention des informations de base."""
        analyzer.load_data()
        info = analyzer.get_basic_info()
        
        assert isinstance(info, dict)
        assert info['n_recipes'] == 4
        assert info['n_variables'] == 12
        assert info['total_missing_values'] == 1  # Une description manquante
        assert 'missing_by_column' in info
        assert 'columns' in info
    
    def test_get_basic_info_without_data(self, analyzer):
        """Test que get_basic_info lève une erreur sans données chargées."""
        with pytest.raises(ValueError, match="Les données doivent être chargées d'abord"):
            analyzer.get_basic_info()
    
    def test_analyze_minutes(self, analyzer):
        """Test l'analyse des minutes."""
        analyzer.load_data()
        stats = analyzer.analyze_minutes()
        
        assert isinstance(stats, dict)
        assert 'mean' in stats
        assert 'median' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert stats['min'] == 0
        assert stats['max'] == 432000
    
    def test_remove_outliers_minutes(self, analyzer):
        """Test la suppression des outliers de minutes."""
        analyzer.load_data()
        
        # Avant nettoyage
        assert len(analyzer.recipe_data) == 4
        
        # Nettoyage avec seuil personnalisé
        cleaned = analyzer.remove_outliers_minutes(max_minutes=1000)
        
        # Après nettoyage (devrait supprimer la recette de 432000 minutes)
        assert len(cleaned) < 4
        assert analyzer.cleaned_data is not None
        assert cleaned['minutes'].max() <= 1000
    
    def test_analyze_contributors(self, analyzer):
        """Test l'analyse des contributeurs."""
        analyzer.load_data()
        stats = analyzer.analyze_contributors()
        
        assert isinstance(stats, dict)
        assert stats['n_unique_contributors'] == 3  # 101, 102, 103
        assert stats['top_contributor_id'] == 101  # Contributeur avec 2 recettes
        assert stats['top_contributor_recipes'] == 2
    
    def test_parse_list_column(self, analyzer):
        """Test le parsing des colonnes liste."""
        analyzer.load_data()
        
        # Test avec les tags
        tags = analyzer.parse_list_column('tags')
        assert isinstance(tags, list)
        assert 'vegetarian' in tags
        assert 'quick' in tags
        assert 'meat' in tags
        
        # Test avec une colonne inexistante
        with pytest.raises(ValueError, match="Colonne .* non trouvée"):
            analyzer.parse_list_column('colonne_inexistante')
    
    def test_analyze_ingredients(self, analyzer):
        """Test l'analyse des ingrédients."""
        analyzer.load_data()
        stats = analyzer.analyze_ingredients()
        
        assert isinstance(stats, dict)
        assert 'n_unique_ingredients' in stats
        assert 'top_10_ingredients' in stats
        assert stats['n_unique_ingredients'] > 0
        
        # Vérifier que 'salt' est présent (apparaît dans plusieurs recettes)
        top_ingredients = [item[0] for item in stats['top_10_ingredients']]
        assert 'salt' in top_ingredients
    
    def test_analyze_tags(self, analyzer):
        """Test l'analyse des tags."""
        analyzer.load_data()
        stats = analyzer.analyze_tags()
        
        assert isinstance(stats, dict)
        assert 'n_unique_tags' in stats
        assert 'top_20_tags' in stats
        assert stats['n_unique_tags'] > 0
        
        # Vérifier que 'vegetarian' est présent
        top_tags = [item[0] for item in stats['top_20_tags']]
        assert 'vegetarian' in top_tags
    
    def test_process_nutrition_scores(self, analyzer):
        """Test le traitement des scores nutritionnels."""
        analyzer.load_data()
        nutrition_data = analyzer.process_nutrition_scores()
        
        assert isinstance(nutrition_data, pd.DataFrame)
        assert 'calories' in nutrition_data.columns
        assert 'total_fat_pct' in nutrition_data.columns
        
        # Vérifier que les valeurs sont converties en float
        assert nutrition_data['calories'].dtype in [np.float64, np.float32]
    
    def test_analyze_nutrition(self, analyzer):
        """Test l'analyse nutritionnelle."""
        analyzer.load_data()
        stats = analyzer.analyze_nutrition()
        
        assert isinstance(stats, dict)
        assert 'calories' in stats
        assert 'mean' in stats['calories']
        assert 'median' in stats['calories']
    
    def test_analyze_steps_and_ingredients_count(self, analyzer):
        """Test l'analyse du nombre d'étapes et d'ingrédients."""
        analyzer.load_data()
        stats = analyzer.analyze_steps_and_ingredients_count()
        
        assert isinstance(stats, dict)
        assert 'n_steps' in stats
        assert 'n_ingredients' in stats
        assert 'mean' in stats['n_steps']
        assert 'median' in stats['n_ingredients']
    
    def test_get_complete_analysis(self, analyzer):
        """Test l'analyse complète."""
        # Ne pas charger les données pour tester le chargement automatique
        analysis = analyzer.get_complete_analysis()
        
        assert isinstance(analysis, dict)
        assert 'basic_info' in analysis
        assert 'minutes_analysis' in analysis
        assert 'contributors_analysis' in analysis
        assert 'ingredients_analysis' in analysis
        assert 'tags_analysis' in analysis
        assert 'steps_ingredients_count' in analysis
        assert 'nutrition_analysis' in analysis
    
    def test_analyze_without_data_raises_error(self, analyzer):
        """Test que les méthodes lèvent des erreurs sans données."""
        with pytest.raises(ValueError):
            analyzer.analyze_minutes()
        
        with pytest.raises(ValueError):
            analyzer.analyze_contributors()
        
        with pytest.raises(ValueError):
            analyzer.parse_list_column('tags')


class TestEdgeCases:
    """Tests pour les cas limites."""
    
    def test_empty_csv_file(self):
        """Test avec un fichier CSV vide."""
        # Créer un fichier CSV vide
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write("name,id,minutes\n")  # Juste les en-têtes
        temp_file.close()
        
        try:
            analyzer = RecipeDataAnalyzer(temp_file.name)
            data = analyzer.load_data()
            assert len(data) == 0
            
            info = analyzer.get_basic_info()
            assert info['n_recipes'] == 0
        finally:
            os.unlink(temp_file.name)
    
    def test_malformed_list_data(self):
        """Test avec des données de liste malformées."""
        data = {
            'name': ['Recipe1'],
            'id': [1],
            'minutes': [30],
            'contributor_id': [101],
            'submitted': ['2020-01-01'],
            'tags': ['invalid_list_format'],  # Format invalide
            'nutrition': ['[100.0, 5.0]'],
            'n_steps': [5],
            'steps': ['["step1"]'],
            'description': ['Desc1'],
            'ingredients': ['invalid_format'],  # Format invalide
            'n_ingredients': [3]
        }
        
        df = pd.DataFrame(data)
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            analyzer = RecipeDataAnalyzer(temp_file.name)
            analyzer.load_data()
            
            # Devrait gérer gracieusement les formats invalides
            tags = analyzer.parse_list_column('tags')
            ingredients = analyzer.parse_list_column('ingredients')
            
            # Les listes malformées devraient être ignorées
            assert isinstance(tags, list)
            assert isinstance(ingredients, list)
        finally:
            os.unlink(temp_file.name)


if __name__ == "__main__":
    pytest.main([__file__])