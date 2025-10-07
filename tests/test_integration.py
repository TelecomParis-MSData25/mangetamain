"""
Tests d'intégration pour le module data_analysis.py
Ces tests utilisent de vraies données pour vérifier le comportement avec le dataset complet.
"""

import pytest
import os
from src.data_analysis import RecipeDataAnalyzer


@pytest.mark.integration
class TestIntegrationRecipeDataAnalyzer:
    """Tests d'intégration avec le vrai dataset."""
    
    @pytest.fixture
    def real_analyzer(self):
        """Crée un analyseur avec le vrai dataset s'il existe."""
        csv_path = 'data/RAW_recipes.csv'
        if not os.path.exists(csv_path):
            pytest.skip("Dataset réel non disponible")
        return RecipeDataAnalyzer(csv_path)
    
    def test_load_real_data(self, real_analyzer):
        """Test le chargement du vrai dataset."""
        data = real_analyzer.load_data()
        
        assert len(data) > 200000  # Dataset connu pour avoir ~231k recettes
        assert 'name' in data.columns
        assert 'minutes' in data.columns
        assert 'ingredients' in data.columns
    
    def test_real_basic_info(self, real_analyzer):
        """Test des informations de base sur le vrai dataset."""
        real_analyzer.load_data()
        info = real_analyzer.get_basic_info()
        
        assert info['n_recipes'] > 200000
        assert info['n_variables'] == 12
        assert info['total_missing_values'] > 0
    
    def test_real_minutes_analysis(self, real_analyzer):
        """Test l'analyse des minutes sur le vrai dataset."""
        real_analyzer.load_data()
        stats = real_analyzer.analyze_minutes()
        
        # Vérifications basées sur l'analyse connue
        assert stats['min'] == 0
        assert stats['max'] > 1000000  # Très grande valeur d'outlier
        assert stats['median'] < 100  # Médiane raisonnable
    
    def test_real_outlier_removal(self, real_analyzer):
        """Test la suppression d'outliers sur le vrai dataset."""
        real_analyzer.load_data()
        initial_count = len(real_analyzer.recipe_data)
        
        cleaned_data = real_analyzer.remove_outliers_minutes()
        
        assert len(cleaned_data) < initial_count
        assert cleaned_data['minutes'].max() <= 30 * 24 * 60  # Max 1 mois
    
    def test_real_ingredients_analysis(self, real_analyzer):
        """Test l'analyse des ingrédients sur le vrai dataset."""
        real_analyzer.load_data()
        stats = real_analyzer.analyze_ingredients()
        
        # Vérifications basées sur l'analyse connue
        assert stats['n_unique_ingredients'] > 10000
        
        # Vérifier que 'salt' est dans le top 10 (connu d'après l'analyse)
        top_ingredients = [item[0] for item in stats['top_10_ingredients']]
        assert 'salt' in top_ingredients
    
    def test_real_tags_analysis(self, real_analyzer):
        """Test l'analyse des tags sur le vrai dataset."""
        real_analyzer.load_data()
        stats = real_analyzer.analyze_tags()
        
        # Vérifications basées sur l'analyse connue
        assert stats['n_unique_tags'] > 500
        
        # Tags connus pour être populaires
        top_tags = [item[0] for item in stats['top_20_tags']]
        assert 'preparation' in top_tags or 'time-to-make' in top_tags
    
    @pytest.mark.slow
    def test_real_complete_analysis(self, real_analyzer):
        """Test l'analyse complète sur le vrai dataset (peut être lent)."""
        analysis = real_analyzer.get_complete_analysis()
        
        # Vérifier que tous les composants sont présents
        expected_keys = [
            'basic_info', 'minutes_analysis', 'contributors_analysis',
            'ingredients_analysis', 'tags_analysis', 'steps_ingredients_count',
            'nutrition_analysis'
        ]
        
        for key in expected_keys:
            assert key in analysis
            assert analysis[key] is not None
    
    def test_real_nutrition_processing(self, real_analyzer):
        """Test le traitement des données nutritionnelles sur le vrai dataset."""
        real_analyzer.load_data()
        nutrition_data = real_analyzer.process_nutrition_scores()
        
        # Vérifier que les colonnes nutritionnelles sont créées
        nutrition_columns = [
            'calories', 'total_fat_pct', 'sugar_pct', 
            'sodium_pct', 'protein_pct', 'saturated_fat_pct', 'carbohydrates_pct'
        ]
        
        for col in nutrition_columns:
            assert col in nutrition_data.columns
    
    def test_real_contributors_analysis(self, real_analyzer):
        """Test l'analyse des contributeurs sur le vrai dataset."""
        real_analyzer.load_data()
        stats = real_analyzer.analyze_contributors()
        
        # Vérifications basées sur l'analyse connue
        assert stats['n_unique_contributors'] > 25000
        assert stats['top_contributor_recipes'] > 1000  # Le top contributeur a beaucoup de recettes
        assert 0 < stats['top_contributor_percentage'] < 5  # Pas plus de 5% du total


@pytest.mark.unit
class TestRecipeDataAnalyzerEdgeCases:
    """Tests supplémentaires pour les cas limites."""
    
    def test_analyze_minutes_with_cleaned_data(self):
        """Test que analyze_minutes utilise les données nettoyées si disponibles."""
        # Ce test nécessiterait des données mockées plus complexes
        pass
    
    def test_parse_list_column_with_mixed_formats(self):
        """Test le parsing avec des formats mixtes."""
        # Ce test pourrait être ajouté avec des données spécifiques
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])