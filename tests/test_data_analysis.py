import pandas as pd
import pytest

from src.data_analysis import RecipeDataAnalyzer


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "minutes": [10, 20, 3000],
            "contributor_id": [1, 1, 2],
            "ingredients": [
                "['salt', 'pepper']",
                "['sugar']",
                "['salt', 'oil']",
            ],
            "tags": [
                "['quick']",
                "['dessert']",
                "['quick', 'family']",
            ],
            "nutrition": [
                "[100, 10, 5, 200, 3, 4, 20]",
                "[200, 15, 7, 300, 5, 6, 30]",
                "[150, 12, 6, 250, 4, 5, 25]",
            ],
            "n_steps": [5, 3, 8],
            "n_ingredients": [7, 4, 6],
        }
    )


@pytest.fixture
def analyzer(tmp_path, sample_dataframe):
    csv_path = tmp_path / "recipes.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    instance = RecipeDataAnalyzer(str(csv_path))
    return instance


def test_load_and_basic_info(analyzer, sample_dataframe):
    loaded = analyzer.load_data()
    pd.testing.assert_frame_equal(loaded, sample_dataframe)

    info = analyzer.get_basic_info()
    assert info["n_recipes"] == len(sample_dataframe)
    assert set(info["columns"]) == set(sample_dataframe.columns)


def test_analyze_minutes(analyzer, sample_dataframe):
    analyzer.recipe_data = sample_dataframe
    stats = analyzer.analyze_minutes()
    assert stats["mean"] == pytest.approx(sample_dataframe["minutes"].mean())
    assert stats["max"] == 3000


def test_remove_outliers_minutes(analyzer, sample_dataframe):
    analyzer.recipe_data = sample_dataframe
    cleaned = analyzer.remove_outliers_minutes(max_minutes=1000)
    assert cleaned["minutes"].max() <= 1000
    assert analyzer.cleaned_data is not None


def test_analyze_contributors(analyzer, sample_dataframe):
    analyzer.recipe_data = sample_dataframe
    analyzer.cleaned_data = sample_dataframe
    stats = analyzer.analyze_contributors()
    assert stats["n_unique_contributors"] == 2
    assert stats["top_contributor_id"] == 1


def test_parse_list_column(analyzer, sample_dataframe):
    analyzer.recipe_data = sample_dataframe
    analyzer.cleaned_data = sample_dataframe
    items = analyzer.parse_list_column("ingredients")
    assert {"salt", "pepper", "sugar", "oil"} <= set(items)


def test_parse_list_column_missing(analyzer, sample_dataframe):
    analyzer.recipe_data = sample_dataframe
    with pytest.raises(ValueError):
        analyzer.parse_list_column("unknown")


def test_analyze_ingredients(analyzer, sample_dataframe):
    analyzer.recipe_data = sample_dataframe
    analyzer.cleaned_data = sample_dataframe
    stats = analyzer.analyze_ingredients()
    assert stats["n_unique_ingredients"] >= 3
    assert stats["top_10_ingredients"]


def test_analyze_tags(analyzer, sample_dataframe):
    analyzer.recipe_data = sample_dataframe
    analyzer.cleaned_data = sample_dataframe
    stats = analyzer.analyze_tags()
    assert stats["n_unique_tags"] >= 3
    assert stats["top_20_tags"]


def test_process_and_analyze_nutrition(analyzer, sample_dataframe):
    analyzer.recipe_data = sample_dataframe
    processed = analyzer.process_nutrition_scores()
    for column in [
        "calories",
        "total_fat_pct",
        "sugar_pct",
        "sodium_pct",
        "protein_pct",
        "saturated_fat_pct",
        "carbohydrates_pct",
    ]:
        assert column in processed.columns

    stats = analyzer.analyze_nutrition()
    assert "calories" in stats
    assert stats["calories"]["mean"] == pytest.approx(processed["calories"].mean())


def test_analyze_steps_and_ingredients_count(analyzer, sample_dataframe):
    analyzer.recipe_data = sample_dataframe
    stats = analyzer.analyze_steps_and_ingredients_count()
    assert stats["n_steps"]["max"] == 8
    assert stats["n_ingredients"]["median"] == pytest.approx(sample_dataframe["n_ingredients"].median())


def test_get_complete_analysis(analyzer, sample_dataframe):
    analyzer.recipe_data = sample_dataframe
    analyzer.cleaned_data = sample_dataframe
    analysis = analyzer.get_complete_analysis()
    expected_keys = {
        "basic_info",
        "minutes_analysis",
        "contributors_analysis",
        "ingredients_analysis",
        "tags_analysis",
        "steps_ingredients_count",
        "nutrition_analysis",
    }
    assert expected_keys == set(analysis.keys())
