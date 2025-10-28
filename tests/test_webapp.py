import builtins
import importlib
import logging
import sys
import types
from unittest.mock import Mock

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# ---------------------------------------------------------------------------
# Stub external dependencies before importing the application modules.
# ---------------------------------------------------------------------------

streamlit_stub = types.ModuleType("streamlit")
streamlit_stub.plotly_chart = Mock()
streamlit_stub.set_page_config = Mock()
streamlit_stub.cache_data = lambda func: func
streamlit_stub.error = Mock()
streamlit_stub.warning = Mock()
streamlit_stub.info = Mock()
streamlit_stub.success = Mock()
streamlit_stub.checkbox = Mock(return_value=False)
streamlit_stub.markdown = Mock()
sys.modules["streamlit"] = streamlit_stub

sklearn_module = types.ModuleType("sklearn")
linear_model_module = types.ModuleType("sklearn.linear_model")
ensemble_module = types.ModuleType("sklearn.ensemble")


class DummyLinearRegression:
    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        return np.zeros(len(X))


class DummyRandomForestRegressor(DummyLinearRegression):
    pass


linear_model_module.LinearRegression = DummyLinearRegression
ensemble_module.RandomForestRegressor = DummyRandomForestRegressor
sklearn_module.linear_model = linear_model_module
sklearn_module.ensemble = ensemble_module
sys.modules["sklearn"] = sklearn_module
sys.modules["sklearn.linear_model"] = linear_model_module
sys.modules["sklearn.ensemble"] = ensemble_module

# Optional dependencies occasionally required by Plotly/Pandas
defusedxml_module = types.ModuleType("defusedxml")
defusedxml_et_module = types.ModuleType("defusedxml.ElementTree")
defusedxml_module.ElementTree = defusedxml_et_module
sys.modules["defusedxml"] = defusedxml_module
sys.modules["defusedxml.ElementTree"] = defusedxml_et_module

xarray_module = types.ModuleType("xarray")
sys.modules["xarray"] = xarray_module

# Minimal dataset_analysis package
dataset_analysis_module = types.ModuleType("dataset_analysis")
dataset_preprocessing_module = types.ModuleType("dataset_analysis.dataset_preprocessing")
dataset_utils_module = types.ModuleType("dataset_analysis.utils")


def _stub_build_analysis_dataset(*, save=False):
    recipes = pd.DataFrame({"id": [1, 2, 3]})
    interactions = pd.DataFrame({"id": [1, 2, 3]})
    analysis = pd.DataFrame({"bayes_mean": [4.0, 4.2, 3.9], "effort_score": [20, 25, 30]})
    return recipes, interactions, analysis


def _stub_summarize_by_effort_quantiles(data, *, score_col, targets, quantiles, labels):
    summary = pd.DataFrame(
        {
            ("bayes_mean", "mean"): np.linspace(4.5, 3.5, quantiles),
        },
        index=pd.Index(labels, name="quartile"),
    )
    return {"summary": summary}


dataset_preprocessing_module.build_analysis_dataset = _stub_build_analysis_dataset
dataset_utils_module.summarize_by_effort_quantiles = _stub_summarize_by_effort_quantiles

dataset_analysis_module.dataset_preprocessing = dataset_preprocessing_module
dataset_analysis_module.utils = dataset_utils_module
dataset_analysis_module.__path__ = []

sys.modules["dataset_analysis"] = dataset_analysis_module
sys.modules["dataset_analysis.dataset_preprocessing"] = dataset_preprocessing_module
sys.modules["dataset_analysis.utils"] = dataset_utils_module

# Import after stubbing optional dependencies
webapp_utils = importlib.import_module("src.webapp_utils")
sys.modules["webapp_utils"] = webapp_utils
webapp = importlib.import_module("src.webapp")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_streamlit_mocks():
    """Ensure Streamlit mocks are fresh before every test."""
    streamlit_stub.plotly_chart.reset_mock()
    streamlit_stub.set_page_config.reset_mock()
    streamlit_stub.error.reset_mock()
    streamlit_stub.warning.reset_mock()
    streamlit_stub.info.reset_mock()
    streamlit_stub.success.reset_mock()
    streamlit_stub.checkbox.reset_mock()
    streamlit_stub.markdown.reset_mock()
    yield


@pytest.fixture
def mock_logger(monkeypatch):
    logger = Mock()
    monkeypatch.setattr(webapp, "logger", logger)
    return logger


# ---------------------------------------------------------------------------
# Rich Streamlit stub for integration-style tests
# ---------------------------------------------------------------------------


class DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyColumn(DummyContext):
    def metric(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def selectbox(self, label, options, index=0, **kwargs):
        if not options:
            return None
        index = min(max(index, 0), len(options) - 1)
        return options[index]

    def slider(self, label, min_value=None, max_value=None, value=None, **kwargs):
        if value is not None:
            return value
        if isinstance(min_value, (int, float)) and isinstance(max_value, (int, float)):
            return (min_value + max_value) / 2
        return value

    def multiselect(self, label, options, default=None, **kwargs):
        if default is not None:
            return list(default)
        return list(options) if options is not None else []


class DummySidebar:
    def image(self, *args, **kwargs):
        return None

    def header(self, *args, **kwargs):
        return None

    def slider(self, label, min_value=None, max_value=None, value=None, **kwargs):
        if value is not None:
            return value
        if isinstance(min_value, (int, float)) and isinstance(max_value, (int, float)):
            return (min_value + max_value) / 2
        return value

    def multiselect(self, label, options, default=None, **kwargs):
        if default is not None:
            return list(default)
        return list(options) if options is not None else []


class DummyExpander(DummyContext):
    pass


class DummyTab(DummyContext):
    pass


class DummyStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.sidebar = DummySidebar()
        self._page_config_calls: list[dict] = []
        self.cache_data = lambda func: func

    # Layout & display helpers -------------------------------------------------
    def set_page_config(self, **kwargs):
        self._page_config_calls.append(kwargs)

    def plotly_chart(self, *args, **kwargs):
        return None

    def columns(self, spec, **kwargs):
        count = spec if isinstance(spec, int) else len(tuple(spec))
        return [DummyColumn() for _ in range(count)]

    def expander(self, *args, **kwargs):
        return DummyExpander()

    def tabs(self, names):
        return tuple(DummyTab() for _ in names)

    # Widgets ------------------------------------------------------------------
    def checkbox(self, *args, **kwargs):
        return kwargs.get("value", True)

    def selectbox(self, label, options, index=0, **kwargs):
        if not options:
            return None
        index = min(max(index, 0), len(options) - 1)
        return options[index]

    def radio(self, label, options, index=0, **kwargs):
        if not options:
            return None
        index = min(max(index, 0), len(options) - 1)
        return options[index]

    def slider(self, label, min_value=None, max_value=None, value=None, **kwargs):
        if value is not None:
            return value
        if isinstance(min_value, (int, float)) and isinstance(max_value, (int, float)):
            return (min_value + max_value) / 2
        return min_value

    def multiselect(self, label, options, default=None, **kwargs):
        if default is not None:
            return list(default)
        return list(options) if options is not None else []

    # Text & data outputs ------------------------------------------------------
    def title(self, *args, **kwargs):
        return None

    def header(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None




# ---------------------------------------------------------------------------
# Tests for src.webapp
# ---------------------------------------------------------------------------

def test_import_analysis_modules_success(mock_logger):
    use_real, build_func = webapp._import_analysis_modules()
    assert use_real is True
    assert callable(build_func)
    mock_logger.info.assert_called_once_with("Modules d'analyse chargés avec succès.")


def test_import_analysis_modules_missing(monkeypatch):
    mock_logger = Mock()
    monkeypatch.setattr(webapp, "logger", mock_logger)
    original_module = sys.modules.get("dataset_analysis.dataset_preprocessing")
    original_attr = getattr(dataset_analysis_module, "dataset_preprocessing", None)
    dummy_module = types.ModuleType("dataset_analysis.dataset_preprocessing")
    sys.modules["dataset_analysis.dataset_preprocessing"] = dummy_module
    dataset_analysis_module.dataset_preprocessing = dummy_module
    try:
        use_real, build_func = webapp._import_analysis_modules()
    finally:
        if original_module is not None:
            sys.modules["dataset_analysis.dataset_preprocessing"] = original_module
        else:
            del sys.modules["dataset_analysis.dataset_preprocessing"]
        if original_attr is not None:
            dataset_analysis_module.dataset_preprocessing = original_attr
        else:
            delattr(dataset_analysis_module, "dataset_preprocessing")

    assert use_real is False
    assert build_func is None
    mock_logger.warning.assert_called_once()


def test_plotly_config_for_stretch():
    config = webapp._plotly_config("stretch")
    assert config["responsive"] is True
    assert "modeBarButtonsToRemove" in config


def test_plotly_display_calls_streamlit_plotly_chart():
    fig = go.Figure()
    webapp._plotly_display(fig, width="stretch")
    streamlit_stub.plotly_chart.assert_called_once()


def test_plotly_display_invalid_width():
    fig = go.Figure()
    with pytest.raises(ValueError):
        webapp._plotly_display(fig, width="invalid")  # type: ignore[arg-type]


def test_generate_sample_data_structure(mock_logger):
    df = webapp.generate_sample_data(8)
    assert len(df) == 8
    expected_cols = {
        "id",
        "effort_score",
        "bayes_mean",
        "effort_category",
        "n_steps",
        "n_ingredients",
    }
    assert expected_cols.issubset(df.columns)
    mock_logger.info.assert_called_once()


def test_generate_sample_data_reproducible(mock_logger):
    df_a = webapp.generate_sample_data(50)
    df_b = webapp.generate_sample_data(50)
    pd.testing.assert_frame_equal(df_a, df_b)
    mock_logger.info.assert_called()


def test_load_real_datasets_success(mock_logger):
    recipes = pd.DataFrame({"id": [1, 2]})
    interactions = pd.DataFrame({"id": [1, 2]})
    analysis = pd.DataFrame({"id": [1, 2]})

    def build_dataset(*, save=False):
        assert save is False
        return recipes, interactions, analysis

    result = webapp.load_real_datasets(build_dataset)
    assert result == (recipes, interactions, analysis, True)
    assert mock_logger.info.call_count >= 1


def test_load_real_datasets_file_not_found(monkeypatch):
    mock_logger = Mock()
    monkeypatch.setattr(webapp, "logger", mock_logger)
    streamlit_error = Mock()
    monkeypatch.setattr(streamlit_stub, "error", streamlit_error)

    def build_dataset(*, save=False):
        raise FileNotFoundError("missing")

    result = webapp.load_real_datasets(build_dataset)
    assert result == (None, None, None, False)
    mock_logger.error.assert_called_once()
    streamlit_error.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for src.webapp_utils
# ---------------------------------------------------------------------------

def test_compute_histogram_figure_returns_plotly_figure():
    series = pd.Series([1, 2, 3, 4, 5], name="minutes")
    fig = webapp_utils.compute_histogram_figure(series, var_label="Minutes")
    assert isinstance(fig, go.Figure)


def test_compute_descriptive_stats_basic_values():
    series = pd.Series([1, 2, 3, 4, 5], name="scores")
    stats = webapp_utils.compute_descriptive_stats(series)
    assert stats.mean == pytest.approx(3.0)
    assert stats.median == pytest.approx(3.0)
    assert stats.count == 5


def test_fit_simple_regression_returns_result():
    data = pd.DataFrame({"effort_score": [1, 2, 3, 4], "bayes_mean": [2, 4, 6, 8]})
    result = webapp_utils.fit_simple_regression(data, "effort_score", "bayes_mean")
    assert result is not None
    assert result.r_squared == pytest.approx(1.0)


def test_compute_effort_pattern_orders_categories():
    data = pd.DataFrame(
        {
            "effort_category": ["Modéré", "Facile", "Très Facile", "Modéré"],
            "bayes_mean": [4.0, 4.5, 4.8, 4.1],
        }
    )
    pattern = webapp_utils.compute_effort_pattern(data)
    assert pattern is not None
    assert pattern.categories[0] == "Très Facile"
    assert len(pattern.observed) == len(pattern.categories)


def test_compute_quartile_pattern_from_existing_column():
    data = pd.DataFrame(
        {
            "effort_quartile": ["Q1", "Q2", "Q3", "Q4"],
            "bayes_mean": [4.5, 4.2, 4.0, 3.8],
        }
    )
    pattern = webapp_utils.compute_quartile_pattern(data)
    assert pattern is not None
    assert list(pattern.labels) == ["Q1", "Q2", "Q3", "Q4"]


def test_compute_quartile_pattern_without_precomputed_quartile():
    data = pd.DataFrame(
        {
            "effort_score": [10, 20, 30, 40],
            "bayes_mean": [4.2, 4.1, 4.0, 3.9],
        }
    )
    pattern = webapp_utils.compute_quartile_pattern(
        data, effort_col_candidates=("effort_quartile", "effort_score")
    )
    assert pattern is not None
    assert len(pattern.labels) == 4


def test_compute_quartile_pattern_handles_errors(monkeypatch):
    data = pd.DataFrame(
        {
            "effort_score": [10, 20, 30, 40],
            "bayes_mean": [4.2, 4.1, 4.0, 3.9],
        }
    )

    def failing_quantiles(*args, **kwargs):
        raise ValueError("failure")

    monkeypatch.setattr(webapp_utils, "summarize_by_effort_quantiles", failing_quantiles)
    pattern = webapp_utils.compute_quartile_pattern(
        data, effort_col_candidates=("effort_quartile", "effort_score")
    )
    assert pattern is None


def test_compute_quartile_pattern_requires_bayes_mean():
    data = pd.DataFrame({"effort_score": [10, 20, 30]})
    assert webapp_utils.compute_quartile_pattern(data) is None


def test_describe_numeric_columns_filters_non_numeric():
    data = pd.DataFrame({"numeric": [1, 2, 3], "text": ["a", "b", "c"]})
    stats = webapp_utils.describe_numeric_columns(data, ["numeric", "text"])
    assert list(stats.keys()) == ["numeric"]


def test_infer_filter_options_returns_ranges():
    data = pd.DataFrame(
        {
            "age_months": [1.5, 10.2, 20.7],
            "n_interactions": [3, 5, 7],
            "effort_category": ["Facile", "Modéré", "Facile"],
        }
    )
    options = webapp_utils.infer_filter_options(data)
    assert options.age_range == (1.0, 21.0)
    assert options.interactions_range == (3, 7)
    assert options.effort_categories == ["Facile", "Modéré"]


# ---------------------------------------------------------------------------
# Additional coverage on Streamlit orchestration
# ---------------------------------------------------------------------------


def test_configure_streamlit_calls_set_page_config():
    webapp._configure_streamlit()
    streamlit_stub.set_page_config.assert_called_once()


def test_prepare_analysis_data_real_branch(monkeypatch):
    analysis_df = pd.DataFrame(
        {
            "bayes_mean": [4.2, 4.0],
            "effort_score": [20.0, 25.0],
        }
    )

    def fake_import():
        def dummy_builder(*, save=False):
            return analysis_df, analysis_df, analysis_df

        return True, dummy_builder

    monkeypatch.setattr(webapp, "_import_analysis_modules", fake_import)
    monkeypatch.setattr(
        webapp,
        "load_real_datasets",
        lambda build_func: (analysis_df, analysis_df, analysis_df, True),
    )

    result_df, origin = webapp._prepare_analysis_data()
    pd.testing.assert_frame_equal(result_df, analysis_df)
    assert origin == "réelles"


def test_prepare_analysis_data_fallback_branch(monkeypatch):
    fallback_df = pd.DataFrame({"bayes_mean": [3.9], "effort_score": [18.0]})
    monkeypatch.setattr(webapp, "_import_analysis_modules", lambda: (False, None))
    monkeypatch.setattr(webapp, "generate_sample_data", lambda: fallback_df)

    result_df, origin = webapp._prepare_analysis_data()
    pd.testing.assert_frame_equal(result_df, fallback_df)
    assert origin == "simulées"


def test_render_sidebar_with_dummy_streamlit(monkeypatch):
    dataset = webapp.generate_sample_data(50)
    dummy_st = DummyStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", dummy_st)
    monkeypatch.setattr(webapp, "st", dummy_st)
    monkeypatch.setattr(webapp.Image, "open", lambda *_: object())

    filtered = webapp._render_sidebar("réelles", dataset)
    assert not filtered.empty
    assert set(filtered.columns) == set(dataset.columns)


def test_render_storytelling_full_flow(monkeypatch):
    data = webapp.generate_sample_data(120)
    dummy_st = DummyStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", dummy_st)
    monkeypatch.setattr(webapp, "st", dummy_st)
    monkeypatch.setattr(webapp.Image, "open", lambda *_: object())
    monkeypatch.setattr(webapp.px, "scatter", lambda *a, **k: go.Figure())
    monkeypatch.setattr(webapp.px, "histogram", lambda *a, **k: go.Figure())
    monkeypatch.setattr(webapp.px, "bar", lambda *a, **k: go.Figure())
    monkeypatch.setattr(webapp.px, "imshow", lambda *a, **k: go.Figure())

    def fake_exists(path):
        return str(path).endswith("rapport_analyse_effort_popularite.md")

    monkeypatch.setattr(webapp.Path, "exists", fake_exists)
    monkeypatch.setattr(
        webapp.Path,
        "read_text",
        lambda self, encoding="utf-8": "# Rapport\n![Alt](./images/sample.png)",
    )

    webapp.render_storytelling(data, "réelles", len(data))


def test_display_about_tab_missing_file(monkeypatch):
    def raise_not_found(self, encoding="utf-8"):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(webapp.Path, "exists", lambda self: False)
    monkeypatch.setattr(webapp.Path, "read_text", raise_not_found)

    webapp.display_about_tab()
    streamlit_stub.error.assert_called_once()


def test_main_invokes_render_storytelling(monkeypatch):
    dataset = webapp.generate_sample_data(20)
    monkeypatch.setattr(webapp, "_prepare_analysis_data", lambda: (dataset, "réelles"))
    monkeypatch.setattr(webapp, "_render_sidebar", lambda origin, df: df)
    called: list[tuple] = []
    monkeypatch.setattr(
        webapp,
        "render_storytelling",
        lambda filtered, origin, total_recipes: called.append(
            (len(filtered), origin, total_recipes)
        ),
    )

    streamlit_stub.set_page_config.reset_mock()
    webapp.main()

    assert called and called[0][1] == "réelles"
    streamlit_stub.set_page_config.assert_called_once()


def test_setup_logging_fallback(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src.logger":
            raise ImportError("forced")
        return original_import(name, globals, locals, fromlist, level)

    original_logger = webapp.logger
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(webapp, "logger", None)

    fallback = webapp._setup_logging()
    assert isinstance(fallback, logging.Logger)

    webapp.logger = original_logger


def test_compute_story_indicators_missing_columns():
    data = pd.DataFrame({"other": [1, 2, 3]})
    indicators = webapp._compute_story_indicators(data, quartile_pattern=None)
    assert np.isnan(indicators["avg_popularity"])
    assert indicators["quartile_means"] is None


class _MetricsColumn:
    def __init__(self):
        self.values = []

    def metric(self, label, value):
        self.values.append((label, value))


class _MetricsStreamlit:
    def __init__(self):
        self.captions: list[str] = []
        self.columns_instances = [_MetricsColumn() for _ in range(3)]

    def columns(self, count):
        return self.columns_instances

    def caption(self, text):
        self.captions.append(text)


def test_render_metrics_handles_missing_values(monkeypatch):
    stub = _MetricsStreamlit()
    monkeypatch.setattr(webapp, "st", stub)

    indicators = {
        "recipes": 5,
        "avg_popularity": np.nan,
        "avg_effort": np.nan,
        "share_very_easy": np.nan,
        "share_very_hard": np.nan,
    }

    webapp._render_metrics(indicators, total_count=10)
    assert stub.captions and "Filtres actifs" in stub.captions[0]


def test_render_metrics_shows_category_caption(monkeypatch):
    stub = _MetricsStreamlit()
    monkeypatch.setattr(webapp, "st", stub)

    indicators = {
        "recipes": 5,
        "avg_popularity": 4.2,
        "avg_effort": 30.0,
        "share_very_easy": 0.4,
        "share_very_hard": 0.2,
    }

    webapp._render_metrics(indicators, total_count=5)
    assert any("Répartition" in caption for caption in stub.captions)


def test_render_patterns_handles_missing(monkeypatch):
    streamlit_stub.info.reset_mock()
    webapp._render_patterns(None, None)
    streamlit_stub.info.assert_called_once()


def test_render_correlation_matrix_requires_variables():
    streamlit_stub.info.reset_mock()
    webapp._render_correlation_matrix(pd.DataFrame({"other": [1, 2]}))
    streamlit_stub.info.assert_called_once()


def test_render_correlation_matrix_drops_empty_columns():
    streamlit_stub.info.reset_mock()
    df = pd.DataFrame(
        {
            "log_minutes": [np.nan, np.nan],
            "effort_score": [1.0, 2.0],
        }
    )
    webapp._render_correlation_matrix(df)
    streamlit_stub.info.assert_called_once()


def test_render_explorer_empty_dataset():
    streamlit_stub.info.reset_mock()
    webapp._render_explorer(pd.DataFrame())
    streamlit_stub.info.assert_called_once()


def test_render_explorer_missing_columns(monkeypatch):
    streamlit_stub.info.reset_mock()
    monkeypatch.setattr(streamlit_stub, "write", Mock(), raising=False)
    df = pd.DataFrame({"effort_score": [10, 20]})
    webapp._render_explorer(df)
    streamlit_stub.info.assert_called_once()


def test_render_scenario_planner_missing_required_columns(monkeypatch):
    dummy_st = DummyStreamlit()
    dummy_st.info = Mock()
    dummy_st.write = Mock()
    monkeypatch.setitem(sys.modules, "streamlit", dummy_st)
    monkeypatch.setattr(webapp, "st", dummy_st)

    df = pd.DataFrame({"minutes": [10], "bayes_mean": [4.0]})
    webapp._render_scenario_planner(df)
    dummy_st.info.assert_called_once()


def test_render_scenario_planner_missing_series(monkeypatch):
    dummy_st = DummyStreamlit()
    dummy_st.info = Mock()
    dummy_st.write = Mock()
    monkeypatch.setitem(sys.modules, "streamlit", dummy_st)
    monkeypatch.setattr(webapp, "st", dummy_st)

    df = pd.DataFrame(
        {
            "minutes": [np.nan, np.nan],
            "n_ingredients": [np.nan, np.nan],
            "bayes_mean": [4.0, 4.2],
        }
    )
    webapp._render_scenario_planner(df)
    dummy_st.info.assert_called_once()


class _ScenarioColumn:
    def __init__(self, value):
        self.value = value

    def slider(self, *args, **kwargs):
        return self.value


class _ScenarioStreamlit(DummyStreamlit):
    def __init__(self):
        super().__init__()
        self.warning = Mock()
        self.write = Mock()
        self.markdown = Mock()
        self.dataframe = Mock()

    def columns(self, spec, **kwargs):
        return [_ScenarioColumn(0), _ScenarioColumn(0), _ScenarioColumn(5)]

    def radio(self, *args, **kwargs):
        return "Peu importe"


def test_render_scenario_planner_no_matching_recipes(monkeypatch):
    dummy_st = _ScenarioStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", dummy_st)
    monkeypatch.setattr(webapp, "st", dummy_st)

    df = pd.DataFrame(
        {
            "minutes": [30, 40],
            "n_ingredients": [5, 7],
            "bayes_mean": [4.5, 4.6],
        }
    )

    webapp._render_scenario_planner(df)
    dummy_st.warning.assert_called_once()


def test_render_storytelling_empty_dataset(monkeypatch):
    streamlit_stub.warning.reset_mock()
    empty_df = pd.DataFrame(columns=["bayes_mean"])
    webapp.render_storytelling(empty_df, "réelles", total_recipes=0)
    streamlit_stub.warning.assert_called_once()


def test_display_about_tab_generic_exception(monkeypatch):
    monkeypatch.setattr(webapp.Path, "exists", lambda self: True)
    monkeypatch.setattr(webapp.Path, "read_text", lambda self, encoding="utf-8": (_ for _ in ()).throw(ValueError("boom")))

    streamlit_stub.error.reset_mock()
    webapp.display_about_tab()
    streamlit_stub.error.assert_called_once()


def test_main_handles_empty_dataset(monkeypatch):
    monkeypatch.setattr(webapp, "_prepare_analysis_data", lambda: (pd.DataFrame(), "réelles"))
    monkeypatch.setattr(webapp, "_render_sidebar", Mock())
    monkeypatch.setattr(webapp, "render_storytelling", Mock())

    streamlit_stub.set_page_config.reset_mock()
    streamlit_stub.error.reset_mock()

    webapp.main()

    streamlit_stub.error.assert_called_once()
