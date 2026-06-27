"""Tests for eval loaders and scorers edge cases."""

from __future__ import annotations


import pytest

from koboi.eval.config import EvalConfig, SuiteConfig, RegressionConfig


class TestEvalConfig:
    def test_default_values(self):
        config = EvalConfig({})
        assert config.threshold == 0.6
        assert config.parallel is False
        assert config.max_concurrency == 5
        assert config.output_dir == "eval_results"

    def test_with_suites(self):
        data = {
            "threshold": 0.8,
            "suites": [
                {"name": "test-suite", "framework": "yaml", "source": "data.yaml"},
            ],
        }
        config = EvalConfig(data)
        assert len(config.suites) == 1
        assert config.suites[0].name == "test-suite"

    def test_with_regression(self):
        data = {
            "regression": {
                "baseline_dir": "baselines",
                "alert_on_regression": False,
                "regression_threshold": 0.1,
            },
        }
        config = EvalConfig(data)
        assert config.regression.baseline_dir == "baselines"
        assert config.regression.alert_on_regression is False

    def test_get_suite(self):
        data = {"suites": [{"name": "my-suite"}]}
        config = EvalConfig(data)
        suite = config.get_suite("my-suite")
        assert suite is not None
        assert suite.name == "my-suite"

    def test_get_suite_not_found(self):
        config = EvalConfig({})
        assert config.get_suite("nonexistent") is None

    def test_parse_suite_defaults(self):
        suite = EvalConfig._parse_suite({})
        assert suite.name == "unnamed"
        assert suite.framework == "yaml"


class TestSuiteConfig:
    def test_defaults(self):
        suite = SuiteConfig(name="test")
        assert suite.framework == "yaml"
        assert suite.source == ""
        assert suite.scorers == []
        assert suite.max_cases is None
        assert suite.tags == []
        assert suite.metadata == {}


class TestRegressionConfig:
    def test_defaults(self):
        config = RegressionConfig()
        assert config.baseline_dir == "eval_baselines"
        assert config.alert_on_regression is True
        assert config.regression_threshold == 0.05


class TestEvalConfigBuildSuite:
    async def test_build_suite_not_found(self):
        config = EvalConfig({})
        with pytest.raises(ValueError, match="not found"):
            await config.build_suite("nonexistent")
