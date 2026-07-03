"""Tests for koboi/eval/config.py -- Eval configuration."""

from __future__ import annotations


from koboi.eval.config import EvalConfig, SuiteConfig, RegressionConfig


class TestSuiteConfig:
    def test_defaults(self):
        s = SuiteConfig(name="test")
        assert s.name == "test"
        assert s.framework == "yaml"
        assert s.source == ""
        assert s.scorers == []
        assert s.max_cases is None
        assert s.tags == []
        assert s.metadata == {}

    def test_custom(self):
        s = SuiteConfig(
            name="my_suite",
            framework="bfcl",
            source="data.json",
            scorers=["tool_usage"],
            max_cases=100,
            tags=["fast"],
        )
        assert s.framework == "bfcl"
        assert s.max_cases == 100


class TestRegressionConfig:
    def test_defaults(self):
        r = RegressionConfig()
        assert r.baseline_dir == "eval_baselines"
        assert r.alert_on_regression is True
        assert r.regression_threshold == 0.05


class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig({})
        assert cfg.threshold == 0.6
        assert cfg.parallel is False
        assert cfg.max_concurrency == 5
        assert cfg.output_dir == "eval_results"
        assert cfg.default_scorers == []
        assert cfg.suites == []

    def test_custom_values(self):
        data = {
            "threshold": 0.8,
            "parallel": True,
            "max_concurrency": 10,
            "output_dir": "/tmp/eval",
            "scorers": ["tool_usage", "keyword_presence"],
        }
        cfg = EvalConfig(data)
        assert cfg.threshold == 0.8
        assert cfg.parallel is True
        assert cfg.max_concurrency == 10
        assert len(cfg.default_scorers) == 2

    def test_parse_suites(self):
        data = {
            "suites": [
                {"name": "s1", "framework": "yaml", "source": "cases.yaml"},
                {"name": "s2", "framework": "bfcl", "max_cases": 50},
            ]
        }
        cfg = EvalConfig(data)
        assert len(cfg.suites) == 2
        assert cfg.suites[0].name == "s1"
        assert cfg.suites[1].max_cases == 50

    def test_parse_regression(self):
        data = {"regression": {"baseline_dir": "/tmp/baselines", "regression_threshold": 0.1}}
        cfg = EvalConfig(data)
        assert cfg.regression.baseline_dir == "/tmp/baselines"
        assert cfg.regression.regression_threshold == 0.1

    def test_get_suite_found(self):
        data = {"suites": [{"name": "s1"}]}
        cfg = EvalConfig(data)
        assert cfg.get_suite("s1") is not None

    def test_get_suite_not_found(self):
        cfg = EvalConfig({})
        assert cfg.get_suite("nonexistent") is None

    def test_build_scorers_empty(self):
        cfg = EvalConfig({})
        scorers = cfg.build_scorers()
        assert scorers == []

    def test_build_scorers_from_names(self):
        cfg = EvalConfig({"scorers": ["tool_usage", "keyword_presence"]})
        scorers = cfg.build_scorers()
        assert len(scorers) == 2

    def test_build_scorers_unknown_skipped(self):
        cfg = EvalConfig({"scorers": ["nonexistent_scorer"]})
        scorers = cfg.build_scorers()
        assert len(scorers) == 0
