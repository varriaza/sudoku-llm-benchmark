"""Tests for config loading and defaults."""
from __future__ import annotations
import textwrap

from sudoku_bench.config import BenchmarkConfig, load_config


def test_benchmark_config_save_llm_output_defaults_false():
    """save_llm_output defaults to False when not specified."""
    cfg = BenchmarkConfig()
    assert cfg.save_llm_output is False


def test_load_config_save_llm_output_not_set_defaults_false(tmp_path):
    """When save_llm_output is absent from the config file, it defaults to False."""
    yaml = textwrap.dedent("""\
        model:
          api_base: "http://localhost:8000/v1"
        puzzles: []
        benchmark:
          results_file: "results/benchmark.csv"
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml)

    config = load_config(config_file)
    assert config.benchmark.save_llm_output is False


def test_load_config_save_llm_output_true(tmp_path):
    """save_llm_output: true is parsed correctly."""
    yaml = textwrap.dedent("""\
        model:
          api_base: "http://localhost:8000/v1"
        puzzles: []
        benchmark:
          save_llm_output: true
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml)

    config = load_config(config_file)
    assert config.benchmark.save_llm_output is True


def test_load_config_save_llm_output_false_explicit(tmp_path):
    """save_llm_output: false is parsed correctly."""
    yaml = textwrap.dedent("""\
        model:
          api_base: "http://localhost:8000/v1"
        puzzles: []
        benchmark:
          save_llm_output: false
    """)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml)

    config = load_config(config_file)
    assert config.benchmark.save_llm_output is False
