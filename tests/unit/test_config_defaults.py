import pytest

from webplanner.config import load_config


def test_cyclic_default_includes_raise_a_useful_error(tmp_path):
    (tmp_path / "a.yaml").write_text("defaults: [b.yaml]\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("defaults: [a.yaml]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cycl"):
        load_config(tmp_path / "a.yaml")


def test_shared_defaults_are_not_treated_as_cycles(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 7\n", encoding="utf-8")
    (tmp_path / "a.yaml").write_text("defaults: [base.yaml]\nexperiment: a\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("defaults: [base.yaml]\nseed: 8\n", encoding="utf-8")
    (tmp_path / "root.yaml").write_text("defaults: [a.yaml, b.yaml]\n", encoding="utf-8")
    cfg = load_config(tmp_path / "root.yaml")
    assert cfg.seed == 8 and cfg.experiment == "a"

