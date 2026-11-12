import pytest

from webplanner.config import load_config


@pytest.mark.parametrize("yaml", ["false", "[]", "experiment", "- experiment"])
def test_config_rejects_nonmapping_documents(tmp_path, yaml):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml, encoding="utf-8")
    with pytest.raises(ValueError, match="experiment.yaml: configuration must be a mapping"):
        load_config(path)


@pytest.mark.parametrize("yaml", ["defaults: base.yaml", "defaults: [42]", "defaults: ['']"])
def test_config_rejects_malformed_base_paths(tmp_path, yaml):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml, encoding="utf-8")
    with pytest.raises(ValueError, match="defaults must be a list of non-empty paths"):
        load_config(path)


def test_config_rejects_override_that_replaces_a_section():
    with pytest.raises(ValueError, match="configuration section 'data' must be a mapping"):
        load_config(overrides=["data=null"])


def test_empty_config_document_preserves_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("# no overrides\n", encoding="utf-8")
    assert load_config(path) == load_config()
