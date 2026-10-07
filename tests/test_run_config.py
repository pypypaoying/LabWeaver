"""Configuration precedence, source-relative paths, and safe validation."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from labweaver.config import ConfigurationError
from labweaver.run_config import load_intake_config


def _toml(path, content):
    path.write_text("[intake]\n" + content, encoding="utf-8")
    return path


def test_zero_argument_load_uses_builtins_when_default_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_intake_config()
    assert config.csv_path == tmp_path / "examples/data/survey.csv"
    assert config.task is None
    assert config.task_file == tmp_path / "examples/tasks/survey.txt"
    assert config.mode == "live"
    assert config.encoding == "auto"
    assert config.delimiter == "auto"
    assert config.sample_rows == 5
    assert config.output_dir == tmp_path / "runs"
    assert config.env_file is None
    assert config.material_paths == ()
    with pytest.raises(FrozenInstanceError):
        config.mode = "offline"


def test_explicit_missing_file_is_not_silently_defaulted(tmp_path):
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_intake_config(tmp_path / "missing.toml")


def test_public_local_and_cli_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _toml(tmp_path / "labweaver.toml",
          'csv = "public.csv"\ntask = "公开任务"\nmode = "offline"\nsample_rows = 2\n')
    _toml(tmp_path / "labweaver.local.toml",
          'csv = "local.csv"\ntask_file = "local-task.txt"\nsample_rows = 3\n')
    config = load_intake_config(overrides={"csv": "cli.csv", "task": "命令任务", "sample_rows": 0})
    assert config.csv_path == tmp_path / "cli.csv"
    assert config.task == "命令任务"
    assert config.task_file is None
    assert config.mode == "offline"
    assert config.sample_rows == 0


def test_default_file_missing_still_accepts_sibling_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _toml(tmp_path / "labweaver.local.toml", 'task = "本地任务"\nmode = "offline"\n')
    config = load_intake_config()
    assert config.task == "本地任务"
    assert config.task_file is None
    assert config.mode == "offline"


def test_toml_paths_use_config_directory_and_override_paths_use_cwd(tmp_path, monkeypatch):
    directory = tmp_path / "project"
    directory.mkdir()
    config_path = _toml(directory / "custom.toml",
                        'csv = "data.csv"\ntask_file = "task.txt"\noutput_dir = "artifacts"\n'
                        'env_file = "secrets.env"\n')
    monkeypatch.chdir(tmp_path)
    config = load_intake_config(Path("project/custom.toml"))
    assert config.csv_path == directory / "data.csv"
    assert config.task_file == directory / "task.txt"
    assert config.output_dir == directory / "artifacts"
    assert config.env_file == directory / "secrets.env"
    override = load_intake_config(config_path, overrides={
        "csv": Path("selected.csv"), "task_file": "selected-task.txt", "output_dir": "output",
        "env_file": "cli.env",
    })
    assert override.csv_path == tmp_path / "selected.csv"
    assert override.task_file == tmp_path / "selected-task.txt"
    assert override.output_dir == tmp_path / "output"
    assert override.env_file == tmp_path / "cli.env"


def test_defaults_follow_explicit_configuration_directory(tmp_path, monkeypatch):
    folder = tmp_path / "elsewhere"
    folder.mkdir()
    public = _toml(folder / "custom.toml", 'mode = "offline"\n')
    monkeypatch.chdir(tmp_path)
    config = load_intake_config(public)
    assert config.csv_path == folder / "examples/data/survey.csv"
    assert config.task_file == folder / "examples/tasks/survey.txt"
    assert config.output_dir == folder / "runs"


def test_dotenv_is_selected_from_config_directory_without_reading_secrets(tmp_path, monkeypatch):
    folder = tmp_path / "project"
    folder.mkdir()
    public = _toml(folder / "labweaver.toml", 'mode = "offline"\n')
    (folder / ".env").write_text("LLM_API_KEY=do-not-read-test-secret\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config = load_intake_config(public)
    assert config.env_file == folder / ".env"
    assert "do-not-read-test-secret" not in repr(config)


def test_explicit_env_path_is_validated_by_model_configuration_later(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_intake_config(overrides={"env_file": "not-created.env"})
    assert config.env_file == tmp_path / "not-created.env"
    assert not config.env_file.exists()


def test_switching_task_sources_clears_previous_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    public = _toml(tmp_path / "labweaver.toml", 'task = "任务文本"\n')
    config = load_intake_config(public)
    assert config.task == "任务文本"
    assert config.task_file is None
    selected = load_intake_config(public, overrides={"task_file": "other.txt"})
    assert selected.task is None
    assert selected.task_file == tmp_path / "other.txt"


@pytest.mark.parametrize("layer", ["public", "local", "override"])
def test_two_task_sources_in_a_single_layer_are_rejected(tmp_path, monkeypatch, layer):
    monkeypatch.chdir(tmp_path)
    content = 'task = "one"\ntask_file = "two.txt"\n'
    overrides = None
    if layer == "public":
        _toml(tmp_path / "labweaver.toml", content)
    elif layer == "local":
        _toml(tmp_path / "labweaver.local.toml", content)
    else:
        overrides = {"task": "one", "task_file": "two.txt"}
    with pytest.raises(ConfigurationError, match="either 'task' or 'task_file'"):
        load_intake_config(overrides=overrides)


def test_utf8_bom_and_chinese_task_are_supported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "labweaver.toml").write_text(
        '[intake]\ntask = "分析中文问卷"\nencoding = "gb18030"\ndelimiter = ";"\n',
        encoding="utf-8-sig",
    )
    config = load_intake_config()
    assert config.task == "分析中文问卷"
    assert config.encoding == "gb18030"
    assert config.delimiter == ";"


@pytest.mark.parametrize("value", ["tab", "\\t", "\t"])
def test_tab_separator_forms_are_supported(tmp_path, monkeypatch, value):
    monkeypatch.chdir(tmp_path)
    assert load_intake_config(overrides={"delimiter": value}).delimiter == "\t"


@pytest.mark.parametrize("setting,value", [
    ("csv", ""), ("csv", 1), ("csv", None),
    ("task", " "), ("task", ["text"]),
    ("task_file", False), ("env_file", {}), ("output_dir", ""),
    ("mode", "automatic"), ("mode", True),
    ("encoding", ""), ("encoding", 1),
    ("sample_rows", -1), ("sample_rows", True), ("sample_rows", "5"),
    ("delimiter", "::"), ("delimiter", "\n"), ("delimiter", "\r"),
    ("delimiter", '"'), ("delimiter", "\x00"), ("delimiter", False),
])
def test_invalid_settings_are_clear_and_do_not_echo_values(tmp_path, monkeypatch, setting, value):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigurationError) as caught:
        load_intake_config(overrides={setting: value})
    assert setting in str(caught.value)


@pytest.mark.parametrize("layer", ["public", "local", "override"])
def test_unknown_keys_are_rejected_without_echoing_secret_values(tmp_path, monkeypatch, layer):
    monkeypatch.chdir(tmp_path)
    overrides = None
    if layer == "override":
        overrides = {"api_key": "do-not-show-test-secret"}
    else:
        file = "labweaver.toml" if layer == "public" else "labweaver.local.toml"
        _toml(tmp_path / file, 'api_key = "do-not-show-test-secret"\n')
    with pytest.raises(ConfigurationError) as caught:
        load_intake_config(overrides=overrides)
    assert "api_key" in str(caught.value)
    assert "do-not-show-test-secret" not in str(caught.value)


def test_invalid_toml_does_not_echo_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "labweaver.toml").write_text(
        '[intake]\ntask = "do-not-show-test-secret\n', encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="invalid TOML") as caught:
        load_intake_config()
    assert "do-not-show-test-secret" not in str(caught.value)


@pytest.mark.parametrize("document", ['[other]\nmode="offline"\n', 'intake = "invalid"\n'])
def test_invalid_sections_are_rejected(tmp_path, monkeypatch, document):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "labweaver.toml").write_text(document, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_intake_config()


def test_configuration_directory_is_not_a_readable_toml_file(tmp_path):
    with pytest.raises(ConfigurationError, match="could not be read"):
        load_intake_config(tmp_path)


def test_overrides_must_be_a_mapping(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigurationError, match="dictionary"):
        load_intake_config(overrides=[("mode", "offline")])


def test_material_paths_follow_their_configuration_source(tmp_path, monkeypatch):
    folder = tmp_path / "project"
    folder.mkdir()
    public = _toml(folder / "custom.toml", 'materials = ["requirements.md", "methods.pdf"]\n')
    monkeypatch.chdir(tmp_path)
    config = load_intake_config(public)
    assert config.material_paths == (folder / "requirements.md", folder / "methods.pdf")
    selected = load_intake_config(public, overrides={"materials": [Path("chosen.txt")]})
    assert selected.material_paths == (tmp_path / "chosen.txt",)


def test_local_materials_replace_public_and_empty_override_disables_retrieval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _toml(tmp_path / "labweaver.toml", 'materials = ["public.md"]\n')
    _toml(tmp_path / "labweaver.local.toml", 'materials = ["local.txt"]\n')
    assert load_intake_config().material_paths == (tmp_path / "local.txt",)
    assert load_intake_config(overrides={"materials": []}).material_paths == ()


def test_material_configuration_does_not_read_source_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_intake_config(overrides={"materials": ("not-created.pdf",)})
    assert config.material_paths == (tmp_path / "not-created.pdf",)
    assert not config.material_paths[0].exists()


@pytest.mark.parametrize("value", ["one.md", None, False, {}, [None], [True], [""], [1]])
def test_invalid_material_arrays_are_configuration_errors(tmp_path, monkeypatch, value):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigurationError, match="materials"):
        load_intake_config(overrides={"materials": value})


def test_repeated_cli_materials_override_configuration_as_cwd_paths(tmp_path, monkeypatch, capsys):
    from labweaver.cli import main
    import labweaver.runtime.intake as intake_runtime

    monkeypatch.chdir(tmp_path)
    _toml(tmp_path / "labweaver.toml", 'materials = ["old.md"]\n')
    observed = []

    def observe(config):
        observed.append(config)
        return {"status": "completed"}, tmp_path / "run.json"

    monkeypatch.setattr(intake_runtime, "execute_intake", observe)
    assert main(["intake", "--offline", "--material", "first.md", "--material", "second.pdf"]) == 0
    capsys.readouterr()
    assert observed[0].material_paths == (tmp_path / "first.md", tmp_path / "second.pdf")
