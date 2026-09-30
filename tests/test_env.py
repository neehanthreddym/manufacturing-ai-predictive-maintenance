import os

from predictmaint.service.app import create_app


def test_app_loads_working_directory_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EXPLANATION_MODE", raising=False)
    (tmp_path / ".env").write_text("EXPLANATION_MODE=groq\n")
    create_app()
    assert os.getenv("EXPLANATION_MODE") == "groq"


def test_shell_environment_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EXPLANATION_MODE", "template")
    (tmp_path / ".env").write_text("EXPLANATION_MODE=groq\n")
    create_app()
    assert os.getenv("EXPLANATION_MODE") == "template"


def test_missing_env_is_optional(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert create_app() is not None
