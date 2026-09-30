import io
import zipfile

import pytest

from predictmaint.cli import main
from predictmaint.config import RiskConfig, load_config
from predictmaint.data.download import download_data


def test_config(tmp_path):
    file = tmp_path / "config.yaml"
    file.write_text("random_seed: 7")
    assert load_config(file).random_seed == 7
    for text in [
        "- list",
        "split: {test_size: 0.8, validation_size: 0.8}",
        "a: [",
        "random_sead: 1",
    ]:
        file.write_text(text)
        with pytest.raises(ValueError):
            load_config(file)
    with pytest.raises(ValueError):
        load_config(tmp_path / "missing")
    with pytest.raises(ValueError):
        RiskConfig(low_max=0.9)


def test_download(tmp_path, monkeypatch):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("ai4i2020.csv", "test csv")

    class Response(io.BytesIO):
        pass

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(stream.getvalue()))
    path = tmp_path / "raw.csv"
    assert download_data(path).read_text() == "test csv"

    def fail(*a, **k):
        raise OSError("network")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    assert download_data(path) == path
    with pytest.raises(RuntimeError):
        download_data(path, True)


def test_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("random_seed: 42")
    with pytest.raises(FileNotFoundError):
        main(["evaluate", "--config", "config.yaml"])
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/final_metrics.json").write_text('{"measured":true}')
    main(["evaluate", "--config", "config.yaml"])
    assert "measured" in capsys.readouterr().out
