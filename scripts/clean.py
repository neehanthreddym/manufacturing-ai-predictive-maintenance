"""Remove only disposable Python cache directories, preserving data and artifacts."""

import shutil
from pathlib import Path

for path in list(Path("src").rglob("__pycache__")) + list(Path("tests").rglob("__pycache__")):
    shutil.rmtree(path)
for name in [".pytest_cache", ".ruff_cache", "htmlcov"]:
    if Path(name).is_dir():
        shutil.rmtree(name)
