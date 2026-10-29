"""Import smoke tests for local launch."""

import importlib
import subprocess
import sys
from pathlib import Path


def test_import_app_package() -> None:
    importlib.import_module("app")


def test_import_ui_client_backend_client() -> None:
    module = importlib.import_module("app.ui_client")
    assert hasattr(module, "BackendClient")


def test_import_ui_module_with_repo_root_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, r'%s'); import app.ui; import app.ui_client" % root,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_streamlit_entrypoint_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import streamlit_app"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(root)},
    )
    assert result.returncode == 0, result.stderr
