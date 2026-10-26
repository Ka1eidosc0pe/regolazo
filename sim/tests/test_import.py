import importlib


def test_package_imports() -> None:
    assert importlib.import_module("sim") is not None
