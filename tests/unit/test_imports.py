from __future__ import annotations


def test_package_imports() -> None:
    import inference_engine

    assert inference_engine is not None


def test_api_app_imports() -> None:
    from inference_engine.adapters.api.app import app

    assert app.title == "InferenceLedger Reference Executor"
