"""app 模块单测：应用工厂与 OpenAPI 文档可用性。"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

import simserver
from simserver import create_app


def test_package_exports_factory() -> None:
    assert simserver.create_app is create_app


def test_factory_returns_independent_instances() -> None:
    first, second = create_app(), create_app()
    assert isinstance(first, FastAPI)
    assert first is not second


def test_openapi_served() -> None:
    response = TestClient(create_app()).get("/openapi.json")
    assert response.status_code == 200
    info = response.json()["info"]
    assert info["title"] == "InferenceSimulator"
    assert info["version"] == simserver.__version__
