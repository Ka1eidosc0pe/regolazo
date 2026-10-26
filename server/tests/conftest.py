from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.app import create_app
from server.config import Settings
from server.fake_engine import FakeEngine
from server.fixture import write_fixture_recording
from shared.recording import Recording


@pytest.fixture(scope="session")
def recording_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_fixture_recording(tmp_path_factory.mktemp("matches") / "demo-0001", n_frames=500)


@pytest.fixture(scope="session")
def recording(recording_dir: Path) -> Recording:
    return Recording.open(recording_dir)


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine("demo-0001")


@pytest.fixture
def app(recording_dir: Path, engine: FakeEngine) -> FastAPI:
    return create_app(recording_dir, engine, Settings(languages=("en", "es", "fr")))


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
