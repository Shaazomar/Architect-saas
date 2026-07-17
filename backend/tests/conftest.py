from __future__ import annotations

import pytest

from app import store
from app.config import settings
from app.devtools.sample_plan import sample_plan_png


@pytest.fixture()
def plan_png() -> bytes:
    return sample_plan_png()


@pytest.fixture()
def isolated_data_dir(tmp_path, monkeypatch):
    """Point the job store at a throwaway directory and reset its connection."""
    store.close()
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    yield tmp_path / "data"
    store.close()
