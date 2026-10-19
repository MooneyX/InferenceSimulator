"""服务层测试不读写本地部署的结果缓存。"""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_result_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMULATOR_CACHE_PATH", str(tmp_path / "results.sqlite3"))
