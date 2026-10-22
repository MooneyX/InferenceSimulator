"""本地部署的只读任务检查及 PowerShell 停服前保护。"""

import json
import os
import runpy
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "deploy-local-state.py"
DEPLOY = ROOT / "deploy-local.ps1"


@pytest.fixture
def state(monkeypatch):
    monkeypatch.delenv("SIMULATOR_CACHE_PATH", raising=False)
    monkeypatch.delenv("SIMULATOR_JOBS_PATH", raising=False)
    return runpy.run_path(str(CHECKER))


def write_jobs(path: Path, statuses: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE jobs (summary TEXT)")
        for status in statuses:
            job = {
                "id": status,
                "kind": "scan",
                "status": status,
                "completed": 1,
                "failed": 1,
                "total": 3,
            }
            connection.execute("INSERT INTO jobs VALUES (?)", (json.dumps(job),))


def test_jobs_path_matches_server_config(state, monkeypatch) -> None:
    assert state["jobs_path"]() == Path(".simulation-cache/jobs.sqlite3")
    monkeypatch.setenv("SIMULATOR_CACHE_PATH", "custom/results.sqlite3")
    assert state["jobs_path"]() == Path("custom/jobs.sqlite3")
    monkeypatch.setenv("SIMULATOR_JOBS_PATH", "history/tasks.sqlite3")
    assert state["jobs_path"]() == Path("history/tasks.sqlite3")
    monkeypatch.setenv("SIMULATOR_JOBS_PATH", "")
    assert state["jobs_path"]() == Path("custom/jobs.sqlite3")


@pytest.mark.parametrize("status", ["queued", "running"])
def test_active_jobs_block_without_writing(state, tmp_path, monkeypatch, capsys, status) -> None:
    path = tmp_path / "只读任务.sqlite3"
    write_jobs(path, [status, "completed"])
    before = path.read_bytes()
    monkeypatch.setenv("SIMULATOR_JOBS_PATH", str(path))
    assert state["main"]() == 2
    output = capsys.readouterr().out
    assert status in output
    assert "进度 2/3" in output
    assert path.read_bytes() == before


def test_terminal_jobs_allow_deployment(state, tmp_path, monkeypatch) -> None:
    path = tmp_path / "jobs.sqlite3"
    write_jobs(path, ["completed", "failed", "cancelled", "interrupted"])
    monkeypatch.setenv("SIMULATOR_JOBS_PATH", str(path))
    before = path.read_bytes()
    assert state["main"]() == 0
    assert path.read_bytes() == before


def test_empty_database_allows_deployment(state, tmp_path, monkeypatch) -> None:
    path = tmp_path / "jobs.sqlite3"
    write_jobs(path, [])
    monkeypatch.setenv("SIMULATOR_JOBS_PATH", str(path))
    assert state["main"]() == 0


@pytest.mark.parametrize("problem", ["missing", "corrupt", "schema", "json", "status"])
def test_unreadable_state_blocks_without_creating_database(
    state,
    tmp_path,
    monkeypatch,
    capsys,
    problem,
) -> None:
    path = tmp_path / "jobs.sqlite3"
    if problem == "corrupt":
        path.write_bytes(b"not a sqlite database")
    elif problem == "schema":
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE unrelated (id TEXT)")
    elif problem in ("json", "status"):
        write_jobs(path, ["unknown"])
        if problem == "json":
            with sqlite3.connect(path) as connection:
                connection.execute("UPDATE jobs SET summary = 'invalid json'")
    before = path.read_bytes() if path.exists() else None
    monkeypatch.setenv("SIMULATOR_JOBS_PATH", str(path))
    assert state["main"]() == 1
    assert "无法确认本地任务状态" in capsys.readouterr().err
    assert (path.read_bytes() if path.exists() else None) == before


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 本地部署脚本")
@pytest.mark.parametrize("status", [None, "completed", "queued", "running", "missing"])
def test_powershell_gate_before_stopping_services(tmp_path, status) -> None:
    path = tmp_path / "jobs.sqlite3"
    if status not in (None, "missing"):
        write_jobs(path, [status])
    environment = {**os.environ, "SIMULATOR_JOBS_PATH": str(path)}
    root = str(ROOT).replace("'", "''")
    python = sys.executable.replace("'", "''")
    script = str(DEPLOY).replace("'", "''")
    backend = "" if status is None else "[pscustomobject]@{ProcessId=123}"
    command = (
        "$ErrorActionPreference='Stop'; "
        "$tokens=$null; $errors=$null; "
        f"$ast=[System.Management.Automation.Language.Parser]::ParseFile('{script}', "
        "[ref]$tokens, [ref]$errors); "
        "if ($errors.Count) { throw ($errors | Out-String) }; "
        "$function=$ast.Find({param($node) "
        "$node -is [System.Management.Automation.Language.FunctionDefinitionAst] "
        "-and $node.Name -eq 'Assert-NoActiveJobs'}, $true); "
        "Invoke-Expression $function.Extent.Text; "
        f"$root='{root}'; function Get-ProjectService {{ {backend} }}; "
        "function Join-Path($Path, $ChildPath) { "
        rf"if ($ChildPath -eq '.venv\Scripts\python.exe') {{ return '{python}' }}; "
        r"Microsoft.PowerShell.Management\Join-Path $Path $ChildPath }; "
        "$stops=$ast.FindAll({param($node) "
        "$node -is [System.Management.Automation.Language.CommandAst] "
        "-and $node.GetCommandName() -eq 'Stop-ProjectService'}, $true); "
        "$check=$ast.Find({param($node) "
        "$node -is [System.Management.Automation.Language.CommandAst] "
        "-and $node.GetCommandName() -eq 'Assert-NoActiveJobs'}, $true); "
        "foreach ($stop in $stops) { "
        "if ($stop.Extent.StartOffset -lt $check.Extent.StartOffset) "
        "{ throw 'service stops before task check' } }; "
        "try { Assert-NoActiveJobs; exit 0 } catch { exit 1 }"
    )
    result = subprocess.run(
        [shutil.which("powershell.exe"), "-NoProfile", "-Command", command],
        env=environment,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == (0 if status in (None, "completed") else 1), result.stderr
