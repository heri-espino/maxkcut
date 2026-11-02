"""Guard against missing pwsh on the university Windows 11 self-hosted runner."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORKFLOWS=[
    ".github/workflows/experiment025-seed-round.yml",
    ".github/workflows/experiment025-seed-shard.yml",
    ".github/workflows/experiment025-quality-first.yml",
]


def test_windows_jobs_use_inbox_powershell_not_missing_pwsh():
    for path in WORKFLOWS:
        y=(ROOT/path).read_text(encoding="utf-8")
        assert "shell: powershell" in y, path
        assert "shell: pwsh" not in y, path
        assert "MAXKCUT_CONDA_PREFIX" in y, path
    for path in WORKFLOWS[1:]:
        y=(ROOT/path).read_text(encoding="utf-8")
        assert "RUNNER_NAME -ne 'PCAG103A-13'" in y, path


def test_windows_preflight_gates_scientific_jobs():
    y=(ROOT/WORKFLOWS[0]).read_text(encoding="utf-8")
    assert "  windows_preflight:" in y
    assert "    needs: [validate, windows_preflight]" in y
    assert "experiment025-windows-preflight.ps1" in y
    assert y.index("  windows_preflight:") < y.index("  batch0:")


def test_conda_python_resolution_is_non_destructive_and_reproducible():
    runner=(ROOT/"scripts/experiment025-vm.ps1").read_text(encoding="utf-8")
    smoke=(ROOT/"scripts/experiment025-windows-preflight.ps1").read_text(encoding="utf-8")
    for script in [runner,smoke]:
        assert "conda env list --json" in script
        assert "MAXKCUT_CONDA_PREFIX" in script
        assert "python.exe" in script
        assert "Library\\bin" in script
        assert "CONDA_PREFIX" in script
    assert "conda activate maxkcut" in runner
    assert "python -m maxkcut.experiment025_bestcut" not in smoke
    assert "Experiment025" in smoke
