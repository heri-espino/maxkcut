# Read-only interactive SDP/EVA geometry explorer (Windows PowerShell).
# Safe from a separate worktree; never starts an SDP/EVA solve.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = Join-Path $root "src"

python -c "import streamlit, plotly, sklearn, networkx; import maxkcut_viz.geometry_viz"
if ($LASTEXITCODE -ne 0) {
    throw "Visualization dependencies missing. Run: python -m pip install -e '.[viz]'"
}
python -m streamlit run (Join-Path $root "src/maxkcut_viz/geometry_dashboard.py")
if ($LASTEXITCODE -ne 0) {
    throw "Streamlit dashboard exited with an error"
}
