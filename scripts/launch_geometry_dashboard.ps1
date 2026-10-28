# Read-only interactive SDP/EVA geometry explorer (Windows PowerShell).
# No SDP/EVA optimization occurs in the visualization process.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = Join-Path $root "src"

python -c "import streamlit, plotly, sklearn, networkx; import maxkcut.geometry_viz"
if ($LASTEXITCODE -ne 0) {
    throw "Visualization dependencies missing. Run: python -m pip install -e '.[visualization]'"
}
python -m streamlit run (Join-Path $root "src/maxkcut/geometry_dashboard.py")
if ($LASTEXITCODE -ne 0) {
    throw "Streamlit dashboard exited with an error"
}
