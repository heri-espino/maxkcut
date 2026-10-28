"""Real saved-factor Streamlit smoke test: no SDP/EVA continuous optimizer."""
from pathlib import Path
import pytest


def test_streamlit_dashboard_real_archived_g22_loads():
    st = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    # This test uses the G22 artifacts already committed in experiment002.
    required = [
        root / "data/G22.txt",
        root / "results/gset/experiment002-eva/cases/G22.json",
        root / "results/gset/experiment002-eva/factors/G22.npz",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("G22 factor not included in this checkout")

    app = AppTest.from_file(
        str(root / "src/maxkcut/geometry_dashboard.py"),
        default_timeout=120,
    )
    app.run()
    assert not app.exception, [entry.message for entry in app.exception]
    assert app.title[0].value == "SDP & EVA geometric explorer"
    assert len(app.tabs) == 6
