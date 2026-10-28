"""Interactive, visualization-only browser for saved SDP/EVA G-set artifacts.

Run from repository root:
    python -m streamlit run src/maxkcut/geometry_dashboard.py

No optimizer is imported/called from this module. All EVA evolution frames
must be pre-recorded; existing sweep logs contain objectives only.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from maxkcut.geometry_viz import (
    DEFAULT_EVA, archived_roundings, discover_eva_cases,
    load_convex_embedding, load_eva_artifacts, load_sweep_log,
    pairwise_cosines, positive_edge_communities, project_vectors,
    replay_kmeans, repo_root, sample_vertices, vertex_metrics,
)

st.set_page_config(page_title="Max-Cut • SDP geometry", layout="wide",
                   page_icon="🔵")


@st.cache_resource(show_spinner=False)
def case_resource(name: str, source: str, updated: int):
    """Read-only; mtime invalidates long-lived Streamlit sessions."""
    return load_eva_artifacts(name, source=Path(source))


@st.cache_data(show_spinner=False)
def partition_resource(name: str, case_json: str, factor_sha: str, source: str):
    return archived_roundings(name, json.loads(case_json), factor_sha, source=Path(source))


@st.cache_data(show_spinner="Projecting the saved solution…")
def projection_resource(name: str, source: str, file_mtime: int, method: str,
                        sample: int, seed: int):
    _g, y, _case, _sha = case_resource(name, source, file_mtime)
    indices = sample_vertices(len(y), sample, seed)
    z, information = project_vectors(y, method, selected=indices, seed=seed)
    return z, indices, information


@st.cache_data(show_spinner="Identifying positive-weight communities…")
def communities_resource(name: str, source: str, file_mtime: int, seed: int):
    graph, _, _, _ = case_resource(name, source, file_mtime)
    return positive_edge_communities(graph, seed=seed)


def scatter_cloud(points, colors, label, indices, *, point_size=3.5, title=None):
    colors = np.asarray(colors)
    category = label in ("Cut side", "K-means cluster", "Positive-edge community")
    if category:
        colors = colors.astype(int)
        palette = ["#008b8e", "#f18f4e", "#8d7ac7", "#aebc42", "#c66b93",
                   "#547cc8", "#9b744f", "#637f6d"]
        n = len(set(colors))
        if n <= 8:
            c = [palette[int(x) % len(palette)] for x in colors]
            scale = None
        else:
            c = colors
            scale = "Turbo"
    else:
        c = colors
        scale = "Viridis"
    trace = go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers", customdata=np.asarray(indices) + 1,
        hovertemplate="Vertex %{customdata}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        marker=dict(size=point_size, opacity=0.85, color=c,
                    colorscale=scale, showscale=not category,
                    colorbar=dict(title=label) if not category else None),
    )
    fig = go.Figure([trace])
    fig.update_layout(
        title=title, height=660, margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(xaxis_title="Dimension 1", yaxis_title="Dimension 2",
                   zaxis_title="Dimension 3", aspectmode="data"),
        uirevision="constant",
    )
    return fig


def selected_colors(graph, name, case, factor_sha, source, mode, method, restart,
                    file_mtime, seed):
    if mode in ("Degree", "Absolute weighted degree", "Signed weighted degree"):
        return vertex_metrics(graph)[mode]
    if mode == "Vertex ID":
        return np.arange(graph.shape[0])
    if mode == "Positive-edge community":
        return communities_resource(name, str(source), file_mtime, seed)
    trials = partition_resource(
        name, json.dumps(case, sort_keys=True), factor_sha, str(source)
    )
    if method not in trials:
        raise ValueError(f"Archived method {method} is not available")
    for number, score, labels in trials[method]:
        if number == restart:
            return labels
    raise ValueError(f"No archived restart {restart}")


def method_choices(trials):
    return sorted(trials, key=lambda m: (not m.startswith("K-MeansNM"), m))


def kmeans_frames_figure(coords, sample, trace, size):
    """Plotly frames replay ROUNDING; vectors/3D coordinates stay fixed."""
    frames = []
    for state in trace:
        frame = go.Frame(
            data=[go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=size, opacity=0.8,
                    color=np.asarray(state["labels"])[sample],
                    cmin=-0.5, cmax=1.5,
                    colorscale=[[0, "#008b8e"], [0.49, "#008b8e"],
                                [0.5, "#f18f4e"], [1, "#f18f4e"]],
                ),
                customdata=sample + 1,
                hovertemplate="Vertex %{customdata}<extra></extra>",
            )],
            name=str(state["step"]),
            traces=[0],
        )
        frames.append(frame)
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=35, b=0),
        scene=dict(aspectmode="data"),
        updatemenus=[dict(type="buttons", showactive=False, x=0.01, y=0,
                          buttons=[
                              dict(label="Play", method="animate",
                                   args=[None, {"frame": {"duration": 500,
                                                           "redraw": True},
                                                "fromcurrent": True,
                                                "transition": {"duration": 0}}]),
                              dict(label="Pause", method="animate",
                                   args=[[None], {"frame": {"duration": 0,
                                                             "redraw": False},
                                                  "mode": "immediate"}]),
                          ])],
        sliders=[dict(
            active=0, pad={"t": 45},
            steps=[dict(
                label=str(frame["step"]), method="animate",
                args=[[str(frame["step"])],
                      {"frame": {"duration": 0, "redraw": True},
                       "mode": "immediate", "transition": {"duration": 0}}]
            ) for frame in trace],
        )],
    )
    return fig


def _case_info(source, name):
    file = repo_root() / source / "cases" / f"{name}.json"
    return file.stat().st_mtime_ns


def main():
    st.title("SDP & EVA geometric explorer")
    st.caption("Read-only research dashboard · saved Gram matrices, EVA factors, and archived cuts · no SDP solving")
    root = repo_root()
    with st.sidebar:
        st.header("Data and visualization")
        source = Path(st.text_input("EVA result directory", str(DEFAULT_EVA)))
        graphs = discover_eva_cases(root, source)
        if not graphs:
            st.error("No completed EVA cases found. Select a directory containing cases/ and factors/.")
            st.stop()
        default_graph = "G22" if "G22" in graphs else graphs[0]
        name = st.selectbox("G-set graph", graphs, index=graphs.index(default_graph))
        projection = st.selectbox("Projection", ["PCA", "t-SNE", "UMAP"])
        limit = st.slider("Maximum plotted vertices", 200, 10000, 2500, step=100)
        seed = st.number_input("Visualization seed", min_value=0, max_value=2_000_000, value=2026)
        size = st.slider("Point size", 1.0, 9.0, 3.0, step=0.5)
        if projection != "PCA":
            st.warning("t-SNE/UMAP distort global distances; interpret shape cautiously.")
            limit = min(limit, 3500)
        st.caption("Individual G-sets use independent coordinate frames. Axis orientation is not comparable between graphs.")

    updated = _case_info(source, name)
    graph, vectors, case, factor_sha = case_resource(name, str(source), updated)
    trials = partition_resource(name, json.dumps(case, sort_keys=True),
                                factor_sha, str(source))
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Vertices", f"{graph.shape[0]:,}")
    mcol2.metric("Edges", f"{graph.nnz // 2:,}")
    mcol3.metric("Factor dimension", vectors.shape[1])
    mcol4.metric("EVA sweeps", case["factor_sweeps"])
    st.caption(f"Solver: EVA factorized vector relaxation (not a certified convex SDP optimum) · "
               f"Graph SHA256: {case['graph_sha256'][:20]}…")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "3D geometry", "Clustering replay", "EVA progress",
        "Compare G-sets", "Diagnostics"
    ])

    with tab1:
        left, right = st.columns([4, 1])
        with right:
            source_type = st.radio("Coordinates", [
                "Saved EVA vectors", "Cached convex SDP Gram matrix"
            ])
            coloring = st.selectbox("Color by", [
                "Cut side", "K-means cluster", "Degree",
                "Absolute weighted degree", "Signed weighted degree",
                "Positive-edge community", "Vertex ID",
            ])
            methods = method_choices(trials)
            chosen_method = st.selectbox("Archived rounding method", methods)
            possible = [row[0] for row in trials[chosen_method]]
            best = max(trials[chosen_method], key=lambda row: row[1])
            chosen_restart = st.selectbox(
                "Restart", possible,
                index=possible.index(best[0]) if best[0] in possible else 0,
            )
            current_score = next(row[1] for row in trials[chosen_method]
                                 if row[0] == chosen_restart)
            st.metric("Archived weighted cut", f"{current_score:,.0f}")
            st.caption("Cut labels are taken from saved archives, not recalculated.")
        with left:
            if source_type == "Saved EVA vectors":
                coords, sample, info = projection_resource(
                    name, str(source), updated, projection, limit, seed
                )
            else:
                try:
                    if projection != "PCA":
                        st.warning("Convex cache supports top-three Gram eigenvectors, not nonlinear embeddings.")
                    _g, full_coords, info = load_convex_embedding(name, root)
                    sample = sample_vertices(len(full_coords), limit, seed)
                    coords = full_coords[sample]
                except (OSError, ValueError, MemoryError) as exc:
                    st.warning(f"Cached convex SDP unavailable for this graph: {exc}")
                    st.stop()
            try:
                selected = selected_colors(graph, name, case, factor_sha, source,
                                           coloring, chosen_method, chosen_restart,
                                           updated, seed)
            except (ImportError, ValueError) as exc:
                st.warning(str(exc))
                selected = vertex_metrics(graph)["Degree"]
                coloring = "Degree"
            fig = scatter_cloud(coords, selected[sample], coloring, sample,
                                point_size=size,
                                title=f"{name} · {source_type} · {projection if source_type.startswith('Saved') else 'Gram spectral'}")
            st.plotly_chart(fig, use_container_width=True)
            if source_type == "Cached convex SDP Gram matrix":
                st.info("Cut coloring here comes from an EVA rounding and is overlaid on convex SDP coordinates. "
                        "It is NOT the convex solver's own rounded partition.")
            else:
                st.caption(f"PCA explained variance (top 3): "
                           f"{sum(info.get('explained_variance_ratio', [])):.1%}"
                           if projection == "PCA" else str(info))
            data = pd.DataFrame({
                "vertex": sample + 1, "x": coords[:, 0],
                "y": coords[:, 1], "z": coords[:, 2],
                coloring: selected[sample],
            })
            st.download_button("Export projected coordinates (CSV)",
                               data.to_csv(index=False).encode("utf8"),
                               file_name=f"{name}-{projection.lower()}-3d.csv",
                               mime="text/csv")

    with tab2:
        st.subheader("How clustering changes the saved vector partition")
        st.write("Replays spherical k-means only. **The underlying EVA factor is never optimized or changed.**")
        method = st.selectbox("K-means initialization",
                              ["K-MeansNM_EVA", "K-Means2N_EVA"],
                              key="replay_method")
        run = st.number_input("Restart index", min_value=0, max_value=999,
                              value=0, key="replay_restart")
        with st.spinner("Replaying clustering from the saved factor…"):
            trace = replay_kmeans(vectors, graph, name, method,
                                  restart=int(run),
                                  seed=case["provenance"]["rounding_seed"])
            coords, sample, _ = projection_resource(name, str(source), updated,
                                                       "PCA", limit, seed)
        st.plotly_chart(kmeans_frames_figure(coords, sample, trace, size),
                        use_container_width=True)
        trajectory = pd.DataFrame([
            {"Iteration": f["step"], "Current cut": f["cut"],
             "Best cut so far": f["best_cut"]} for f in trace
        ])
        st.line_chart(trajectory.set_index("Iteration"))
        st.caption("Each clustering iteration may reduce the actual cut; K-MeansNM "
                   "preserves the best cut visited, including its GW-like initialization.")

    with tab3:
        st.subheader("Ángel's EVA optimization: recorded objective history")
        rows = load_sweep_log(name, root, source)
        if rows:
            history = pd.DataFrame(rows).sort_values("sweep")
            st.line_chart(history.set_index("sweep")[["objective"]])
            if "max_angle" in history:
                st.line_chart(history.set_index("sweep")[["max_angle"]])
            st.dataframe(history.tail(12), use_container_width=True, hide_index=True)
            st.download_button("Export EVA sweep history (CSV)",
                               history.to_csv(index=False).encode(),
                               file_name=f"{name}-eva-history.csv")
        else:
            st.info("No intermediate objective history is available for this graph.")
        st.warning("The existing logs contain scalar objective/angle measurements, not "
                   "intermediate vertex coordinates Y(t). Therefore a genuine "
                   "3D animation of EVA's moving vectors cannot be reconstructed "
                   "without rerunning EVA. This dashboard does not rerun it.")

    with tab4:
        st.subheader("Separate views across G-set instances")
        available = discover_eva_cases(root, source)
        defaults = [g for g in ("G11", "G22", "G55", "G60") if g in available]
        selected_graphs = st.multiselect(
            "Choose up to six graphs", available,
            default=defaults or available[:3], max_selections=6,
        )
        for offset in range(0, len(selected_graphs), 2):
            cols = st.columns(2)
            for col, graph_name in zip(cols, selected_graphs[offset:offset+2]):
                with col:
                    try:
                        mtime = _case_info(source, graph_name)
                        g, y, record, sha = case_resource(graph_name, str(source), mtime)
                        coordinates, indices, extra = projection_resource(
                            graph_name, str(source), mtime, "PCA", min(limit, 3000), seed
                        )
                        degrees = vertex_metrics(g)["Degree"]
                        f = scatter_cloud(coordinates, degrees[indices], "Degree", indices,
                                          point_size=max(2., size-1),
                                          title=f"{graph_name} · n={len(y):,} · rank={y.shape[1]}")
                        f.update_layout(height=440)
                        st.plotly_chart(f, use_container_width=True, key=f"gallery-{graph_name}")
                        st.caption(f"EVA objective: {record['objective_vector_relaxation']:,.3f}; "
                                   f"PCA variance: {sum(extra['explained_variance_ratio']):.1%}")
                    except (OSError, ValueError) as exc:
                        st.warning(f"{graph_name}: {exc}")
        st.caption("Each subplot is fit independently. These are descriptive comparisons, "
                   "not a jointly aligned embedding or a shared sphere.")

    with tab5:
        st.subheader("Geometry and rounding distributions")
        vals = pairwise_cosines(vectors, number=12000, seed=seed)
        spectrum = np.linalg.eigvalsh(vectors.T @ vectors)[::-1]
        a, b = st.columns(2)
        with a:
            st.plotly_chart(go.Figure(go.Histogram(x=vals, nbinsx=55)).update_layout(
                title="Sampled pairwise cosine similarity", xaxis_title="Cosine",
                height=350), use_container_width=True)
        with b:
            st.plotly_chart(go.Figure(go.Scatter(y=spectrum[:40], mode="lines+markers")).update_layout(
                title="Top factor-Gram eigenvalues", xaxis_title="Eigenvalue rank",
                height=350), use_container_width=True)
        scores = [{"method": method, "restart": restart, "cut": score}
                  for method, samples in trials.items()
                  for restart, score, _ in samples]
        stats = pd.DataFrame(scores)
        st.dataframe(stats.groupby("method").cut.agg(["count", "mean", "std", "max"])
                     .sort_values("max", ascending=False),
                     use_container_width=True)
        st.plotly_chart(go.Figure([
            go.Box(y=group["cut"], name=method, boxpoints=False)
            for method, group in stats.groupby("method")
        ]).update_layout(title="Archived rounding distributions",
                         yaxis_title="Weighted cut value", height=430),
                         use_container_width=True)
        st.caption("Pairwise cosines are sampled, not a full n×n computation. "
                   "The factor spectrum uses YᵀY, which shares nonzero eigenvalues with YYᵀ.")


if __name__ == "__main__":
    main()
