import importlib.util
import numpy as np
import pytest
from scipy import sparse
from scipy.io import savemat

import maxkcut as mod


def test_sparse_symmetric_graph_and_cut(tmp_path):
    a = sparse.csr_matrix([[0, 2, 0], [2, 0, -1], [0, -1, 0]])
    file = tmp_path / 'G1.mat'
    savemat(file, {'A': a})
    result = mod.load_graph(file)
    assert result.shape == (3, 3)
    assert np.array_equal(result.toarray(), a.toarray())
    edges = mod.edge_arrays(result)
    assert mod.cut_weight(edges, np.array([0, 1, 1])) == 2
    assert mod.cut_weight(edges, np.array([0, 1, 0])) == 1


def test_triangular_graph(tmp_path):
    a = sparse.csr_matrix([[0, 3, 0], [0, 0, 2], [0, 0, 0]])
    file = tmp_path / 'G2.mat'
    savemat(file, {'W': a})
    assert mod.load_graph(file)[1, 0] == 3
    assert mod.load_graph(file)[2, 1] == 2


def test_ambiguous_data_rejected(tmp_path):
    file = tmp_path / 'G3.mat'
    savemat(file, {'X': np.eye(3), 'Y': np.eye(3)})
    with pytest.raises(ValueError, match='Ambiguous'):
        mod.load_graph(file)


def test_fj_kmeans_and_factorization():
    x = np.array([[1., -1., 1.], [-1., 1., -1.], [1., -1., 1.]])
    v, eig = mod.factorize(x)
    assert eig > -1e-10
    assert np.allclose(v @ v.T, x)
    labels = mod.round_fj(v, 2, np.random.default_rng(123))
    assert labels[0] == labels[2] != labels[1]
    km, iterations = mod.round_spherical_kmeans(v, 2, np.random.default_rng(123))
    assert km[0] == km[2] != km[1]
    assert 1 <= iterations <= 100


def test_reject_non_psd():
    with pytest.raises(ValueError, match='non-PSD'):
        mod.factorize(np.array([[1., 2.], [2., 1.]]))


@pytest.mark.skipif(importlib.util.find_spec('cvxpy') is None, reason='cvxpy not installed')
def test_small_sdp_triangle():
    a = sparse.csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    x, info = mod.solve_sdp(a, 2, 'SCS', 1e-5, 5000)
    assert info['solver_status'] in ('optimal', 'optimal_inaccurate')
    assert abs(info['sdp_objective_primal'] - 2.25) < 1e-2
    assert np.allclose(np.diag(x), 1., atol=1e-3)


def test_kmeans_nm_retains_matching_gw_initial_cut():
    # The same Gaussian seed is used for GW and the antipodal 2-means start.
    v = np.eye(5)
    edges = (np.array([0, 0, 1, 2]), np.array([1, 3, 4, 4]),
             np.array([3.0, 2.0, -1.0, 4.0]))
    seed = 2345
    gw = mod.round_fj(v, 2, np.random.default_rng(seed))
    improved, iterations = mod.round_spherical_kmeans(
        v, 2, np.random.default_rng(seed),
        antipodal=True, edges=edges, keep_best=True)
    assert mod.cut_weight(edges, improved) >= mod.cut_weight(edges, gw)
    assert 1 <= iterations <= 100


def test_antipodal_rejects_general_k():
    with pytest.raises(ValueError, match='k=2'):
        mod.round_spherical_kmeans(
            np.eye(3), 3, np.random.default_rng(1), antipodal=True)


def test_public_api_and_version():
    assert mod.__version__ == '0.1.0'
    for name in ('load_graph', 'edge_arrays', 'cut_weight', 'factorize',
                 'round_fj', 'round_spherical_kmeans', 'solve_sdp', 'benchmark'):
        assert callable(getattr(mod, name))


def test_module_cli_inspection(monkeypatch, capsys, tmp_path):
    import sys
    from maxkcut.cli import main
    a = sparse.csr_matrix([[0, 2], [2, 0]])
    savemat(tmp_path / 'G9.mat', {'A': a})
    monkeypatch.setattr(sys, 'argv', ['maxkcut', '--gset-dir', str(tmp_path),
                                       '--graphs', 'G9', '--inspect-only'])
    main()
    output = capsys.readouterr().out
    assert '"graph": "G9"' in output
    assert '"edges": 1' in output



def test_loader_requests_sparse_array_when_supported(monkeypatch, tmp_path):
    """Current SciPy should not rely on the deprecated sparse-type default."""
    from maxkcut import io

    calls = []

    def fake_loadmat(file, *, simplify_cells, spmatrix):
        calls.append((simplify_cells, spmatrix))
        return {"A": sparse.csr_array([[0, 2], [2, 0]])}

    monkeypatch.setattr(io, "loadmat", fake_loadmat)
    matrix = io.load_graph(tmp_path / "G9.mat")
    assert calls == [(True, False)]
    assert sparse.isspmatrix_csr(matrix)
    assert matrix[0, 1] == 2


def test_loader_preserves_support_for_old_scipy_signature(monkeypatch, tmp_path):
    """Earlier SciPy releases have no explicit spmatrix keyword."""
    from maxkcut import io

    calls = []

    def fake_loadmat(file, *, simplify_cells):
        calls.append(simplify_cells)
        return {"A": sparse.csr_matrix([[0, 2], [2, 0]])}

    monkeypatch.setattr(io, "loadmat", fake_loadmat)
    matrix = io.load_graph(tmp_path / "G9.mat")
    assert calls == [True]
    assert matrix.shape == (2, 2)
