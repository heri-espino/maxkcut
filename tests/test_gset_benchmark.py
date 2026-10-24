import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse
from scipy.io import savemat

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'gset_benchmark.py'
spec = importlib.util.spec_from_file_location('gset_benchmark', MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


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
