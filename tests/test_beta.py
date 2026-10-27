import numpy as np
import pytest
from scipy.sparse import csr_matrix

from vqd_nuclear.validation import approximate_betas, exact_betas, resolve_beta_comparison


@pytest.mark.parametrize("energies,expected", [
    ([-8.0], [0.0]), ([-8.0, -6.0], [0.0, 4.0]),
    ([-3.0, 1.0, 5.0], [0.0, 8.0, 16.0]), ([-3.0, -3.0, 5.0], [0.0, 0.0, 16.0]),
])
def test_exact_beta_uses_each_target_gap(energies, expected):
    assert exact_betas(energies, len(energies))["betas"] == pytest.approx(expected)


@pytest.mark.parametrize("sparse", [False, True])
def test_approximate_beta_uses_sorted_diagonal_and_absolute_off_diagonal_row_sums(sparse, monkeypatch):
    matrix = np.array([[5.0, -2.0, 1.0], [-2.0, -3.0, -4.0], [1.0, -4.0, 1.0]])
    matrix = csr_matrix(matrix) if sparse else matrix
    original = matrix.copy()
    monkeypatch.setattr(np.linalg, "eigvalsh", lambda *args: pytest.fail("Approximate beta diagonalizes the matrix"))
    if sparse:
        monkeypatch.setattr(csr_matrix, "toarray", lambda *args, **kwargs: pytest.fail("Sparse matrix was densified"))
    result = approximate_betas(matrix, 3)
    assert result["diagonal_energies"] == [-3.0, 1.0, 5.0]
    assert result["B"] == pytest.approx(6.0)
    assert result["betas"] == pytest.approx([0.0, 16.0, 20.0])
    if sparse:
        assert (matrix != original).nnz == 0
    else:
        np.testing.assert_array_equal(matrix, original)


def test_diagonal_degeneracies_are_not_removed():
    result = approximate_betas(np.diag([7.0, -2.0, -2.0]), 3)
    assert result["B"] == 0.0
    assert result["betas"] == [0.0, 0.0, 9.0]


def test_complex_off_diagonal_entries_use_magnitudes():
    result = approximate_betas([[1.0, 3.0 + 4.0j], [3.0 - 4.0j, -1.0]], 2)
    assert result["B"] == pytest.approx(5.0)
    assert result["betas"] == [0.0, 12.0]


@pytest.mark.parametrize("energies,n_states", [([], 1), ([-1.0], 2), ([-1.0], 0),
    ([-1.0], 1.5), ([-1.0, np.nan], 2), ([np.inf], 1), ([[0.0]], 1), ([2.0, 1.0], 2)])
def test_exact_beta_rejects_invalid_inputs(energies, n_states):
    with pytest.raises(ValueError):
        exact_betas(energies, n_states)


@pytest.mark.parametrize("matrix,n_states", [(np.ones((2, 3)), 2), (np.eye(2), 3),
    (np.eye(2), 0), (np.eye(2), 1.5), (np.array([[np.nan]]), 1),
    (np.array([[0.0, np.inf], [np.inf, 1.0]]), 2), (np.array([[1j]]), 1)])
def test_approximate_beta_rejects_invalid_inputs(matrix, n_states):
    with pytest.raises(ValueError):
        approximate_betas(matrix, n_states)


def test_comparison_has_only_two_calculated_methods():
    matrix = np.diag([-3.0, 1.0, 5.0])
    result = resolve_beta_comparison([-3.0, 1.0, 5.0], matrix, 3)
    assert result["exact"]["betas"] == [0.0, 8.0, 16.0]
    assert result["approximate"]["betas"] == [0.0, 4.0, 8.0]


def test_ground_state_uses_no_penalty_with_either_method():
    result = resolve_beta_comparison([-8.0], np.array([[-8.0, 2.0], [2.0, 1.0]]), 1)
    assert result["exact"]["betas"] == result["approximate"]["betas"] == [0.0]
