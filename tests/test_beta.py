import numpy as np
import pytest

from vqd_nuclear.validation import approximate_betas, exact_betas, resolve_beta_comparison


@pytest.mark.parametrize("energies,n_states,expected", [
    ([-8.0], 1, [0.0]),
    ([-8.0, -6.0], 2, [0.0, 4.0]),
    ([-3.0, 1.0, 5.0], 3, [0.0, 8.0, 16.0]),
    ([-3.0, -3.0, 5.0], 3, [0.0, 0.0, 16.0]),
])
def test_exact_betas_use_each_target_gap(energies, n_states, expected):
    assert exact_betas(np.array(energies), n_states)["betas"] == pytest.approx(expected)


@pytest.mark.parametrize("energies,n_states", [
    ([-8.0], 2), ([], 1), ([-8.0], 0), ([-8.0, np.nan], 2), ([-np.inf], 1),
])
def test_exact_betas_reject_invalid_inputs(energies, n_states):
    with pytest.raises(ValueError):
        exact_betas(np.array(energies), n_states)


@pytest.mark.parametrize("sparse", [False, True])
def test_approximate_betas_sort_diagonal_and_exclude_diagonal_from_bound(sparse):
    from scipy.sparse import csr_matrix

    matrix = np.array([[5.0, -2.0, 1.0], [-2.0, -3.0, -4.0], [1.0, -4.0, 1.0]])
    matrix = csr_matrix(matrix) if sparse else matrix
    original = matrix.copy()
    result = approximate_betas(matrix, 3)
    assert result["diagonal_energies"] == [-3.0, 1.0, 5.0]
    assert result["B"] == pytest.approx(6.0)
    assert result["betas"] == pytest.approx([0.0, 16.0, 20.0])
    np.testing.assert_array_equal(matrix.toarray() if sparse else matrix,
                                  original.toarray() if sparse else original)


def test_approximate_betas_preserve_degenerate_diagonal_entries():
    result = approximate_betas(np.diag([7.0, -2.0, -2.0]), 3)
    assert result["B"] == 0.0
    assert result["betas"] == [0.0, 0.0, 9.0]


def test_approximate_betas_use_complex_off_diagonal_magnitudes():
    result = approximate_betas(np.array([[1.0, 3.0 + 4.0j], [3.0 - 4.0j, -1.0]]), 2)
    assert result["B"] == pytest.approx(5.0)
    assert result["betas"] == pytest.approx([0.0, 12.0])


@pytest.mark.parametrize("matrix,n_states", [
    (np.ones((2, 3)), 2), (np.eye(2), 3), (np.eye(2), 0),
    (np.array([[np.nan]]), 1), (np.array([[0.0, np.inf], [np.inf, 1.0]]), 2),
    (np.array([[1.0j]]), 1),
])
def test_approximate_betas_reject_invalid_inputs(matrix, n_states):
    with pytest.raises(ValueError):
        approximate_betas(matrix, n_states)


def test_comparison_uses_per_target_exact_gap():
    matrix = np.array([[5.0, -2.0, 1.0], [-2.0, -3.0, -4.0], [1.0, -4.0, 1.0]])
    exact = np.linalg.eigvalsh(matrix)
    result = resolve_beta_comparison(exact, matrix, 3)
    assert result["exact"]["betas"] == pytest.approx(2.0 * (exact - exact[0]))
    assert result["approximate"]["betas"] == [0.0, 16.0, 20.0]


def test_comparison_ground_state_does_not_use_beta():
    result = resolve_beta_comparison(np.array([-8.0]), np.array([[-8.0]]), 1)
    assert result["exact"]["betas"] == [0.0]
    assert result["approximate"]["betas"] == [0.0]
