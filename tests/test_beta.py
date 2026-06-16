import numpy as np
import pytest

from vqd_nuclear.validation import (
    beta_from_first_excitation_gap,
    beta_from_largest_requested_gap,
    resolve_beta,
    validate_beta,
)


def test_beta_from_first_excitation_gap():
    beta = beta_from_first_excitation_gap(np.array([-8.0, -6.0]), 2)
    assert beta == pytest.approx(4.0)


def test_beta_from_largest_requested_gap_two_states():
    beta = beta_from_largest_requested_gap(np.array([-8.0, -6.0]), 2)
    assert beta == pytest.approx(4.0)


def test_beta_from_largest_requested_gap_three_states():
    beta = beta_from_largest_requested_gap(np.array([-3.0, 1.0, 5.0]), 3)
    assert beta == pytest.approx(16.0)


def test_resolve_beta_ignores_requested_beta_for_excited_state_run():
    result = resolve_beta(0.1, np.array([-8.0, -6.0]), 2)
    assert result["status"] == "auto_from_largest_requested_gap"
    assert result["requested_beta"] == pytest.approx(0.1)
    assert result["beta"] == pytest.approx(4.0)
    assert result["gap_01"] == pytest.approx(2.0)
    assert result["max_requested_gap"] == pytest.approx(2.0)


def test_resolve_beta_uses_largest_requested_gap_for_three_states():
    result = resolve_beta(0.1, np.array([-3.0, 1.0, 5.0]), 3)
    assert result["status"] == "auto_from_largest_requested_gap"
    assert result["requested_beta"] == pytest.approx(0.1)
    assert result["beta"] == pytest.approx(16.0)
    assert result["gap_01"] == pytest.approx(4.0)
    assert result["max_requested_gap"] == pytest.approx(8.0)


def test_single_state_run_does_not_use_beta():
    result = validate_beta(3.0, np.array([-8.0]), 1)
    assert result["status"] == "not_needed"
    assert result["beta"] == pytest.approx(0.0)
    assert result["max_requested_gap"] is None


def test_beta_requires_two_exact_energies_for_legacy_excited_state_rule():
    with pytest.raises(ValueError):
        beta_from_first_excitation_gap(np.array([-8.0]), 2)


def test_beta_requires_requested_exact_energies_for_multistate_rule():
    with pytest.raises(ValueError):
        beta_from_largest_requested_gap(np.array([-8.0, -6.0]), 3)
