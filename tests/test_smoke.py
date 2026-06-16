import importlib.util

import pytest


def test_import_package():
    import vqd_nuclear

    assert hasattr(vqd_nuclear, "run_workflow")


@pytest.mark.skipif(importlib.util.find_spec("qibo") is None, reason="qibo is not installed")
@pytest.mark.skipif(importlib.util.find_spec("pynucshell") is None, reason="pynucshell is not installed")
def test_build_context_smoke():
    from vqd_nuclear import NumericalConfig, ProblemConfig, RunConfig
    from vqd_nuclear.vqd import build_context

    problem = ProblemConfig(shell="p", n_valence_protons=1, n_valence_neutrons=1, total_j=0)
    run = RunConfig(n_states=1, run_full_quantum_circuit=False, n_workers=1, seed=123)
    numerical = NumericalConfig(max_adapt_iter=1, max_candidates=2, optimizer_maxiter=5, n_jobs=1)
    context = build_context(problem, run, numerical, qibo_backend="numpy")
    assert context.hamiltonian.n_qubits > 0
    assert len(context.active_operator_indices) > 0
