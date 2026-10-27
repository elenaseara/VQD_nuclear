from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qibo")
pytest.importorskip("pynucshell")

from vqd_nuclear import FullQuantumLimits, NumericalConfig, ProblemConfig, RunConfig, vqd


@pytest.mark.parametrize("statevector,circuit", [(True, False), (False, True), (True, True)])
def test_each_route_runs_independent_beta_families_per_target(monkeypatch, statevector, circuit):
    run = RunConfig(n_states=3, run_statevector=statevector, run_full_quantum_circuit=circuit)
    matrix = np.array([[5.0, -2.0, 1.0], [-2.0, -3.0, -4.0], [1.0, -4.0, 1.0]])
    energies = np.linalg.eigvalsh(matrix)
    context = SimpleNamespace(run=run, hamiltonian=SimpleNamespace(matrix=matrix, exact_energies=energies),
                              measurements=SimpleNamespace(clear_cache=lambda: None))
    monkeypatch.setattr(vqd, "build_context", lambda *args, **kwargs: context)
    seeds, calls = [], []
    monkeypatch.setattr(vqd, "set_reproducible_seed", seeds.append)

    def adapt(context, *, previous_ansatze, betas, cost_mode, **options):
        state = len(previous_ansatze)
        assert len(betas) == state
        if state:
            assert betas == [betas[0]] * state
            assert len({r["beta_method"] for r in previous_ansatze}) == 1
            assert all(r["cost_mode"] == cost_mode for r in previous_ansatze)
        result = {"state_index": state, "beta": betas[0] if betas else 0.0, "cost_mode": cost_mode}
        calls.append((result, list(previous_ansatze)))
        return result

    monkeypatch.setattr(vqd, "run_adapt_vqd", adapt)
    output = vqd.run_workflow(ProblemConfig(), run, NumericalConfig(), FullQuantumLimits())
    for key, enabled in [("statevector_results", statevector), ("full_quantum_circuit_results", circuit)]:
        results = output[key]
        assert len(results) == (6 if enabled else 0)
        if enabled:
            assert [r["beta_method"] for r in results] == ["exact"] * 3 + ["approximate"] * 3
            assert [r["state_index"] for r in results] == [0, 1, 2, 0, 1, 2]
            assert [r["beta"] for r in results[:3]] == pytest.approx(2 * (energies - energies[0]))
            assert [r["beta"] for r in results[3:]] == [0.0, 16.0, 20.0]
    for result, previous in calls:
        assert all(r["beta_method"] == result["beta_method"] for r in previous)
    assert seeds == [run.seed] * (2 * (int(statevector) + int(circuit)))
