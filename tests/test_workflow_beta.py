"""Check automatic penalties and independent deflation in the workflow."""
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from vqd_nuclear import FullQuantumLimits, NumericalConfig, ProblemConfig, RunConfig

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("qibo") is None or importlib.util.find_spec("pynucshell") is None,
    reason="qibo and pynucshell are needed to import the workflow",
)


@pytest.mark.parametrize("statevector,full_quantum", [(True, False), (False, True), (True, True)])
def test_workflow_compares_both_betas_per_state(monkeypatch, statevector, full_quantum):
    from vqd_nuclear import vqd

    run = RunConfig(n_states=3, run_statevector=statevector,
                    run_full_quantum_circuit=full_quantum)
    matrix = np.array([[5.0, -2.0, 1.0], [-2.0, -3.0, -4.0], [1.0, -4.0, 1.0]])
    exact_energies = np.linalg.eigvalsh(matrix)
    context = SimpleNamespace(
        run=run, hamiltonian=SimpleNamespace(matrix=matrix, exact_energies=exact_energies),
        measurements=SimpleNamespace(clear_cache=lambda: None),
    )
    monkeypatch.setattr(vqd, "build_context", lambda *args, **kwargs: context)
    seeds = []
    monkeypatch.setattr(vqd, "set_reproducible_seed", seeds.append)
    calls = []

    def fake_adapt(context, *, previous_ansatze, betas, cost_mode, **options):
        state = len(previous_ansatze)
        # Each family can only see its own previous results.
        if state:
            assert previous_ansatze[0]["cost_mode"] == cost_mode
            assert len({r["beta_method"] for r in previous_ansatze}) == 1
        calls.append((cost_mode, state, list(betas), [id(r) for r in previous_ansatze]))
        return {"state_index": state, "cost_mode": cost_mode,
                "beta": betas[0] if betas else 0.0}

    monkeypatch.setattr(vqd, "run_adapt_vqd", fake_adapt)
    output = vqd.run_workflow(ProblemConfig(), run, NumericalConfig(), FullQuantumLimits())
    for key, enabled in [("statevector_results", statevector),
                         ("full_quantum_circuit_results", full_quantum)]:
        results = output[key]
        assert len(results) == (6 if enabled else 0)
        if not enabled:
            continue
        assert [r["beta_method"] for r in results] == ["exact"] * 3 + ["approximate"] * 3
        assert [r["state_index"] for r in results] == [0, 1, 2, 0, 1, 2]
        assert [r["beta"] for r in results[:3]] == pytest.approx(2 * (exact_energies - exact_energies[0]))
        assert [r["beta"] for r in results[3:]] == [0.0, 16.0, 20.0]
        for cost_mode, state, betas, previous_ids in calls:
            if cost_mode == results[0]["cost_mode"] and state:
                assert betas == [betas[0]] * state
                assert previous_ids in [[id(r) for r in results[:state]],
                                        [id(r) for r in results[3:3 + state]]]
    assert seeds == [run.seed] * (2 * (int(statevector) + int(full_quantum)))
