#!/usr/bin/env python3
"""Validation of the "born" sampler (D2) against Qibo on a real nucleus (6Be).

Three subcommands, that can also be chained with `all`:

  exact    No shots. On the real Hamiltonian, pool and ansatz of the nucleus:
           (a) the cached state used by born == the state of the Qibo circuit;
           (b) for EVERY measurement circuit of the energy plan and of every
               G_mu plan: P_born(x) == marginal of the Qibo circuit state;
           (c) penalty circuit (N+1 qubits): (p0, p1) from inner products ==
               |amplitudes|^2 of the Qibo circuit;
           (d) destructive SWAP (2N qubits): P(a, b) == Qibo circuit, and the
               exact mean of (-1)^(a.b) == |<psi_i|psi_k>|^2;
           (e) one candidate scan with each sampler on the SAME engine: same
               circuits and shots counted, and every g^E / g^pen of every
               candidate compatible with the exact value (<z^2> ~ 1, no
               selection bias because ALL candidates are used).
           Points: final states of a JSON (--json) or random ansatze.

  seeds    Launches run_vqd.py with --sampler qibo and --sampler born for the
           same list of seeds (separate processes, optionally in parallel).
           Already finished runs are skipped, so it can be resumed.

  compare  Reads all the JSONs of `seeds` and compares both samplers at two
           levels:
           1. Point by point (independent of the random trajectory): at every
              ADAPT iteration of every run, the reported quantity against its
              EXACT value at the same theta:
                - energy:     (E_reported - E(theta)) / sigma_exact
                - objective:  (obj_reevaluated - obj(theta)) / sigma_exact
                              (excited states: energy + beta * SWAP overlaps)
                - gradient:   (g_selected - g_exact) / sigma_estimated, at the
                              selection point (previous iteration's ansatz).
              Same distribution with both samplers, and <z^2> ~ 1 for energy
              and objective (exact sigma). For the gradient <z^2> may be
              slightly above 1 in both: argmax selection between nearly tied
              candidates (e.g. ops 2 and 4 in 6Be) biases |g| upwards.
           2. Final result per state (depends on the trajectory): fidelity
              with the exact eigenvector, |E(theta) - lambda_k|, layers,
              termination reason, circuits, shots and wall time.
              With 5 + 5 runs this only detects LARGE differences; the
              point-by-point level is the sharp one.

Usage (from anywhere; the repository root is found automatically):

  python check_d2_be6.py exact [--json raw_results_X.json]
  python check_d2_be6.py seeds --out runs_d2 --seeds 1 2 3 4 5 --parallel 2 -- \\
         --energy-shots 2000000 --overlap-shots 2000000 --cobyla-rhoend 1e-2
  python check_d2_be6.py compare runs_d2
  python check_d2_be6.py all --out runs_d2 --seeds 1 2 3 4 5 --parallel 2 -- <run_vqd options>

Everything after `--` goes to run_vqd.py unchanged, after these defaults
(6Be, 3 states, sampled route only, n_jobs = n_workers = 1):
  --shell p --protons 2 --neutrons 0 --J 0 --n-states 3
  --mode full_quantum_circuit --n-jobs 1 --n-workers 1
so any of them can be overridden. --sampler, --seed, --run-label, --results-dir
and --flat-results are set by this script.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# Repository root: the first ancestor of this file that contains vqd_nuclear/.
HERE = Path(__file__).resolve().parent
ROOT = HERE
while not (ROOT / "vqd_nuclear").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
for _p in (HERE, ROOT / "tests", ROOT / "check_scripts", ROOT / "scripts", ROOT):
    if _p.is_dir():
        sys.path.insert(0, str(_p))

DEFAULT_RUN_ARGS = ["--shell", "p", "--protons", "2", "--neutrons", "0", "--J", "0",
                    "--n-states", "3", "--mode", "full_quantum_circuit",
                    "--n-jobs", "1", "--n-workers", "1"]
RESERVED_ARGS = ("--sampler", "--seed", "--run-label", "--results-dir", "--flat-results")
SAMPLERS = ("qibo", "born")
TOL_EXACT = 1e-12

failures = 0


def check(condition: bool, message: str) -> None:
    global failures
    print(("  OK   " if condition else "  FALLO ") + message)
    failures += (not condition)


# =====================================================================
# Context (the only function that needs pynucshell; tests replace it)
# =====================================================================

def make_context(problem: dict, run: dict, numerical: dict):
    """Real VQDContext for the nucleus of `problem`, with the measurement
    strategy of the runs (the energy plan must be the one they used)."""
    from vqd_nuclear.config import NumericalConfig, ProblemConfig, RunConfig
    from vqd_nuclear.vqd import build_context
    p = ProblemConfig(shell=problem["shell"], n_valence_protons=int(problem["n_valence_protons"]),
                      n_valence_neutrons=int(problem["n_valence_neutrons"]),
                      total_j=int(problem["total_j"]))
    r = RunConfig(n_states=int(run.get("n_states", 3)), n_workers=1)
    n = NumericalConfig(n_jobs=1, use_model_space=True,
                        measurement_strategy=numerical.get("measurement_strategy", "paper"))
    return build_context(p, r, n)


BE6 = {"shell": "p", "n_valence_protons": 2, "n_valence_neutrons": 0, "total_j": 0}


# =====================================================================
# exact
# =====================================================================

def marginal(state: np.ndarray, n_qubits: int, measured) -> np.ndarray:
    """P(x) of the `measured` qubits (measured[0] = most significant bit).

    Written with bit operations and bincount, independently of
    sampling.born_probabilities (which uses reshape and transpose).
    """
    p = np.abs(np.asarray(state).ravel()) ** 2
    idx = np.arange(p.size)
    local = np.zeros(p.size, dtype=np.int64)
    for q in measured:
        local = (local << 1) | ((idx >> (n_qubits - 1 - int(q))) & 1)
    return np.bincount(local, weights=p, minlength=2 ** len(measured))


def exact_points(ctx, data: dict | None, n_random: int, rng) -> list[dict]:
    """[{psi: (thetas, ops), previous: [(thetas, ops), ...], beta}] to check."""
    if data is not None:
        for key in ("full_quantum_circuit_results", "statevector_results"):
            results = data.get(key) or []
            if results:
                states = [(list(r["thetas"]), list(r["ops"])) for r in results]
                beta = float(data["run"]["beta"]) if not isinstance(data["run"]["beta"], str) \
                    else float(results[-1].get("beta", 10.0))
                print(f"   Puntos: estados finales de {key} del JSON ({len(states)} estados)")
                return [{"psi": states[k], "previous": states[:k], "beta": beta}
                        for k in range(len(states))]
    n_pool = len(ctx.hamiltonian.qubit_generator_pool)

    def random_ansatz(layers):
        return ([float(x) for x in rng.uniform(-0.6, 0.6, layers)],
                [int(x) for x in rng.integers(0, n_pool, layers)])

    print(f"   Puntos: {n_random} ansatze aleatorios de 3 capas, con 0, 1 y 2 estados previos")
    points = []
    for i in range(n_random):
        points.append({"psi": random_ansatz(3),
                       "previous": [random_ansatz(2) for _ in range(min(i, 2))],
                       "beta": 10.0})
    return points


def run_exact(ctx, points: list[dict], *, scan_shots: int, scan_repeats: int) -> None:
    from vqd_nuclear import sampling as S
    from vqd_nuclear.measurement_plan import apply_circuit
    from vqd_nuclear.measurements import qubit_operator_expectation_from_state

    eng = ctx.measurements
    builder = ctx.ansatz_builder
    n = int(ctx.hamiltonian.n_qubits)
    pool = ctx.hamiltonian.qubit_generator_pool
    n_pool = len(pool)
    plans = [eng.energy_plan] + [eng.gradient_plan(mu) for mu in range(n_pool)]
    n_groups = sum(len(p.all_groups()) for p in plans)
    print(f"   N = {n} qubits, pool de {n_pool} operadores, {n_groups} circuitos de medida "
          f"(energia + {n_pool} planes G_mu)")

    worst = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "dF": 0.0}
    counts = {"b": 0, "c": 0, "d": 0}
    for point in points:
        thetas, ops = point["psi"]
        base = builder.build(thetas, ops)
        psi_qibo = np.asarray(base().state()).ravel()
        psi_born = np.asarray(eng._state(thetas, ops)).ravel()
        worst["a"] = max(worst["a"], float(np.max(np.abs(psi_born - psi_qibo))))

        # (b) every measurement circuit of the energy and gradient plans.
        for plan in plans:
            for group in plan.all_groups():
                circuit = base.copy(deep=True)
                S.add_plan_gates(circuit, group.gates)
                p_qibo = marginal(circuit().state(), n, list(group.support))
                phi = apply_circuit(psi_born, group.gates, n) if group.gates else psi_born
                p_born = S.born_probabilities(phi, n, group.support)
                worst["b"] = max(worst["b"], float(np.max(np.abs(p_qibo - p_born))))
                counts["b"] += 1

        for prev_thetas, prev_ops in point["previous"]:
            psi_j = np.asarray(eng._state(prev_thetas, prev_ops)).ravel()
            inverse = S.inverse_circuit(builder.build(prev_thetas, prev_ops))
            # (c) penalty circuit of every Pauli string of every candidate.
            for mu in range(n_pool):
                for chain, _ in eng.penalty_terms(mu):
                    circuit = S.penalty_gradient_circuit(base, inverse, chain, n, measure=False)
                    amp = np.asarray(circuit().state()).ravel()
                    p0, p1 = S.penalty_probabilities_from_states(psi_born, psi_j, chain, n)
                    worst["c"] = max(worst["c"], abs(p0 - abs(amp[0]) ** 2), abs(p1 - abs(amp[1]) ** 2))
                    counts["c"] += 1
            # (d) destructive SWAP: Qibo circuit of 2N qubits without measurements.
            if 2 * n <= S.MAX_BORN_QUBITS:
                from qibo import gates
                from qibo.models import Circuit
                swap = Circuit(2 * n)
                for gate in base.queue:
                    swap.add(S.copy_gate_with_offset(gate, 0))
                for gate in builder.build(prev_thetas, prev_ops).queue:
                    swap.add(S.copy_gate_with_offset(gate, n))
                for q in range(n):
                    swap.add(gates.CNOT(q, q + n))
                    swap.add(gates.H(q))
                p_qibo = np.abs(np.asarray(swap().state()).ravel()) ** 2
                layer = [g for q in range(n) for g in (("CNOT", q, q + n), ("H", q))]
                phi = apply_circuit(np.kron(psi_born, psi_j), layer, 2 * n)
                p_born = S.born_probabilities(phi, 2 * n, range(2 * n))
                worst["d"] = max(worst["d"], float(np.max(np.abs(p_qibo - p_born))))
                idx = np.arange(p_qibo.size)
                a, b = idx >> n, idx & ((1 << n) - 1)
                parity = np.array([bin(int(x)).count("1") & 1 for x in (a & b)])
                mean_sign = float(np.dot(p_qibo, 1 - 2 * parity))
                fidelity = float(abs(np.vdot(psi_j, psi_born)) ** 2)
                worst["dF"] = max(worst["dF"], abs(mean_sign - fidelity))
                counts["d"] += 1

    check(worst["a"] < TOL_EXACT, f"(a) estado de born == estado del circuito de Qibo "
          f"({len(points)} puntos): max dif {worst['a']:.1e}")
    check(worst["b"] < TOL_EXACT, f"(b) {counts['b']} circuitos de energia y gradiente: "
          f"max |P_qibo - P_born| = {worst['b']:.1e}")
    if counts["c"]:
        check(worst["c"] < TOL_EXACT, f"(c) {counts['c']} circuitos de penalizacion (N+1 qubits): "
              f"max |p_qibo - p_born| = {worst['c']:.1e}")
    if counts["d"]:
        check(worst["d"] < TOL_EXACT and worst["dF"] < TOL_EXACT,
              f"(d) {counts['d']} SWAP destructivos ({2 * n} qubits): max |P_qibo - P_born| = "
              f"{worst['d']:.1e}; media exacta de (-1)^(a.b) - F = {worst['dF']:.1e}")

    # (e) candidate scans with both samplers on the same engine.
    point = max(points, key=lambda p: len(p["previous"]))
    thetas, ops = point["psi"]
    previous = [{"thetas": t, "ops": o} for t, o in point["previous"]]
    betas = [point["beta"]] * len(previous)
    prev_states = [np.asarray(builder.state(t, o)).ravel() for t, o in point["previous"]]
    candidates = list(range(n_pool))
    exact = {}
    for mu in candidates:
        g_e, g_p, *_ = eng.statevector_vqd_gradient(thetas, ops, pool[mu], prev_states, betas)
        exact[mu] = (float(g_e), float(g_p))
    e_exact = qubit_operator_expectation_from_state(ctx.hamiltonian.qubit_hamiltonian,
                                                    np.asarray(builder.state(thetas, ops)).ravel(), n)
    original = eng.sampler
    usage, z_values, t_scan = {}, {}, {}
    try:
        for sampler in SAMPLERS:
            eng.sampler = sampler
            zs, deterministic_bad = [], 0
            start = dict(eng.usage)
            t0 = time.perf_counter()
            for _ in range(scan_repeats):
                scan = eng.sampled_candidate_scan(thetas, ops, candidates, nshots=scan_shots,
                                                  previous_ansatze=previous, betas=betas,
                                                  selection_criterion="full_vqd",
                                                  penalty_nshots=scan_shots)
                for mu in candidates:
                    for value, err, ref in ((scan.g_energy[mu], scan.g_energy_stderr[mu], exact[mu][0]),
                                            (scan.g_penalty[mu], scan.g_penalty_stderr[mu], exact[mu][1])):
                        if err > 0:
                            zs.append((value - ref) / err)
                        elif abs(value - ref) > 1e-9:
                            deterministic_bad += 1
            t_scan[sampler] = (time.perf_counter() - t0) / scan_repeats
            usage[sampler] = {k: eng.usage[k] - start[k] for k in start}
            z_values[sampler] = (np.asarray(zs), deterministic_bad)
    finally:
        eng.sampler = original

    check(usage["qibo"] == usage["born"],
          f"(e) barrido con {len(previous)} estados previos: mismos circuitos y shots contados "
          f"({usage['born']} en {scan_repeats} barridos)")
    for sampler in SAMPLERS:
        zs, bad = z_values[sampler]
        z2 = float(np.mean(zs ** 2))
        se = float(np.sqrt(2.0 / len(zs)))
        check(bad == 0 and abs(z2 - 1.0) < 4 * se + 0.1,
              f"(e) {sampler}: g^E y g^pen de los {n_pool} candidatos frente al exacto: "
              f"<z^2> = {z2:.3f} +- {se:.3f}, <z> = {float(np.mean(zs)):+.3f} "
              f"(n = {len(zs)}), {t_scan[sampler] * 1e3:.0f} ms por barrido")
    print(f"   (energia exacta en el punto del barrido: {e_exact:.6f})")


# =====================================================================
# seeds
# =====================================================================

def find_run_vqd(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    for candidate in (ROOT / "scripts" / "run_vqd.py", ROOT / "run_vqd.py", HERE / "run_vqd.py"):
        if candidate.is_file():
            return candidate
    raise SystemExit("No encuentro run_vqd.py: pasalo con --run-vqd RUTA")


def run_label(sampler: str, seed: int) -> str:
    return f"{sampler}_s{seed}"


def json_path(out: Path, label: str) -> Path:
    return out / label / f"raw_results_{label}.json"


def run_seeds(out: Path, seeds: list[int], extra: list[str], *, parallel: int,
              run_vqd: Path, force: bool) -> None:
    out = Path(out).resolve()             # run_vqd runs with cwd = repository root
    for name in RESERVED_ARGS:
        if name in extra:
            raise SystemExit(f"{name} lo fija este script; quitalo de las opciones de run_vqd.")
    if "--n-jobs" in extra and extra[extra.index("--n-jobs") + 1] != "1":
        print("  AVISO: con --n-jobs > 1 la ruta muestreada no es reproducible y los "
              "procesos compiten por los nucleos.")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}

    jobs = []
    for seed in seeds:                       # interleaved: partial results stay balanced
        for sampler in SAMPLERS:
            label = run_label(sampler, seed)
            if json_path(out, label).is_file() and not force:
                print(f"   {label}: ya hecho, se salta")
                continue
            cmd = [sys.executable, str(run_vqd), *DEFAULT_RUN_ARGS, *extra,
                   "--sampler", sampler, "--seed", str(seed),
                   "--results-dir", str(out), "--run-label", label]
            jobs.append((label, sampler, seed, cmd))
    if not jobs:
        return

    env = dict(os.environ)
    if parallel > 1:                          # one core per process
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMBA_NUM_THREADS"):
            env[var] = "1"
    print(f"   {len(jobs)} ejecuciones, {parallel} a la vez. Logs en {out}/<etiqueta>.log")

    def launch(job):
        label, sampler, seed, cmd = job
        t0 = time.perf_counter()
        with open(out / f"{label}.log", "w", encoding="utf-8") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                  cwd=str(ROOT)).returncode
        wall = time.perf_counter() - t0
        print(f"   {label}: {'terminado' if code == 0 else f'ERROR (codigo {code})'} "
              f"en {wall / 60:.1f} min", flush=True)
        return label, {"sampler": sampler, "seed": seed, "cmd": cmd, "returncode": code,
                       "wall_s": wall, "parallel": parallel}

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as executor:
        for label, info in executor.map(launch, jobs):
            manifest[label] = info
            manifest_path.write_text(json.dumps(manifest, indent=2))


# =====================================================================
# compare
# =====================================================================

IGNORED_KEYS = {"sampler", "seed"}


def load_runs(out: Path) -> list[dict]:
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    runs = []
    for path in sorted(out.glob("*/raw_results_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        label = path.parent.name
        runs.append({"label": label, "path": path, "data": data,
                     "sampler": data["numerical"].get("sampler", "qibo"),
                     "seed": int(data["run"]["seed"]),
                     "wall_s": manifest.get(label, {}).get("wall_s")})
    return runs


def config_differences(runs: list[dict]) -> list[str]:
    """Settings that differ between runs, apart from sampler and seed."""
    diffs = []
    ref = runs[0]["data"]
    for section in ("problem", "run", "numerical"):
        keys = set(ref.get(section, {}))
        for r in runs[1:]:
            keys |= set(r["data"].get(section, {}))
        for key in sorted(keys - IGNORED_KEYS):
            values = {json.dumps(r["data"].get(section, {}).get(key), sort_keys=True) for r in runs}
            if len(values) > 1:
                diffs.append(f"{section}.{key}: {sorted(values)}")
    return diffs


def pointwise_z(ctx, data: dict) -> dict[str, list[float]]:
    """z of energy, objective and selected gradient at every iteration of a run."""
    from vqd_nuclear.measurements import qubit_operator_expectation_from_state

    eng, builder = ctx.measurements, ctx.ansatz_builder
    n = int(ctx.hamiltonian.n_qubits)
    pool = ctx.hamiltonian.qubit_generator_pool
    plan = eng.energy_plan
    results = data["full_quantum_circuit_results"]
    finals = [np.asarray(builder.state(r["thetas"], r["ops"])).ravel() for r in results]
    out = {"energy": [], "objective": [], "gradient": []}
    for k, result in enumerate(results):
        e_shots = int(result["energy_shots"])
        o_shots = int(result["overlap_shots"])
        beta = float(result.get("beta", 0.0))
        previous = finals[:k]
        betas = [beta] * k
        selection_point = ([], [])
        for record in result["history"]:
            # Selection point: ansatz BEFORE adding this iteration's operator.
            t_sel, o_sel = selection_point
            psi_sel = np.asarray(builder.state(t_sel, o_sel)).ravel()
            g_e, g_p, *_ = eng.statevector_vqd_gradient(
                t_sel, o_sel, pool[int(record["selected_operator"])], previous, betas, psi=psi_sel)
            score_exact = g_e + g_p if result.get("selection_criterion") == "full_vqd" else g_e
            err = record.get("adapt_gradient_stderr")
            if err is not None and err > 0 and record.get("adapt_gradient") is not None:
                out["gradient"].append((float(record["adapt_gradient"]) - score_exact) / float(err))

            # Reported energy and re-evaluated objective at this iteration's theta.
            psi = np.asarray(builder.state(record["thetas"], record["ops"])).ravel()
            e_exact, variances = plan.exact_statistics(psi, n)
            var_e = sum(variances) / e_shots
            if var_e > 0:
                out["energy"].append((float(record["physical_energy"]) - e_exact) / np.sqrt(var_e))
            if k > 0 and record.get("vqd_objective_reevaluated") is not None:
                fids = [float(abs(np.vdot(prev, psi)) ** 2) for prev in previous]
                obj_exact = e_exact + beta * sum(fids)
                var_o = var_e + sum(beta ** 2 * (1.0 - f ** 2) / o_shots for f in fids)
                out["objective"].append((float(record["vqd_objective_reevaluated"]) - obj_exact)
                                        / np.sqrt(var_o))
            selection_point = (list(record["thetas"]), list(record["ops"]))
    return out


def final_rows(basis, ctx, run: dict) -> list[dict]:
    rows = []
    for k, result in enumerate(run["data"]["full_quantum_circuit_results"]):
        psi = basis.state(result["thetas"], result["ops"])
        weights = basis.weights(psi)
        rows.append({
            "label": run["label"], "sampler": run["sampler"], "seed": run["seed"], "state": k,
            "fidelity": float(sum(weights[j] for j in basis.target_level(k))),
            "exact_energy_error": abs(basis.energy(psi) - float(basis.eigenvalues[k])),
            "reported_energy_error": abs(float(result["energy"]) - float(basis.eigenvalues[k])),
            "n_layers": int(result["n_layers"]),
            "termination_reason": result.get("termination_reason"),
            "circuits_executed": int(result.get("circuits_executed", 0)),
            "shots_used": int(result.get("shots_used", 0)),
            "wall_s": run["wall_s"],
        })
    return rows


def mann_whitney(a, b) -> float | None:
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return None
    if len(a) < 2 or len(b) < 2 or len(set(a) | set(b)) < 2:
        return None                       # all values equal: nothing to test
    p_value = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    return None if not np.isfinite(p_value) else p_value


def describe(values, digits: int = 5) -> str:
    """Mean +- standard error of the mean."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return "-"
    se = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
    return f"{v.mean():.{digits}f} +- {se:.{digits}f}"


def run_compare(out: Path, *, ctx=None, csv: bool = True) -> None:
    runs = [r for r in load_runs(out) if r["data"].get("full_quantum_circuit_results")]
    by_sampler = {s: [r for r in runs if r["sampler"] == s] for s in SAMPLERS}
    print(f"   Ejecuciones: " + ", ".join(f"{s} {len(by_sampler[s])} (semillas "
                                          f"{sorted(r['seed'] for r in by_sampler[s])})"
                                          for s in SAMPLERS))
    if not all(by_sampler.values()):
        raise SystemExit("Hace falta al menos una ejecucion de cada muestreador.")
    diffs = config_differences(runs)
    check(not diffs, "mismos parametros en todas las ejecuciones salvo sampler y seed"
          + ("" if not diffs else ":\n         " + "\n         ".join(diffs)))

    data0 = runs[0]["data"]
    if ctx is None:
        ctx = make_context(data0["problem"], data0["run"], data0["numerical"])
    from fidelity_per_iteration import ExactBasis
    p = data0["problem"]
    basis = ExactBasis(ctx, int(p["n_valence_protons"]) + int(p["n_valence_neutrons"]))

    print("\n   1. Punto a punto (z = (valor medido - exacto) / sigma, en la misma theta)")
    z = {s: {"energy": [], "objective": [], "gradient": []} for s in SAMPLERS}
    for r in runs:
        for key, values in pointwise_z(ctx, r["data"]).items():
            z[r["sampler"]][key].extend(values)
    names = {"energy": "energia", "objective": "objetivo reevaluado", "gradient": "gradiente elegido"}
    for key in ("energy", "objective", "gradient"):
        parts = []
        for s in SAMPLERS:
            v = np.asarray(z[s][key])
            if v.size:
                parts.append(f"{s}: <z> = {v.mean():+.3f}, <z^2> = {np.mean(v ** 2):.3f} "
                             f"+- {np.sqrt(2.0 / v.size):.3f} (n = {v.size})")
        p_value = mann_whitney(z["qibo"][key], z["born"][key])
        print(f"   {names[key]:>20}: " + "; ".join(parts)
              + ("" if p_value is None else f"; Mann-Whitney p = {p_value:.2f}"))
        both = [np.asarray(z[s][key]) for s in SAMPLERS if len(z[s][key])]
        if len(both) == 2:
            if key == "gradient":
                ok = p_value is None or p_value > 0.01
                check(ok, f"{names[key]}: misma distribucion con los dos muestreadores")
            else:
                ok = all(abs(np.mean(v ** 2) - 1.0) < 4 * np.sqrt(2.0 / v.size) + 0.05 for v in both)
                ok = ok and (p_value is None or p_value > 0.01)
                check(ok, f"{names[key]}: <z^2> ~ 1 con los dos y misma distribucion")

    print("\n   2. Resultado final por estado (depende de la trayectoria)")
    rows = [row for r in runs for row in final_rows(basis, ctx, r)]
    n_states = max(row["state"] for row in rows) + 1
    print(f"   {'estado':>6} {'magnitud':>22} {'qibo':>22} {'born':>22} {'p (MW)':>7}")
    for k in range(n_states):
        for key, title, digits in (("fidelity", "fidelidad", 5),
                                   ("exact_energy_error", "|E(theta) - lambda|", 5),
                                   ("n_layers", "capas", 1), ("circuits_executed", "circuitos", 0)):
            values = {s: [row[key] for row in rows if row["state"] == k and row["sampler"] == s]
                      for s in SAMPLERS}
            p_value = mann_whitney(values["qibo"], values["born"])
            print(f"   {k:>6} {title:>22} {describe(values['qibo'], digits):>22} {describe(values['born'], digits):>22} "
                  f"{'-' if p_value is None else f'{p_value:.2f}':>7}")
        for s in SAMPLERS:
            reasons = [row["termination_reason"] for row in rows if row["state"] == k and row["sampler"] == s]
            summary = ", ".join(f"{name} x{reasons.count(name)}" for name in sorted(set(map(str, reasons))))
            print(f"   {'':>6} {'parada (' + s + ')':>22} {summary}")
    for s in SAMPLERS:
        walls = [r["wall_s"] for r in by_sampler[s] if r["wall_s"] is not None]
        if walls:
            print(f"   tiempo por ejecucion ({s}): {np.mean(walls) / 60:.1f} min "
                  f"(min {min(walls) / 60:.1f}, max {max(walls) / 60:.1f})")
    print("   Con 5 + 5 ejecuciones el nivel 2 solo detecta diferencias grandes; p < 0.05 en\n"
          "   una de muchas filas es esperable por azar. El nivel 1 es el decisivo.")

    if csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out / "compare_final_states.csv", index=False)
        flat = [{"sampler": s, "quantity": key, "z": v}
                for s in SAMPLERS for key in z[s] for v in z[s][key]]
        pd.DataFrame(flat).to_csv(out / "compare_pointwise_z.csv", index=False)
        print(f"   CSV: {out / 'compare_final_states.csv'}, {out / 'compare_pointwise_z.csv'}")


# =====================================================================
# command line
# =====================================================================

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_exact = sub.add_parser("exact", help="comprobacion exacta sin shots")
    p_seeds = sub.add_parser("seeds", help="lanza las ejecuciones qibo y born")
    p_compare = sub.add_parser("compare", help="compara las ejecuciones")
    p_all = sub.add_parser("all", help="exact + seeds + compare")
    for p in (p_exact, p_all):
        p.add_argument("--json", default=None, help="JSON cuyos estados finales se comprueban")
        p.add_argument("--random-points", type=int, default=3)
        p.add_argument("--scan-shots", type=int, default=20000)
        p.add_argument("--scan-repeats", type=int, default=10)
    for p in (p_seeds, p_all):
        p.add_argument("--out", type=Path, default=Path("runs_d2"))
        p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
        p.add_argument("--parallel", type=int, default=1,
                       help="ejecuciones simultaneas (como mucho, nucleos fisicos)")
        p.add_argument("--run-vqd", default=None, help="ruta de run_vqd.py si no la encuentra")
        p.add_argument("--force", action="store_true", help="repite las ya terminadas")
    p_compare.add_argument("out", type=Path)
    args = parser.parse_args(argv)

    if args.command in ("exact", "all"):
        print("1. Comprobacion exacta (sin shots)")
        np.random.seed(2026)
        rng = np.random.default_rng(2026)
        data = json.loads(Path(args.json).read_text(encoding="utf-8")) if args.json else None
        problem = data["problem"] if data else BE6
        ctx = make_context(problem, data["run"] if data else {"n_states": 3},
                           data["numerical"] if data else {})
        run_exact(ctx, exact_points(ctx, data, args.random_points, rng),
                  scan_shots=args.scan_shots, scan_repeats=args.scan_repeats)
    if args.command in ("seeds", "all"):
        print("2. Ejecuciones con semillas")
        run_seeds(args.out, args.seeds, extra, parallel=args.parallel,
                  run_vqd=find_run_vqd(args.run_vqd), force=args.force)
    if args.command in ("compare", "all"):
        print("3. Comparacion qibo frente a born")
        run_compare(args.out)

    print("\nTODO OK" if failures == 0 else f"\n{failures} FALLOS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
