#!/usr/bin/env python3
"""Measure context construction and direct projections without dense statevectors."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter
import tracemalloc
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqd_nuclear import NumericalConfig, ProblemConfig, RunConfig, vqd
from vqd_nuclear.model_space import ModelSpaceEngine


def process_peak_bytes() -> int:
    """Peak RSS for this fresh profiling process, including native allocations."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                    "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile")]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not query(handle, ctypes.byref(counters), counters.cb):
            raise OSError("Cannot read peak process memory.")
        return int(counters.peak_working_set)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell", choices=["p", "sd"], default="sd")
    parser.add_argument("--protons", type=int, default=1)
    parser.add_argument("--neutrons", type=int, default=1)
    parser.add_argument("--J", type=int, default=0)
    parser.add_argument("--generators", type=int, default=3)
    parser.add_argument("--all-generators", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.generators < 0:
        parser.error("--generators must be nonnegative")
    timings = {}

    class TimedEngine(ModelSpaceEngine):
        def __init__(self, **kwargs):
            start = perf_counter()
            super().__init__(**kwargs)
            timings["model_space_seconds"] = perf_counter() - start

    def forbid_full_state(*args, **kwargs):
        raise AssertionError("Context/projection called the full-state operator routine.")

    tracemalloc.start()
    start = perf_counter()
    # Fail if either the Hamiltonian or a generator uses the original dense path.
    with patch.object(vqd, "ModelSpaceEngine", TimedEngine), patch.object(
        vqd, "apply_qubit_operator_to_state", forbid_full_state,
    ):
        context = vqd.build_context(
            ProblemConfig(shell=args.shell, n_valence_protons=args.protons,
                          n_valence_neutrons=args.neutrons, total_j=args.J),
            RunConfig(n_states=3, n_workers=1), NumericalConfig(n_jobs=1),
        )
    timings["build_context_seconds"] = perf_counter() - start
    model = context.measurements.model_space
    count = len(context.base_operator_indices) if args.all_generators else min(args.generators, len(context.base_operator_indices))
    projections = []
    for index in context.base_operator_indices[:count]:
        start = perf_counter()
        generator = model.generator(index)
        projections.append({"operator_id": index, "seconds": perf_counter() - start,
                            "preserves_space": generator.preserves_space})
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = {
        "shell": args.shell, "protons": args.protons, "neutrons": args.neutrons, "M": args.J,
        "n_qubits": model.n_qubits, "physical_dimension": len(model.indices), "available": model.available,
        "pool_size": len(context.base_operator_indices), "projected_generators": len(model._generators),
        **timings, "generator_projections": projections, "python_traced_peak_bytes": peak,
        "process_peak_bytes": process_peak_bytes(), "full_state_callback_calls": 0,
        "generator_projection_total_seconds": sum(row["seconds"] for row in projections),
        "memory_tracing_enabled": True,
        "environment": {"python": sys.version, "platform": sys.platform,
                        "pynucshell": version("pynucshell"), "qibo": version("qibo")},
    }
    output = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "generator_projections"}, indent=2))
        print(f"Per-generator timings saved to {args.output}")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
