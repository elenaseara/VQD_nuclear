from pathlib import Path
import importlib.util


def load_run_vqd_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_vqd.py"
    spec = importlib.util.spec_from_file_location("run_vqd_cli", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_short_aliases_are_accepted():
    cli = load_run_vqd_module()
    args = cli.build_parser().parse_args([
        "--shell", "p",
        "--protons", "2",
        "--neutrons", "0",
        "--J", "0",
        "--n-states", "2",
        "--mode", "statevector_exact",
    ])
    assert args.shell == "p"
    assert args.n_valence_protons == 2
    assert args.n_valence_neutrons == 0
    assert args.total_j == 0
    assert cli.resolve_run_routes(args) == (True, False)


def test_cli_mode_full_quantum_circuit():
    cli = load_run_vqd_module()
    args = cli.build_parser().parse_args(["--mode", "full_quantum_circuit"])
    assert cli.resolve_run_routes(args) == (False, True)


def test_cli_mode_both():
    cli = load_run_vqd_module()
    args = cli.build_parser().parse_args(["--mode", "both"])
    assert cli.resolve_run_routes(args) == (True, True)


def test_cli_legacy_route_flags_still_work():
    cli = load_run_vqd_module()
    args = cli.build_parser().parse_args([
        "--statevector", "true",
        "--full-quantum-circuit", "false",
    ])
    assert cli.resolve_run_routes(args) == (True, False)


def test_cli_measurement_strategy_defaults_to_paper():
    cli = load_run_vqd_module()
    args = cli.build_parser().parse_args([])
    assert args.measurement_strategy == "paper"


def test_cli_measurement_strategy_qwc_is_accepted():
    cli = load_run_vqd_module()
    args = cli.build_parser().parse_args(["--measurement-strategy", "qwc"])
    assert args.measurement_strategy == "qwc"
