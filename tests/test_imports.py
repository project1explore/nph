from nph import __version__, repo_root, reports_dir, results_dir
from nph.cli import build_parser


def test_import_nph():
    assert __version__


def test_path_helpers_exist():
    assert callable(repo_root)
    assert callable(results_dir)
    assert callable(reports_dir)
    assert results_dir().name == "results"
    assert reports_dir().name == "reports"


def test_cli_parser_builds():
    parser = build_parser()
    sim_args = parser.parse_args(["simulate"])
    rep_args = parser.parse_args(["make-report"])
    assert sim_args.command == "simulate"
    assert rep_args.command == "make-report"
