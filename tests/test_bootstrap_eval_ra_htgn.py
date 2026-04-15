from scripts.run_bootstrap_eval_ra_htgn import (
    RA_COMPARISON_PAIRS,
    RA_VARIANTS,
    run_bootstrap_eval_ra_htgn,
)


def test_ra_htgn_runner_declares_expected_variants_and_pairs():
    assert RA_VARIANTS == ["ra_htgn", "tgn", "roland", "gat_static"]
    assert RA_COMPARISON_PAIRS == [
        ("ra_htgn", "tgn"),
        ("ra_htgn", "roland"),
        ("ra_htgn", "gat_static"),
    ]


def test_ra_htgn_runner_entrypoint_exists():
    assert callable(run_bootstrap_eval_ra_htgn)
