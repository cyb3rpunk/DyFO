from scripts.run_bootstrap_eval_temporal_kg import (
    TKG_COMPARISON_PAIRS,
    TKG_VARIANTS,
    run_bootstrap_eval_temporal_kg,
)


def test_temporal_kg_runner_declares_expected_variants_and_pairs():
    assert TKG_VARIANTS == ["temporal_kg", "ra_htgn", "tgn", "roland", "gat_static"]
    assert TKG_COMPARISON_PAIRS == [
        ("temporal_kg", "ra_htgn"),
        ("temporal_kg", "tgn"),
        ("temporal_kg", "roland"),
        ("temporal_kg", "gat_static"),
        ("ra_htgn", "tgn"),
        ("ra_htgn", "roland"),
        ("ra_htgn", "gat_static"),
        ("tgn", "roland"),
        ("tgn", "gat_static"),
        ("roland", "gat_static"),
    ]


def test_temporal_kg_runner_entrypoint_exists():
    assert callable(run_bootstrap_eval_temporal_kg)
