import pytest

from dyfo.config import DyFOConfig
from dyfo.core.model_variants import TGNWrapper, build_encoder
from dyfo.core.relation_aware_tgn import RAHTGNEncoder
from dyfo.core.temporal_kg import TemporalKGEncoder


def test_config_accepts_ra_htgn_variant():
    config = DyFOConfig(model_variant="ra_htgn")
    assert config.model_variant == "ra_htgn"


def test_config_accepts_temporal_kg_variant():
    config = DyFOConfig(model_variant="temporal_kg")
    assert config.model_variant == "temporal_kg"


def test_config_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Invalid model_variant"):
        DyFOConfig(model_variant="unknown_variant")


def test_build_encoder_keeps_tgn_and_adds_ra_htgn():
    tgn_encoder = build_encoder(DyFOConfig(model_variant="tgn"), num_nodes=4)
    ra_encoder = build_encoder(DyFOConfig(model_variant="ra_htgn"), num_nodes=4)
    tkg_encoder = build_encoder(DyFOConfig(model_variant="temporal_kg"), num_nodes=4)

    assert isinstance(tgn_encoder, TGNWrapper)
    assert isinstance(ra_encoder, RAHTGNEncoder)
    assert isinstance(tkg_encoder, TemporalKGEncoder)
