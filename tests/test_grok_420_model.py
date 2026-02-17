"""Test for Grok 4.20-beta model support"""

import pytest
from app.services.grok.model import ModelService, Tier, Cost


def test_grok_420_beta_model_exists():
    """Test that grok-4.20-beta model is defined"""
    model = ModelService.get("grok-4.20-beta")
    assert model is not None, "grok-4.20-beta model should exist"
    

def test_grok_420_beta_model_properties():
    """Test that grok-4.20-beta model has correct properties"""
    model = ModelService.get("grok-4.20-beta")
    
    assert model.model_id == "grok-4.20-beta"
    assert model.grok_model == "grok-420"
    assert model.rate_limit_model == "grok-420"
    assert model.model_mode == "MODEL_MODE_GROK_420"
    assert model.cost == Cost.LOW
    assert model.tier == Tier.BASIC
    assert model.display_name == "Grok 4.20 Beta"


def test_grok_420_beta_to_grok_conversion():
    """Test conversion to Grok parameters"""
    grok_model, model_mode = ModelService.to_grok("grok-4.20-beta")
    
    assert grok_model == "grok-420"
    assert model_mode == "MODEL_MODE_GROK_420"


def test_grok_420_beta_pool_selection():
    """Test pool selection for grok-4.20-beta"""
    pool = ModelService.pool_for_model("grok-4.20-beta")
    assert pool == "ssoBasic"
    
    pool_candidates = ModelService.pool_candidates_for_model("grok-4.20-beta")
    assert pool_candidates == ["ssoBasic", "ssoSuper"]


def test_grok_420_beta_rate_limit_model():
    """Test rate limit model mapping"""
    rate_limit_model = ModelService.rate_limit_model_for("grok-4.20-beta")
    assert rate_limit_model == "grok-420"


def test_grok_420_beta_validation():
    """Test model validation"""
    assert ModelService.valid("grok-4.20-beta") is True


def test_grok_420_beta_not_heavy_model():
    """Test that grok-4.20-beta is not a heavy bucket model"""
    assert ModelService.is_heavy_bucket_model("grok-4.20-beta") is False


def test_grok_420_beta_in_model_list():
    """Test that grok-4.20-beta is in the model list"""
    models = ModelService.list()
    model_ids = [m.model_id for m in models]
    assert "grok-4.20-beta" in model_ids
