import pytest
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import PredictPipeline




def test_predict_returns_valid_segment():
    """Check that prediction returns a valid cluster and segment name."""


    pipeline = PredictPipeline()
    result = pipeline.predict(recency=10, frequency=15, monetary=8000)

    assert "cluster" in result
    assert "segment" in result
    assert isinstance(result["cluster"], int)
    assert result["segment"] in ["Champions", "Loyal", "At Risk", "New/Promising"]



def test_predict_champions_segment():
    """A recent, frequent, high-spending customer should be classified as Champion."""

    pipeline = PredictPipeline()
    result = pipeline.predict(recency=5, frequency=20, monetary=10000)

    assert result["segment"] == "Champions"


def test_predict_at_risk_segment():
    """An inactive , infrequent, low-spending customer should be classified as At Risk."""
    pipeline = PredictPipeline()
    result = pipeline.predict(recency=300, frequency=1, monetary=100)


    assert result["segment"] == "At Risk"