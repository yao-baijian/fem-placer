"""
Test suite for ml module - ML-based parameter prediction.

Tests model training, prediction, and dataset handling for alpha parameter optimization.
"""

import os
import sys

# Point to project root: tests/unit_test/ → tests/ → project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tempfile
import pytest
import numpy as np
import pandas as pd

from ml.model import create_default_model, save_model, load_model, get_model_path
from ml.dataset import FIELDNAMES, extract_features_from_placer, get_feature_fieldnames
from ml.train import train_from_csv
from ml.predict import predict_alpha, predict_target


class TestMLAlphaModel:
    """Test ml model creation, saving, and loading."""

    def test_create_default_model(self):
        """Test that default model is created with correct parameters."""
        model = create_default_model()
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert model.n_estimators == 200
        assert model.max_depth == 8

    def test_create_model_with_custom_params(self):
        """Test model creation with custom parameters."""
        model = create_default_model(n_estimators=100, max_depth=5)
        assert model.n_estimators == 100
        assert model.max_depth == 5

    def test_save_and_load_model(self):
        """Test model persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')

            # Create and save model
            model = create_default_model()
            save_model(model, model_path)
            assert os.path.exists(model_path)

            # Load model
            loaded_model = load_model(model_path)
            assert loaded_model is not None
            assert loaded_model.n_estimators == model.n_estimators
            assert loaded_model.max_depth == model.max_depth

    def test_load_nonexistent_model(self):
        """Test loading non-existent model returns None."""
        model = load_model('/nonexistent/path/model.pkl')
        assert model is None


class TestMLAlphaDataset:
    """Test ml dataset utilities."""

    def test_fieldnames_completeness(self):
        """Test that FIELDNAMES contains all expected fields."""
        expected_fields = [
            "opti_insts_num", "avail_sites_num", "fixed_insts_num",
            "utilization", "logic_area_length", "logic_area_width", "io_height",
            "net_count", "hpwl_before", "hpwl_after", "overlap_after", "alpha", "beta"
        ]
        assert FIELDNAMES == expected_fields

    def test_pre_alpha_features(self):
        """Test feature fieldnames exclude target variables and identifiers."""
        feats = get_feature_fieldnames(with_io=False)
        assert "alpha" not in feats
        assert "beta" not in feats
        assert "hpwl_after" not in feats
        assert "overlap_after" not in feats
        assert "instance" not in feats
        assert "opti_insts_num" in feats


class TestMLAlphaTraining:
    """Test ml training functionality."""

    def test_train_from_csv_with_valid_data(self):
        """Test training with valid CSV data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test_data.csv')
            model_path = os.path.join(tmpdir, 'test_model.pkl')

            # Create synthetic training data
            num_samples = 50
            data = {
                "instance": [f"test_{i}" for i in range(num_samples)],
                "opti_insts_num": np.random.randint(50, 200, num_samples),
                "avail_sites_num": np.random.randint(100, 300, num_samples),
                "fixed_insts_num": np.random.randint(10, 50, num_samples),
                "utilization": np.random.uniform(0.3, 0.9, num_samples),
                "logic_area_length": np.random.randint(10, 30, num_samples),
                "logic_area_width": np.random.randint(10, 30, num_samples),
                "io_height": np.random.randint(5, 15, num_samples),
                "net_count": np.random.randint(100, 500, num_samples),
                "hpwl_before": np.random.uniform(1000, 5000, num_samples),
                "hpwl_after": np.random.uniform(800, 4000, num_samples),
                "overlap_after": np.random.randint(0, 10, num_samples),
                "alpha": np.random.uniform(0.5, 2.0, num_samples)
            }

            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)

            result = train_from_csv(csv_path, target="alpha", test_size=0.2)

            assert "mse" in result
            assert "model_path" in result
            assert result["mse"] >= 0
            assert os.path.exists(result["model_path"])

    def test_train_from_csv_missing_file(self):
        """Test training with non-existent CSV raises error."""
        with pytest.raises(FileNotFoundError):
            train_from_csv("/nonexistent/path.csv")

    def test_train_from_csv_missing_target(self):
        """Test training with missing target column raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test_data.csv')

            # Create CSV without target column
            data = {
                "instance": ["test_1", "test_2"],
                "opti_insts_num": [100, 120]
            }
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)

            with pytest.raises(ValueError, match="target column not present"):
                train_from_csv(csv_path, target="alpha")


class TestMLAlphaPrediction:
    """Test ml prediction functionality."""

    def test_predict_alpha_without_model(self):
        """Test prediction without trained model raises error."""
        from unittest.mock import patch
        
        feature_row = {
            "opti_insts_num": 100,
            "avail_sites_num": 200,
            "fixed_insts_num": 20,
            "utilization": 0.5,
            "logic_area_length": 20,
            "logic_area_width": 20,
            "io_height": 10,
            "net_count": 150
        }

        # Mock load_model to return None (model not found)
        with patch('ml.model.load_model', return_value=None):
            with pytest.raises(RuntimeError, match="No trained model found"):
                predict_alpha(feature_row)

    def test_predict_alpha_with_trained_model(self):
        """Test prediction with a trained model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'alpha_model.pkl')

            # Create synthetic training data with proper DataFrame format
            num_samples = 30
            data = {col: np.random.randn(num_samples) for col in get_feature_fieldnames(with_io=False)}
            X = pd.DataFrame(data)
            y = np.random.uniform(0.5, 2.0, num_samples)

            model = create_default_model()
            model.fit(X, y)
            save_model(model, model_path)

            feature_row = {
                "opti_insts_num": 100,
                "avail_sites_num": 200,
                "fixed_insts_num": 20,
                "utilization": 0.5,
                "logic_area_length": 20,
                "logic_area_width": 20,
                "io_height": 10,
                "net_count": 150
            }

            alpha = predict_target(feature_row, target="alpha")
            assert isinstance(alpha, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
