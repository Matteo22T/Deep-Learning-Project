from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.model import (
    build_model,
    check_input_compatibility,
    check_target_compatibility,
    get_model_params,
)
from src.utils.config import ConfigLoader

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment_config.yaml"
N_FEATURES = 768


@pytest.fixture(scope="module")
def config():
    return ConfigLoader(CONFIG_PATH).get_config()


def _fake_data(n_samples: int = 20, n_features: int = N_FEATURES, seed: int = 0):
    """Random features and 0/1 labels with the same shapes as the real data."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_samples, n_features)).astype("float32")
    y_os = rng.integers(0, 2, size=n_samples)
    labels = {"OS_1y": y_os, "PFS_1y": y_os.copy()}
    return features, labels


def test_model_outputs_one_probability_per_target(config):
    model = build_model(config, input_shape=(N_FEATURES,), seed=42)
    features, _ = _fake_data(n_samples=5)

    predictions = model.predict(features, verbose=0)

    assert set(predictions) == {"OS_1y", "PFS_1y"}
    for probs in predictions.values():
        assert probs.shape == (5, 1)
        assert np.all((probs > 0) & (probs < 1))


def test_parameter_count_matches_config(config):
    model = build_model(config, input_shape=(N_FEATURES,), seed=42)
    # hidden_1: 768*64 + 64, hidden_2: 64*32 + 32, two heads: 2 * (32 + 1)
    assert model.count_params() == 51_362


def test_empty_hidden_units_gives_linear_baseline(config):
    model = build_model(
        config, input_shape=(N_FEATURES,), seed=42, overrides={"hidden_units": []}
    )
    assert model.name == "linear_baseline"
    # one weight per feature plus a bias, for each of the two heads
    assert model.count_params() == 2 * (N_FEATURES + 1)


def test_same_seed_gives_same_initial_weights(config):
    weights_a = build_model(config, (N_FEATURES,), seed=42).get_weights()
    weights_b = build_model(config, (N_FEATURES,), seed=42).get_weights()
    weights_c = build_model(config, (N_FEATURES,), seed=7).get_weights()

    assert all(np.array_equal(a, b) for a, b in zip(weights_a, weights_b))
    assert not all(np.array_equal(a, c) for a, c in zip(weights_a, weights_c))


def test_overrides_do_not_modify_config(config):
    before = get_model_params(config)
    build_model(config, (N_FEATURES,), seed=42, overrides={"dropout": 0.5})
    assert get_model_params(config) == before


@pytest.mark.parametrize(
    "overrides",
    [
        {"architecture": "cnn"},
        {"hidden_units": 64},
        {"hidden_units": [64, 0]},
        {"activation": "not_an_activation"},
        {"dropout": 1.0},
        {"l2": -1.0},
        {"optimizer": "sgd"},
        {"learning_rate": 0},
        {"loss_weights": {"OS_1y": 1.0}},
        {"unknown_param": 1},
    ],
)
def test_invalid_params_raise(config, overrides):
    with pytest.raises(ValueError):
        build_model(config, (N_FEATURES,), overrides=overrides)


def test_non_flat_input_shape_raises(config):
    with pytest.raises(ValueError):
        build_model(config, input_shape=(4, N_FEATURES))


def test_check_input_compatibility(config):
    model = build_model(config, (N_FEATURES,), seed=42)
    features, _ = _fake_data()

    check_input_compatibility(model, features)

    # data aggregated with 'concat' given to a model built for 'mean'
    with pytest.raises(ValueError):
        check_input_compatibility(model, np.zeros((5, 4 * N_FEATURES)))

    with_nan = features.copy()
    with_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        check_input_compatibility(model, with_nan)


def test_check_target_compatibility(config):
    model = build_model(config, (N_FEATURES,), seed=42)
    features, labels = _fake_data()

    check_target_compatibility(model, labels, n_samples=len(features))

    with pytest.raises(ValueError):
        check_target_compatibility(model, {"OS_1y": labels["OS_1y"]})
    with pytest.raises(ValueError):
        check_target_compatibility(
            model, {**labels, "PFS_1y": np.full(len(features), 2)}
        )
    with pytest.raises(ValueError):
        check_target_compatibility(model, labels, n_samples=len(features) + 1)


def test_model_is_ready_to_train(config):
    model = build_model(config, (N_FEATURES,), seed=42)
    features, labels = _fake_data()

    history = model.fit(features, labels, epochs=2, batch_size=8, verbose=0)

    assert len(history.history["loss"]) == 2
    assert np.all(np.isfinite(history.history["loss"]))
