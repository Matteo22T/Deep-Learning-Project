from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from src.models.model import build_model
from src.training.trainer import (
    median_best_epoch,
    train_fixed_epochs,
    train_with_early_stopping,
)
from src.utils.config import ConfigLoader

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment_config.yaml"
N_FEATURES = 768


@pytest.fixture(scope="module")
def config():
    return ConfigLoader(CONFIG_PATH).get_config()


def _small_config(config, max_epochs: int = 30, patience: int = 5):
    """Same config with fewer epochs, to keep the tests fast."""
    small = copy.deepcopy(config)
    small.training.max_epochs = max_epochs
    small.training.early_stopping.patience = patience
    return small


def _fake_split(n_train: int = 40, n_val: int = 10, seed: int = 0):
    """Random train/validation data with the same shapes as the real data."""
    rng = np.random.default_rng(seed)
    x_train = rng.normal(size=(n_train, N_FEATURES)).astype("float32")
    x_val = rng.normal(size=(n_val, N_FEATURES)).astype("float32")
    y_train = rng.integers(0, 2, size=n_train)
    y_val = rng.integers(0, 2, size=n_val)
    return (
        x_train,
        {"OS_1y": y_train, "PFS_1y": y_train.copy()},
        x_val,
        {"OS_1y": y_val, "PFS_1y": y_val.copy()},
    )


def test_early_stopping_returns_best_epoch(config):
    cfg = _small_config(config)
    x_train, y_train, x_val, y_val = _fake_split()
    model = build_model(cfg, (N_FEATURES,), seed=42)

    result = train_with_early_stopping(model, x_train, y_train, x_val, y_val, cfg, seed=42)

    val_loss = result.history["val_loss"]
    assert result.epochs_trained == len(val_loss) <= cfg.training.max_epochs
    assert 1 <= result.best_epoch <= result.epochs_trained
    assert val_loss[result.best_epoch - 1] == min(val_loss)


@pytest.mark.parametrize("patience", [3, 100])
def test_returned_model_has_best_weights(config, patience):
    # patience=3 stops early, patience=100 runs until max_epochs:
    # in both cases the returned model must have the weights of the best epoch.
    cfg = _small_config(config, max_epochs=25, patience=patience)
    x_train, y_train, x_val, y_val = _fake_split()
    model = build_model(cfg, (N_FEATURES,), seed=3)

    result = train_with_early_stopping(model, x_train, y_train, x_val, y_val, cfg, seed=3)

    y_val_columns = {k: np.asarray(v, dtype="float32").reshape(-1, 1) for k, v in y_val.items()}
    loss_now = result.model.evaluate(x_val, y_val_columns, verbose=0, return_dict=True)["loss"]
    assert loss_now == pytest.approx(min(result.history["val_loss"]), rel=1e-5)


def test_same_seed_gives_same_training(config):
    cfg = _small_config(config, max_epochs=5)
    data = _fake_split()

    histories = []
    for _ in range(2):
        model = build_model(cfg, (N_FEATURES,), seed=7)
        histories.append(train_with_early_stopping(model, *data, cfg, seed=7).history)

    assert histories[0] == histories[1]


def test_fixed_epochs_runs_exactly_n_epochs(config):
    x_train, y_train, _, _ = _fake_split()
    model = build_model(config, (N_FEATURES,), seed=42)

    result = train_fixed_epochs(model, x_train, y_train, n_epochs=7, config=config, seed=42)

    assert result.epochs_trained == 7
    assert result.best_epoch is None
    assert "val_loss" not in result.history

    with pytest.raises(ValueError):
        train_fixed_epochs(
            build_model(config, (N_FEATURES,)), x_train, y_train, n_epochs=0, config=config
        )


def test_already_trained_model_is_rejected(config):
    x_train, y_train, x_val, y_val = _fake_split()
    model = build_model(config, (N_FEATURES,), seed=42)
    train_fixed_epochs(model, x_train, y_train, n_epochs=1, config=config)

    with pytest.raises(ValueError, match="already been trained"):
        train_with_early_stopping(model, x_train, y_train, x_val, y_val, config)


def test_wrong_data_is_rejected(config):
    x_train, y_train, x_val, y_val = _fake_split()
    model = build_model(config, (N_FEATURES,), seed=42)

    # a target is missing
    with pytest.raises(ValueError):
        train_with_early_stopping(
            model, x_train, {"OS_1y": y_train["OS_1y"]}, x_val, y_val, config
        )
    # validation labels do not match the validation features
    with pytest.raises(ValueError):
        train_with_early_stopping(
            model, x_train, y_train, x_val, {k: v[:-1] for k, v in y_val.items()}, config
        )


@pytest.mark.parametrize(
    "key, value",
    [
        ("batch_size", 0),
        ("max_epochs", -1),
        ("early_stopping.monitor", "loss"),
        ("early_stopping.monitor", "val_auc"),
        ("early_stopping.patience", -1),
        ("early_stopping.restore_best_weights", "yes"),
    ],
)
def test_invalid_training_settings_raise(config, key, value):
    cfg = copy.deepcopy(config)
    section = cfg.training
    *parents, name = key.split(".")
    for parent in parents:
        section = getattr(section, parent)
    setattr(section, name, value)

    x_train, y_train, x_val, y_val = _fake_split()
    model = build_model(cfg, (N_FEATURES,), seed=42)
    with pytest.raises(ValueError):
        train_with_early_stopping(model, x_train, y_train, x_val, y_val, cfg)


def test_median_best_epoch():
    assert median_best_epoch([35, 42, 38, 50]) == 40
    assert median_best_epoch([10, 11]) == 11
    assert median_best_epoch([12]) == 12

    with pytest.raises(ValueError):
        median_best_epoch([])
    with pytest.raises(ValueError):
        median_best_epoch([0, 5])
