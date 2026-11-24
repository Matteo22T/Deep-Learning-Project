from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from tensorflow import keras

from src.models.model import check_input_compatibility, check_target_compatibility

# This module trains a model built with src.models.model.build_model.
# It only runs the training: splitting, scaling, the nested cross-validation
# loop, metrics and model selection are handled elsewhere.
#
# Inside the nested cross-validation it is used in two steps:
#   1. inner folds: train_with_early_stopping() on (inner train, inner validation)
#      gives the best epoch of each inner fold;
#   2. refit: train_fixed_epochs() on the whole outer training set for
#      median_best_epoch(best epochs of the inner folds) epochs.
#
# Every training run needs a new model from build_model: a model that has
# already been trained is rejected, so weights learned on one fold can never
# leak into another one.


@dataclass
class TrainingResult:
    """Outcome of one training run.

    Attributes:
        model: the trained model. With early stopping it holds the weights of
            the best epoch, not the ones of the last epoch.
        history: values per epoch, e.g. {"loss": [...], "val_loss": [...],
            "OS_1y_loss": [...], ...}. Plain floats, so it can be saved as JSON.
        epochs_trained: number of epochs actually run.
        best_epoch: epoch (counted from 1) with the lowest monitored loss.
            None when there is no validation set (refit).
    """

    model: keras.Model
    history: dict[str, list[float]]
    epochs_trained: int
    best_epoch: int | None = None


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def get_training_params(config: SimpleNamespace) -> dict[str, Any]:
    """Read the training settings from config.training."""
    training = getattr(config, "training", SimpleNamespace())
    early_stopping = getattr(training, "early_stopping", SimpleNamespace())
    return {
        "batch_size": getattr(training, "batch_size", None),
        "max_epochs": getattr(training, "max_epochs", None),
        "monitor": getattr(early_stopping, "monitor", None),
        "patience": getattr(early_stopping, "patience", None),
        "restore_best_weights": getattr(early_stopping, "restore_best_weights", None),
    }


def validate_training_params(params: Mapping[str, Any]) -> None:
    """Raise ValueError if a training setting is missing or has an invalid value."""
    batch_size = params.get("batch_size")
    if not _is_int(batch_size) or batch_size <= 0:
        raise ValueError(f"training.batch_size must be a positive integer, got {batch_size!r}.")

    max_epochs = params.get("max_epochs")
    if not _is_int(max_epochs) or max_epochs <= 0:
        raise ValueError(f"training.max_epochs must be a positive integer, got {max_epochs!r}.")

    # Only validation losses are allowed: lower is better, so the best epoch is
    # the one with the minimum value.
    monitor = params.get("monitor")
    if not isinstance(monitor, str) or not monitor.startswith("val_") or not monitor.endswith("loss"):
        raise ValueError(
            "training.early_stopping.monitor must be a validation loss such as 'val_loss', "
            f"got {monitor!r}."
        )

    patience = params.get("patience")
    if not _is_int(patience) or patience < 0:
        raise ValueError(
            f"training.early_stopping.patience must be a non-negative integer, got {patience!r}."
        )

    if not isinstance(params.get("restore_best_weights"), bool):
        raise ValueError(
            "training.early_stopping.restore_best_weights must be true or false, "
            f"got {params.get('restore_best_weights')!r}."
        )


def _check_untrained(model: keras.Model) -> None:
    """Accept only compiled models that have never been trained."""
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        raise ValueError("The model is not compiled: build it with build_model().")
    if int(np.asarray(optimizer.iterations)) > 0:
        raise ValueError(
            "The model has already been trained. Build a new one with build_model() "
            "for every training run, otherwise weights learned on another fold are reused."
        )


def _prepare(
    model: keras.Model, features: Any, labels: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Check the data against the model and convert it to float32 arrays."""
    features = np.asarray(features, dtype="float32")
    check_input_compatibility(model, features)
    check_target_compatibility(model, labels, n_samples=len(features))
    # Labels as columns (n, 1), the same shape as the model outputs.
    prepared = {
        name: np.asarray(values, dtype="float32").reshape(-1, 1) for name, values in labels.items()
    }
    return features, prepared


def _to_plain_history(history: Mapping[str, Sequence[Any]]) -> dict[str, list[float]]:
    return {name: [float(v) for v in values] for name, values in history.items()}


def train_with_early_stopping(
    model: keras.Model,
    x_train: Any,
    y_train: Mapping[str, Any],
    x_val: Any,
    y_val: Mapping[str, Any],
    config: SimpleNamespace,
    seed: int | None = None,
    verbose: int = 0,
) -> TrainingResult:
    """Train `model` on the training set and stop when the validation loss stops improving.

    Used on the inner folds of the nested cross-validation. Training runs for at
    most training.max_epochs epochs and stops after training.early_stopping.patience
    epochs without improvement of the monitored validation loss. With
    restore_best_weights the returned model has the weights of the best epoch.

    Args:
        model: a new model from build_model() (never trained before).
        x_train, x_val: features, shape (n_samples, n_features).
        y_train, y_val: labels as a dict with one 0/1 array per target,
            e.g. {"OS_1y": ..., "PFS_1y": ...}.
        config: configuration loaded with ConfigLoader.
        seed: if given, the random generators are seeded right before training,
            so the order of the samples in each epoch is reproducible.
        verbose: Keras verbosity (0 = silent).
    """
    params = get_training_params(config)
    validate_training_params(params)
    _check_untrained(model)
    x_train, y_train = _prepare(model, x_train, y_train)
    x_val, y_val = _prepare(model, x_val, y_val)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor=params["monitor"],
        patience=params["patience"],
        restore_best_weights=params["restore_best_weights"],
    )

    if seed is not None:
        keras.utils.set_random_seed(seed)

    fit_history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=params["max_epochs"],
        batch_size=params["batch_size"],
        shuffle=True,
        callbacks=[early_stopping],
        verbose=verbose,
    )
    history = _to_plain_history(fit_history.history)

    monitor = params["monitor"]
    if monitor not in history:
        raise ValueError(
            f"Monitored value {monitor!r} not found in the training history. "
            f"Available: {sorted(history)}."
        )
    monitored = history[monitor]
    return TrainingResult(
        model=model,
        history=history,
        epochs_trained=len(monitored),
        best_epoch=int(np.argmin(monitored)) + 1,
    )


def train_fixed_epochs(
    model: keras.Model,
    x_train: Any,
    y_train: Mapping[str, Any],
    n_epochs: int,
    config: SimpleNamespace,
    seed: int | None = None,
    verbose: int = 0,
) -> TrainingResult:
    """Train `model` for exactly `n_epochs` epochs, without validation and early stopping.

    Used for the refit on the whole outer training set, where there is no
    validation set left: the number of epochs comes from the inner folds
    (see median_best_epoch).
    """
    params = get_training_params(config)
    validate_training_params(params)
    if not _is_int(n_epochs) or n_epochs <= 0:
        raise ValueError(f"n_epochs must be a positive integer, got {n_epochs!r}.")
    _check_untrained(model)
    x_train, y_train = _prepare(model, x_train, y_train)

    if seed is not None:
        keras.utils.set_random_seed(seed)

    fit_history = model.fit(
        x_train,
        y_train,
        epochs=int(n_epochs),
        batch_size=params["batch_size"],
        shuffle=True,
        verbose=verbose,
    )
    history = _to_plain_history(fit_history.history)
    return TrainingResult(
        model=model,
        history=history,
        epochs_trained=len(history["loss"]),
        best_epoch=None,
    )


def median_best_epoch(best_epochs: Sequence[int]) -> int:
    """Number of epochs for the refit: median of the best epochs of the inner folds.

    The median is rounded to the nearest integer (halves go up), e.g.
    [35, 42, 38, 50] -> 40 and [10, 11] -> 11. Unlike the mean, it is not
    pulled up or down by a single unusual fold.
    """
    if len(best_epochs) == 0 or not all(_is_int(e) and e > 0 for e in best_epochs):
        raise ValueError(f"best_epochs must be a non-empty list of positive integers, got {best_epochs!r}.")
    return int(math.floor(float(np.median(best_epochs)) + 0.5))
