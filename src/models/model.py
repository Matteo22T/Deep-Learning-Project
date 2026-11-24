from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
from tensorflow import keras

# This module builds the Keras networks used in the experiments.
# It only defines, builds and compiles the model: loading data, splitting,
# scaling, the nested cross-validation loop, metrics and hyperparameter
# selection are handled elsewhere.
#
# Usage:
#   config = ConfigLoader("configs/experiment_config.yaml").get_config()
#   model = build_model(config, input_shape=X_train.shape[1:], seed=42)
#   check_input_compatibility(model, X_train)
#   check_target_compatibility(model, {"OS_1y": y_os, "PFS_1y": y_pfs})
#   model.fit(X_train, {"OS_1y": y_os, "PFS_1y": y_pfs}, ...)

_ALLOWED_ARCHITECTURES = {"mlp"}
_OPTIMIZERS = {
    "adam": keras.optimizers.Adam,
    "rmsprop": keras.optimizers.RMSprop,
}

# Parameters read from config.model (network structure) and from
# config.training (only the ones needed to compile the model).
_MODEL_KEYS = ("architecture", "hidden_units", "activation", "dropout", "l2")
_COMPILE_KEYS = ("optimizer", "learning_rate", "loss_weights")


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _to_dict(value: Any) -> Any:
    """Recursively convert the SimpleNamespace objects created by ConfigLoader back to dicts."""
    if isinstance(value, SimpleNamespace):
        return {key: _to_dict(item) for key, item in vars(value).items()}
    if isinstance(value, list):
        return [_to_dict(item) for item in value]
    return value


def get_model_params(
    config: SimpleNamespace, overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Collect the parameters needed to build and compile the model.

    Values are taken from config.model and config.training. `overrides` replaces
    some of them (e.g. one combination of the grid search) without modifying
    the config itself.
    """
    model_section = _to_dict(getattr(config, "model", SimpleNamespace()))
    training_section = _to_dict(getattr(config, "training", SimpleNamespace()))

    params = {key: model_section.get(key) for key in _MODEL_KEYS}
    params.update({key: training_section.get(key) for key in _COMPILE_KEYS})

    if overrides:
        unknown = set(overrides) - set(params)
        if unknown:
            raise ValueError(
                f"Unknown model parameters in overrides: {sorted(unknown)}. "
                f"Allowed: {sorted(params)}."
            )
        params.update(overrides)
    return params


def validate_model_params(params: Mapping[str, Any], targets: Sequence[str]) -> None:
    """Raise ValueError if a parameter is missing or has an invalid value."""
    if params.get("architecture") not in _ALLOWED_ARCHITECTURES:
        raise ValueError(
            f"architecture must be one of {sorted(_ALLOWED_ARCHITECTURES)}, "
            f"got {params.get('architecture')!r}."
        )

    hidden_units = params.get("hidden_units")
    if not isinstance(hidden_units, list) or not all(_is_positive_int(u) for u in hidden_units):
        raise ValueError(
            "hidden_units must be a list of positive integers "
            f"(an empty list gives the linear baseline), got {hidden_units!r}."
        )

    activation = params.get("activation")
    if not isinstance(activation, str):
        raise ValueError(f"activation must be a string, got {activation!r}.")
    try:
        keras.activations.get(activation)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Unknown activation function: {activation!r}.") from exc

    dropout = params.get("dropout")
    if not _is_number(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be a number in [0, 1), got {dropout!r}.")

    l2 = params.get("l2")
    if not _is_number(l2) or l2 < 0:
        raise ValueError(f"l2 must be a non-negative number, got {l2!r}.")

    if params.get("optimizer") not in _OPTIMIZERS:
        raise ValueError(
            f"optimizer must be one of {sorted(_OPTIMIZERS)}, got {params.get('optimizer')!r}."
        )

    learning_rate = params.get("learning_rate")
    if not _is_number(learning_rate) or learning_rate <= 0:
        raise ValueError(f"learning_rate must be a positive number, got {learning_rate!r}.")

    loss_weights = params.get("loss_weights")
    if not isinstance(loss_weights, Mapping) or set(loss_weights) != set(targets):
        raise ValueError(
            f"loss_weights must have exactly one entry per target {list(targets)}, "
            f"got {loss_weights!r}."
        )
    if not all(_is_number(w) and w > 0 for w in loss_weights.values()):
        raise ValueError(f"loss_weights must be positive numbers, got {loss_weights!r}.")


def build_model(
    config: SimpleNamespace,
    input_shape: Sequence[int],
    seed: int | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> keras.Model:
    """Build and compile the network described in the config, ready for model.fit().

    Structure: input -> [Dense -> Dropout] for each entry of hidden_units ->
    one Dense(1, sigmoid) output per target. With hidden_units = [] there are
    no hidden layers and each output is a logistic regression (linear baseline).

    Args:
        config: configuration loaded with ConfigLoader.
        input_shape: shape of one sample, usually X_train.shape[1:]
            ((768,) with 'mean' aggregation, (3072,) with 'concat').
        seed: if given, every random generator is seeded before the weights
            are created, so the same seed always gives the same initial network
            (and, on CPU, the same training run).
        overrides: optional values replacing the ones in the config.

    Returns:
        A compiled keras.Model with one output per target, named as the target
        (e.g. "OS_1y", "PFS_1y"). Labels for fit() must be a dict with the same keys.
    """
    targets = list(config.experiment.targets)
    params = get_model_params(config, overrides)
    validate_model_params(params, targets)

    input_shape = tuple(input_shape)
    if len(input_shape) != 1 or not _is_positive_int(int(input_shape[0])):
        raise ValueError(
            "The MLP expects one flat feature vector per sample, i.e. input_shape=(n_features,), "
            f"got {input_shape!r}."
        )
    input_shape = (int(input_shape[0]),)

    if seed is not None:
        keras.utils.set_random_seed(seed)

    inputs = keras.Input(shape=input_shape, name="embedding")
    x = inputs
    for i, units in enumerate(params["hidden_units"], start=1):
        x = keras.layers.Dense(
            units,
            activation=params["activation"],
            kernel_regularizer=keras.regularizers.l2(params["l2"]),
            name=f"hidden_{i}",
        )(x)
        x = keras.layers.Dropout(params["dropout"], name=f"dropout_{i}")(x)

    # One sigmoid head per target, all attached to the same shared representation x.
    outputs = {
        target: keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_regularizer=keras.regularizers.l2(params["l2"]),
            name=target,
        )(x)
        for target in targets
    }

    name = "mlp" if params["hidden_units"] else "linear_baseline"
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)

    optimizer = _OPTIMIZERS[params["optimizer"]](learning_rate=params["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss={target: "binary_crossentropy" for target in targets},
        loss_weights={target: float(params["loss_weights"][target]) for target in targets},
    )
    return model


def check_input_compatibility(model: keras.Model, features: np.ndarray) -> None:
    """Raise ValueError if `features` cannot be fed to `model`.

    Catches the common mistakes before training starts: data aggregated with a
    different strategy than the one the model was built for, empty arrays,
    NaN or infinite values.
    """
    features = np.asarray(features)
    expected = tuple(model.inputs[0].shape)[1:]

    if features.ndim != len(expected) + 1 or tuple(features.shape[1:]) != expected:
        raise ValueError(
            f"Input shape {features.shape} does not match the model input (n_samples, "
            f"{', '.join(str(d) for d in expected)}). Check that the aggregation strategy "
            "of the data is the same one used to build the model."
        )
    if features.shape[0] == 0:
        raise ValueError("Input contains no samples.")
    if not np.issubdtype(features.dtype, np.number):
        raise ValueError(f"Input must be numeric, got dtype {features.dtype}.")
    if not np.all(np.isfinite(features)):
        raise ValueError("Input contains NaN or infinite values.")


def check_target_compatibility(
    model: keras.Model,
    labels: Mapping[str, Any],
    n_samples: int | None = None,
) -> None:
    """Raise ValueError if `labels` does not match the outputs of `model`.

    `labels` must be a dict with one entry per model output (same names as the
    targets), each containing only 0/1 values. If `n_samples` is given, every
    entry must have that length (usually len(X_train)).
    """
    output_names = set(model.output_names)
    if not isinstance(labels, Mapping) or set(labels) != output_names:
        given = sorted(labels) if isinstance(labels, Mapping) else type(labels).__name__
        raise ValueError(
            f"Labels must be a dict with keys {sorted(output_names)}, got {given}."
        )

    for name, values in labels.items():
        array = np.asarray(values)
        if array.ndim == 2 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim != 1:
            raise ValueError(f"Labels for {name!r} must be one-dimensional, got shape {array.shape}.")
        if n_samples is not None and len(array) != n_samples:
            raise ValueError(
                f"Labels for {name!r} have {len(array)} values, expected {n_samples}."
            )
        if not np.isin(array, (0, 1)).all():
            raise ValueError(f"Labels for {name!r} must contain only 0 and 1.")
