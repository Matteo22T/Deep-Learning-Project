from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

"""
    Configuration loader for YAML files. with this you can do like this:
    config = ConfigLoader("path/to/config.yaml").get_config()
    and then access the configuration values as attributes, e.g.:
    config.experiment.name
    config.splitting.n_splits
    config.preprocessing.features.prefix
    config.preprocessing.sequences.order
    ecc..
"""


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_strict_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _to_namespace(value: Any) -> Any:
    """Recursively convert mappings to SimpleNamespace and lists element-wise."""
    if isinstance(value, dict):
        return SimpleNamespace(**{str(k): _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


class ConfigLoader:
    """Loads, validates, and exposes YAML settings as nested SimpleNamespace objects."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.raw_data: dict[str, Any] = self._load_yaml(self.config_path)
        self._validate(self.raw_data)
        self.config: SimpleNamespace = _to_namespace(self.raw_data)

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Top-level YAML configuration must be a mapping, got {type(loaded).__name__}."
            )
        return loaded

    @staticmethod
    def _validate(data: dict[str, Any]) -> None:
        # 1. Experiment settings
        exp = data.get("experiment")
        if not isinstance(exp, dict):
            raise ValueError("Missing or invalid 'experiment' section.")
        random_seed = exp.get("random_seed")
        if not _is_strict_int(random_seed):
            raise ValueError("config.experiment.random_seed must be an integer.")

        # 2. Splitting settings
        split = data.get("splitting")
        if not isinstance(split, dict):
            raise ValueError("Missing or invalid 'splitting' section.")

        n_splits = split.get("n_splits")
        if not _is_strict_int(n_splits) or n_splits < 2:
            raise ValueError("config.splitting.n_splits must be an integer >= 2.")

        ratios = split.get("synthetic_ratios")
        if not isinstance(ratios, list) or len(ratios) == 0:
            raise ValueError(
                "config.splitting.synthetic_ratios must be a non-empty list."
            )

        seen_ratios: set[float] = set()
        for r in ratios:
            if not _is_strict_number(r):
                raise ValueError(f"Invalid synthetic ratio type/value: {r!r}")
            r_float = float(r)
            if r_float < 0.0:
                raise ValueError(f"Synthetic ratio cannot be negative: {r!r}")
            if r_float in seen_ratios:
                raise ValueError(f"Duplicate synthetic ratio is not allowed: {r!r}")
            seen_ratios.add(r_float)

        # Ensure ratios are ordered consistently for nested quota slicing (0% <= 25% <= 50% <= ...)
        ratio_floats = [float(r) for r in ratios]
        if ratio_floats != sorted(ratio_floats):
            raise ValueError(
                "config.splitting.synthetic_ratios must be sorted in non-decreasing order "
                "for nested quota sampling."
            )

        stride = split.get("fold_seed_stride", 1)
        if not _is_strict_int(stride) or stride < 0:
            raise ValueError(
                "config.splitting.fold_seed_stride must be a non-negative integer."
            )

        tol = split.get("quota_integer_tol", 1e-9)
        if not _is_strict_number(tol) or float(tol) <= 0.0:
            raise ValueError(
                "config.splitting.quota_integer_tol must be a positive number."
            )

        # 3. Preprocessing settings
        prep = data.get("preprocessing")
        if not isinstance(prep, dict):
            raise ValueError("Missing or invalid 'preprocessing' section.")

        feats = prep.get("features")
        if (
            not isinstance(feats, dict)
            or not isinstance(feats.get("prefix"), str)
            or not feats["prefix"]
        ):
            raise ValueError(
                "config.preprocessing.features.prefix must be a non-empty string."
            )

        seqs = prep.get("sequences")
        order = seqs.get("order") if isinstance(seqs, dict) else None
        if (
            not isinstance(order, list)
            or len(order) == 0
            or not all(isinstance(s, str) and s for s in order)
        ):
            raise ValueError(
                "config.preprocessing.sequences.order must be a non-empty list of strings."
            )
        if len(set(order)) != len(order):
            raise ValueError(
                "config.preprocessing.sequences.order must not contain duplicate entries."
            )

        scaling = prep.get("scaling")
        eps = scaling.get("epsilon") if isinstance(scaling, dict) else None
        if not _is_strict_number(eps) or float(eps) <= 0.0:
            raise ValueError(
                "config.preprocessing.scaling.epsilon must be a positive number."
            )

        agg = prep.get("aggregation")
        if not isinstance(agg, dict):
            raise ValueError("Missing or invalid 'preprocessing.aggregation' section.")
        strategy = agg.get("strategy")
        if strategy not in ("mean", "concat", "per_sequence"):
            raise ValueError(
                f"Invalid aggregation strategy {strategy!r}; expected 'mean', 'concat', or 'per_sequence'."
            )
        if not isinstance(agg.get("strict_sequences"), bool):
            raise ValueError(
                "config.preprocessing.aggregation.strict_sequences must be a boolean."
            )
        if not isinstance(agg.get("allow_duplicates"), bool):
            raise ValueError(
                "config.preprocessing.aggregation.allow_duplicates must be a boolean."
            )

    def get_config(self) -> SimpleNamespace:
        """Return the validated configuration SimpleNamespace."""
        return self.config
