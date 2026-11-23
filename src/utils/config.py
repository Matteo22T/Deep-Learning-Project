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

_ALLOWED_TARGETS = {"OS_1y", "PFS_1y"}
_ALLOWED_TUNING_METRICS = {"roc_auc", "pr_auc", "brier_score"}
_ALLOWED_TUNING_AGGREGATIONS = {"mean", "concat"}


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

        if "targets" in exp:
            targets = exp["targets"]
            if (
                not isinstance(targets, list)
                or len(targets) == 0
                or not all(isinstance(t, str) and t in _ALLOWED_TARGETS for t in targets)
            ):
                raise ValueError(
                    f"config.experiment.targets must be a non-empty list of valid targets "
                    f"from {sorted(_ALLOWED_TARGETS)}."
                )
            if len(set(targets)) != len(targets):
                raise ValueError("config.experiment.targets must not contain duplicates.")

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

        # 4. Optional evaluation settings (experiment_config.yaml)
        if "evaluation" in data:
            ConfigLoader._validate_evaluation(data["evaluation"])

        # 5. Optional tuning settings (experiment_config.yaml)
        if "tuning" in data:
            ConfigLoader._validate_tuning(data["tuning"])

        # 6. Optional framework-agnostic model/training placeholders
        if "model" in data and not isinstance(data["model"], dict):
            raise ValueError("config.model must be a mapping when provided.")
        if "training" in data and not isinstance(data["training"], dict):
            raise ValueError("config.training must be a mapping when provided.")

    @staticmethod
    def _validate_evaluation(ev: Any) -> None:
        if not isinstance(ev, dict):
            raise ValueError("Invalid 'evaluation' section: expected a mapping.")

        threshold = ev.get("decision_threshold")
        if not _is_strict_number(threshold) or not (0.0 < float(threshold) < 1.0):
            raise ValueError(
                "config.evaluation.decision_threshold must be a number strictly in (0, 1)."
            )

        patient_agg = ev.get("patient_aggregation")
        if patient_agg != "mean_prob":
            raise ValueError(
                f"Invalid config.evaluation.patient_aggregation {patient_agg!r}; expected 'mean_prob'."
            )

        n_resamples = ev.get("bootstrap_resamples")
        if not _is_strict_int(n_resamples) or n_resamples < 100:
            raise ValueError(
                "config.evaluation.bootstrap_resamples must be an integer >= 100."
            )

        if not isinstance(ev.get("bootstrap_stratified"), bool):
            raise ValueError(
                "config.evaluation.bootstrap_stratified must be a boolean."
            )

        conf_level = ev.get("confidence_level")
        if not _is_strict_number(conf_level) or not (0.0 < float(conf_level) < 1.0):
            raise ValueError(
                "config.evaluation.confidence_level must be a number strictly in (0, 1)."
            )

        if not isinstance(ev.get("check_clinical_consistency"), bool):
            raise ValueError(
                "config.evaluation.check_clinical_consistency must be a boolean."
            )

        if not isinstance(ev.get("run_exploratory_consistency_correction"), bool):
            raise ValueError(
                "config.evaluation.run_exploratory_consistency_correction must be a boolean."
            )

    @staticmethod
    def _validate_tuning(tuning: Any) -> None:
        if not isinstance(tuning, dict):
            raise ValueError("Invalid 'tuning' section: expected a mapping.")

        protocol = tuning.get("protocol")
        if protocol != "nested_cv":
            raise ValueError(
                f"Invalid config.tuning.protocol {protocol!r}; expected 'nested_cv'."
            )

        inner_n_splits = tuning.get("inner_n_splits")
        if not _is_strict_int(inner_n_splits) or inner_n_splits < 2:
            raise ValueError(
                "config.tuning.inner_n_splits must be an integer >= 2."
            )

        inner_stride = tuning.get("inner_seed_stride", 100)
        if not _is_strict_int(inner_stride) or inner_stride < 0:
            raise ValueError(
                "config.tuning.inner_seed_stride must be a non-negative integer."
            )

        selection_metric = tuning.get("selection_metric")
        if (
            not isinstance(selection_metric, str)
            or selection_metric not in _ALLOWED_TUNING_METRICS
        ):
            raise ValueError(
                f"Invalid config.tuning.selection_metric {selection_metric!r}; "
                f"expected one of {sorted(_ALLOWED_TUNING_METRICS)}."
            )

        if "tie_breaker_metrics" in tuning:
            tie_breakers = tuning["tie_breaker_metrics"]
            if (
                not isinstance(tie_breakers, list)
                or not all(
                    isinstance(m, str) and m in _ALLOWED_TUNING_METRICS
                    for m in tie_breakers
                )
            ):
                raise ValueError(
                    f"config.tuning.tie_breaker_metrics must be a list of metrics "
                    f"from {sorted(_ALLOWED_TUNING_METRICS)}."
                )
            if len(set(tie_breakers)) != len(tie_breakers):
                raise ValueError(
                    "config.tuning.tie_breaker_metrics must not contain duplicates."
                )
            if selection_metric in tie_breakers:
                raise ValueError(
                    "config.tuning.tie_breaker_metrics must not duplicate selection_metric."
                )

        refit_strategy = tuning.get("refit_epoch_strategy")
        if refit_strategy != "median_best_epoch":
            raise ValueError(
                f"Invalid config.tuning.refit_epoch_strategy {refit_strategy!r}; "
                "expected 'median_best_epoch'."
            )

        search_space = tuning.get("search_space")
        if not isinstance(search_space, dict):
            raise ValueError("Missing or invalid 'tuning.search_space' section.")

        agg_strategies = search_space.get("aggregation_strategies")
        if (
            not isinstance(agg_strategies, list)
            or len(agg_strategies) == 0
            or not all(
                isinstance(s, str) and s in _ALLOWED_TUNING_AGGREGATIONS
                for s in agg_strategies
            )
        ):
            raise ValueError(
                f"config.tuning.search_space.aggregation_strategies must be a non-empty "
                f"list of strategies from {sorted(_ALLOWED_TUNING_AGGREGATIONS)}."
            )
        if len(set(agg_strategies)) != len(agg_strategies):
            raise ValueError(
                "config.tuning.search_space.aggregation_strategies must not contain duplicates."
            )

    def get_config(self) -> SimpleNamespace:
        """Return the validated configuration SimpleNamespace."""
        return self.config