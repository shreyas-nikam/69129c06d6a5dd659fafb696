import pytest
import numpy as np
import pandas as pd
import pandas.testing as pdt

from definition_ed55d82e2ac349209e3bcfd6a1417b80 import simulate_model_performance_by_sample_bias


def _make_binary_df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    group = rng.choice(["A", "B", "C"], size=n, p=[0.6, 0.3, 0.1])
    # Create a binary target with some signal
    logits = 0.8 * f1 - 0.5 * f2 + (group == "A") * 0.3 + (group == "B") * -0.2
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < probs).astype(int)
    return pd.DataFrame({"f1": f1, "f2": f2, "target": y, "group": group})


def _make_multiclass_df(n=150, seed=1):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    group = rng.choice(["X", "Y"], size=n, p=[0.7, 0.3])
    # Multiclass target based on noisy thresholds
    raw = 1.2 * f1 - 0.7 * f2 + (group == "X") * 0.2 + rng.normal(scale=0.5, size=n)
    y = np.digitize(raw, bins=[-0.5, 0.5])  # values in {0,1,2}
    return pd.DataFrame({"f1": f1, "f2": f2, "target": y, "group": group})


def _has_overall_metric_column(columns, metric_keyword):
    cl = [c.lower() for c in columns]
    return any(("overall" in c and metric_keyword.lower()[:3] in c) for c in cl)


def _has_subgroup_column(columns):
    cl = set(c.lower() for c in columns)
    # Common possibilities implementers might use
    candidates = {"group", "subgroup", "demographic_group"}
    return any(c in cl for c in candidates)


def test_runs_binary_and_multiclass_and_overall_metric_present():
    df_bin = _make_binary_df()
    df_mc = _make_multiclass_df()

    res_bin = simulate_model_performance_by_sample_bias(
        df=df_bin,
        feature_columns=["f1", "f2"],
        target_column="target",
        demographic_column="group",
        model_type="logistic",
        metric="accuracy",
        n_trials=3,
        test_size=0.2,
        random_state=42,
    )
    assert isinstance(res_bin, pd.DataFrame)
    assert len(res_bin) >= 3
    cols_bin = set(res_bin.columns)
    # Expect some metadata about configuration
    assert "model_type" in cols_bin
    assert "metric" in cols_bin
    assert any(k in cols_bin for k in ("test_size", "split_test_size"))
    assert any(k in cols_bin for k in ("random_state", "seed"))
    assert _has_overall_metric_column(res_bin.columns, "accuracy")

    res_mc = simulate_model_performance_by_sample_bias(
        df=df_mc,
        feature_columns=["f1", "f2"],
        target_column="target",
        demographic_column="group",
        model_type="logistic",
        metric="accuracy",
        n_trials=2,
        test_size=0.25,
        random_state=123,
    )
    assert isinstance(res_mc, pd.DataFrame)
    assert len(res_mc) >= 2
    assert _has_overall_metric_column(res_mc.columns, "accuracy")


def test_includes_demographic_breakdown_when_provided():
    df = _make_binary_df()
    res = simulate_model_performance_by_sample_bias(
        df=df,
        feature_columns=["f1", "f2"],
        target_column="target",
        demographic_column="group",
        model_type="logistic",
        metric="accuracy",
        n_trials=2,
        test_size=0.3,
        random_state=0,
    )
    assert isinstance(res, pd.DataFrame)
    assert _has_subgroup_column(res.columns)


def test_no_demographic_breakdown_when_none():
    df = _make_binary_df()
    res = simulate_model_performance_by_sample_bias(
        df=df,
        feature_columns=["f1", "f2"],
        target_column="target",
        demographic_column=None,
        model_type="logistic",
        metric="accuracy",
        n_trials=2,
        test_size=0.3,
        random_state=0,
    )
    assert isinstance(res, pd.DataFrame)
    # Expect no subgroup breakdown columns when demographic_column is None
    assert not _has_subgroup_column(res.columns)


def test_invalid_metric_raises():
    df = _make_binary_df()
    with pytest.raises(ValueError):
        simulate_model_performance_by_sample_bias(
            df=df,
            feature_columns=["f1", "f2"],
            target_column="target",
            demographic_column="group",
            model_type="logistic",
            metric="not_a_metric",
            n_trials=2,
            test_size=0.2,
            random_state=0,
        )


def test_invalid_inputs_raise():
    df = _make_binary_df()
    # Invalid test_size
    with pytest.raises(ValueError):
        simulate_model_performance_by_sample_bias(
            df=df,
            feature_columns=["f1", "f2"],
            target_column="target",
            demographic_column="group",
            model_type="logistic",
            metric="accuracy",
            n_trials=2,
            test_size=1.0,  # boundary invalid (should be < 1)
            random_state=0,
        )
    # Empty features
    with pytest.raises((ValueError, TypeError)):
        simulate_model_performance_by_sample_bias(
            df=df,
            feature_columns=[],
            target_column="target",
            demographic_column="group",
            model_type="logistic",
            metric="accuracy",
            n_trials=2,
            test_size=0.2,
            random_state=0,
        )
