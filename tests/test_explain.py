import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.models.explain import top_reasons


def test_top_reasons_sorted_and_additive():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 4)), columns=list("abcd"))
    y = (X["a"] + 0.5 * X["b"] + rng.normal(size=500) > 0).astype(int)
    model = xgb.XGBClassifier(n_estimators=30, max_depth=3).fit(X, y)
    explainer = shap.TreeExplainer(model)

    reasons = top_reasons(explainer, X.iloc[[0]], n=3)
    assert len(reasons) == 3
    magnitudes = [abs(r["shap"]) for r in reasons]
    assert magnitudes == sorted(magnitudes, reverse=True)

    exp = explainer(X.iloc[[0]])
    raw = model.predict(X.iloc[[0]], output_margin=True)[0]
    assert np.isclose(exp.base_values[0] + exp.values[0].sum(), raw, atol=1e-3)