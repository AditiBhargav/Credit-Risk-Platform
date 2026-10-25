from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.explain import load_artifacts, top_reasons

OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

model, columns, meta, explainer = load_artifacts()

test = pd.read_csv("data/processed/test.csv")
X = test.drop(columns="target")[columns]  # enforce training column order
X_sample = X.sample(5000, random_state=42).reset_index(drop=True)

explanation = explainer(X_sample)
proba = model.predict_proba(X_sample)[:, 1]


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT / name}")


# ---------- Global explanations ----------
shap.plots.beeswarm(explanation, max_display=15, show=False)
save("shap_summary.png")

shap.plots.bar(explanation, max_display=15, show=False)
save("shap_importance.png")

shap.plots.scatter(explanation[:, "fico_avg"], color=explanation[:, "term"], show=False)
save("shap_fico_dependence.png")

# ---------- Local explanations ----------
cases = {
    "high_risk": int(proba.argmax()),
    "low_risk": int(proba.argmin()),
    "borderline": int(np.abs(proba - meta["threshold"]).argmin()),
}

for name, i in cases.items():
    shap.plots.waterfall(explanation[i], max_display=10, show=False)
    save(f"shap_waterfall_{name}.png")

    print(f"\n=== {name}: default probability {proba[i]:.1%} "
          f"(approval threshold {meta['threshold']:.0%}) ===")
    for r in top_reasons(explainer, X_sample.iloc[[i]], n=5):
        print(f"  {r['feature']:<34} value={r['value']}  "
              f"shap={r['shap']:+.3f}  ({r['effect']})")

# ---------- Proof that SHAP adds up exactly ----------
i = cases["borderline"]
raw = model.predict(X_sample.iloc[[i]], output_margin=True)[0]
rebuilt = explanation.base_values[i] + explanation.values[i].sum()
print(f"\nModel raw score: {raw:.4f} | base value + sum(SHAP): {rebuilt:.4f}")
print(f"Probability from raw score: {1 / (1 + np.exp(-raw)):.4f} | model output: {proba[i]:.4f}")