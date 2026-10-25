import json
import os
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
BATCH_CHUNK = 500
INT_COLS = ["term", "fico_score", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
            "total_acc", "mort_acc", "pub_rec_bankruptcies", "emp_length"]
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
            "small_business", "car", "medical", "moving", "vacation", "house", "wedding",
            "renewable_energy", "educational"]
EXAMPLES = [
    {"loan_amnt": 12000, "term": 36, "annual_inc": 65000, "dti": 18.5, "fico_score": 705,
     "delinq_2yrs": 0, "inq_last_6mths": 1, "open_acc": 9, "pub_rec": 0, "revol_bal": 11000,
     "revol_util": 52.0, "total_acc": 21, "mort_acc": 1, "pub_rec_bankruptcies": 0,
     "emp_length": 5, "earliest_cr_line": "2008-06-01", "home_ownership": "RENT",
     "verification_status": "Source Verified", "purpose": "debt_consolidation",
     "application_type": "Individual"},
    {"loan_amnt": 25000, "term": 60, "annual_inc": 40000, "dti": 35, "fico_score": 665,
     "delinq_2yrs": 1, "inq_last_6mths": 3, "open_acc": 14, "pub_rec": 1, "revol_bal": 15000,
     "revol_util": None, "total_acc": 12, "mort_acc": None, "pub_rec_bankruptcies": 1,
     "emp_length": None, "earliest_cr_line": "2012-03-01", "home_ownership": "RENT",
     "verification_status": "Not Verified", "purpose": "small_business",
     "application_type": "Individual"},
]

st.set_page_config(page_title="Credit Risk Intelligence", page_icon="🏦", layout="wide")


# ---------------- API helpers ----------------
def api_post(path, payload):
    try:
        return httpx.post(f"{API_URL}{path}", json=payload, timeout=120)
    except httpx.HTTPError:
        st.error(f"Cannot reach the scoring API at {API_URL}. "
                 "Start it in another terminal with: `uvicorn src.api.main:app`")
        st.stop()


def api_get(path):
    try:
        r = httpx.get(f"{API_URL}{path}", timeout=10)
        return r.json() if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


def show_validation_errors(resp, row_offset=0):
    st.error("The API rejected the input. First problems found:")
    for err in resp.json().get("detail", [])[:8]:
        loc = [str(x) for x in err["loc"] if x != "body"]
        if len(loc) >= 3 and loc[0] == "applications":
            where = f"data row {int(loc[1]) + 1 + row_offset}, field '{loc[2]}'"
        else:
            where = ".".join(loc)
        st.write(f"- **{where}**: {err['msg']}")


# ---------------- Display helpers ----------------
def fmt(v):
    return "not provided" if v is None else f"{v:,.2f}".rstrip("0").rstrip(".")


def gauge(p, t):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Default probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#264653"},
            "steps": [
                {"range": [0, 50 * t], "color": "#b7e4c7"},
                {"range": [50 * t, 100 * t], "color": "#ffe8a3"},
                {"range": [100 * t, 200 * t], "color": "#ffb38a"},
                {"range": [200 * t, 100], "color": "#f28b82"},
            ],
            "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.8, "value": t * 100},
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def contribution_chart(pred):
    df = pd.DataFrame(pred["risk_drivers"] + pred["protective_factors"])
    if df.empty:
        return None
    df = df.sort_values("shap")
    fig = go.Figure(go.Bar(
        x=df["shap"], y=df["label"], orientation="h",
        marker_color=np.where(df["shap"] > 0, "#d1495b", "#2a9d8f"),
        customdata=df["value"],
        hovertemplate="%{y}<br>value: %{customdata}<br>impact: %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="What moved this score (red raises risk, green lowers it)",
        xaxis_title="Impact on risk score (log-odds)",
        height=340, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def render_prediction(pred):
    p, t = pred["default_probability"], pred["threshold"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Default probability", f"{p:.1%}")
    c2.metric("Risk band", pred["risk_band"].replace("_", " ").title())
    c3.metric("Model recommendation", pred["recommendation"].upper(),
              help=f"Decline when probability is at or above {t:.0%} (cost-based threshold)")

    left, right = st.columns([1, 2])
    left.plotly_chart(gauge(p, t), use_container_width=True)
    fig = contribution_chart(pred)
    if fig:
        right.plotly_chart(fig, use_container_width=True)

    if pred["risk_drivers"]:
        st.markdown("**Principal reasons (factors raising risk)**")
        for f in pred["risk_drivers"]:
            st.markdown(f"- {f['label']}: {fmt(f['value'])}")
    st.caption("Model recommendation only. The final credit decision rests with the credit officer.")


# ---------------- Batch helpers ----------------
def score_batch(df):
    df = df.copy()
    int_cols = [c for c in INT_COLS if c in df.columns]
    try:
        df[int_cols] = df[int_cols].astype("Int64")
    except (TypeError, ValueError):
        st.error("Some whole-number columns contain decimals or text. Please check the file.")
        return None

    records = json.loads(df.to_json(orient="records"))
    preds = []
    bar = st.progress(0.0, text="Scoring...")
    for start in range(0, len(records), BATCH_CHUNK):
        resp = api_post("/predict/batch", {"applications": records[start:start + BATCH_CHUNK]})
        if resp.status_code == 422:
            bar.empty()
            show_validation_errors(resp, row_offset=start)
            return None
        if resp.status_code != 200:
            bar.empty()
            st.error(f"API error {resp.status_code}: {resp.text[:300]}")
            return None
        preds.extend(resp.json()["predictions"])
        bar.progress(min((start + BATCH_CHUNK) / len(records), 1.0))
    bar.empty()

    res = pd.DataFrame({
        "default_probability": [p["default_probability"] for p in preds],
        "risk_band": [p["risk_band"] for p in preds],
        "recommendation": [p["recommendation"] for p in preds],
        "top_risk_driver": [p["risk_drivers"][0]["label"] if p["risk_drivers"] else "" for p in preds],
    })
    return pd.concat([df.reset_index(drop=True), res], axis=1), preds[0]["threshold"]


def render_batch(out, t):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applications", f"{len(out):,}")
    c2.metric("Approval rate", f"{(out['recommendation'] == 'approve').mean():.1%}")
    c3.metric("Mean default probability", f"{out['default_probability'].mean():.1%}")
    c4.metric("Flagged for decline", f"{(out['recommendation'] == 'decline').sum():,}")

    fig = px.histogram(out, x="default_probability", color="recommendation", nbins=40,
                       color_discrete_map={"approve": "#2a9d8f", "decline": "#d1495b"})
    fig.add_vline(x=t, line_dash="dash", annotation_text="threshold")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(out.sort_values("default_probability", ascending=False), use_container_width=True)
    st.download_button("Download scored results", out.to_csv(index=False).encode(),
                       "scored_applications.csv", "text/csv")


# ---------------- Page ----------------
st.title("🏦 Credit Risk Intelligence Platform")
st.caption("Explainable default-risk scoring for credit officers")

tab_single, tab_batch, tab_model = st.tabs(["Score an applicant", "Batch scoring", "Model card"])

# ----- Tab 1: single applicant -----
with tab_single:
    with st.form("applicant_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Loan**")
            loan_amnt = st.number_input("Loan amount ($)", 500, 100000, 12000, step=500)
            term = st.selectbox("Term (months)", [36, 60])
            purpose = st.selectbox("Purpose", PURPOSES)
            application_type = st.selectbox("Application type", ["Individual", "Joint App"])
        with c2:
            st.markdown("**Applicant**")
            annual_inc = st.number_input("Annual income ($)", 1000, 10_000_000, 65000, step=1000)
            emp_length = st.selectbox("Employment length (years)",
                                      ["Not provided"] + list(range(0, 11)), index=6)
            home_ownership = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
            verification_status = st.selectbox("Income verification",
                                               ["Verified", "Source Verified", "Not Verified"], index=1)
            fico_score = st.number_input("FICO score", 300, 850, 705)
        with c3:
            st.markdown("**Credit profile**")
            dti = st.number_input("Debt-to-income (%)", 0.0, 200.0, 18.5, step=0.5)
            revol_bal = st.number_input("Revolving balance ($)", 0.0, 1_000_000.0, 11000.0, step=500.0)
            revol_util = st.number_input("Revolving utilization (%)", 0.0, 300.0, 52.0, step=1.0)
            open_acc = st.number_input("Open credit lines", 0, 80, 9)
            total_acc = st.number_input("Total credit accounts", 0, 150, 21)
            earliest = st.date_input("Earliest credit line", value=date(2008, 6, 1),
                                     min_value=date(1950, 1, 1), max_value=date.today())
        c4, c5, c6, c7 = st.columns(4)
        inq_last_6mths = c4.number_input("Inquiries (6 months)", 0, 30, 1)
        delinq_2yrs = c5.number_input("Delinquencies (2 years)", 0, 30, 0)
        pub_rec = c6.number_input("Public records", 0, 20, 0)
        mort_acc = c7.number_input("Mortgage accounts", 0, 30, 1)
        pub_rec_bankruptcies = st.number_input("Public-record bankruptcies", 0, 10, 0)
        submitted = st.form_submit_button("Score applicant", type="primary")

    if submitted:
        payload = {
            "loan_amnt": loan_amnt, "term": term, "annual_inc": annual_inc, "dti": dti,
            "fico_score": int(fico_score), "delinq_2yrs": int(delinq_2yrs),
            "inq_last_6mths": int(inq_last_6mths), "open_acc": int(open_acc),
            "pub_rec": int(pub_rec), "revol_bal": revol_bal, "revol_util": revol_util,
            "total_acc": int(total_acc), "mort_acc": int(mort_acc),
            "pub_rec_bankruptcies": int(pub_rec_bankruptcies),
            "emp_length": None if emp_length == "Not provided" else int(emp_length),
            "earliest_cr_line": earliest.isoformat(), "home_ownership": home_ownership,
            "verification_status": verification_status, "purpose": purpose,
            "application_type": application_type,
        }
        resp = api_post("/predict", payload)
        if resp.status_code == 200:
            render_prediction(resp.json())
        elif resp.status_code == 422:
            show_validation_errors(resp)
        else:
            st.error(f"API error {resp.status_code}: {resp.text[:300]}")

# ----- Tab 2: batch -----
with tab_batch:
    st.subheader("Score a CSV of applications")
    st.download_button("Download CSV template",
                       pd.DataFrame(EXAMPLES).to_csv(index=False).encode(),
                       "applications_template.csv", "text/csv")
    file = st.file_uploader("Upload applications CSV", type="csv")
    if file is not None:
        df_in = pd.read_csv(file)
        st.write(f"Loaded {len(df_in):,} applications")
        st.dataframe(df_in.head(), use_container_width=True)
        if st.button("Score all applications", type="primary"):
            result = score_batch(df_in)
            if result is not None:
                st.session_state["batch_result"] = result
    if "batch_result" in st.session_state:
        render_batch(*st.session_state["batch_result"])

# ----- Tab 3: model card -----
with tab_model:
    info = api_get("/model-info")
    if info is None:
        st.warning("Could not load model info from the API.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test AUC", f"{info['test_auc']:.3f}")
        m2.metric("Test PR-AUC", f"{info['test_pr_auc']:.3f}")
        m3.metric("Test KS", f"{info['test_ks']:.3f}")
        m4.metric("Decision threshold", f"{info['threshold']:.0%}")
        st.write(f"The threshold minimises expected cost, assuming approving a defaulter costs "
                 f"{info['cost_fn']}x as much as rejecting a good borrower "
                 f"(cost ratio {info['cost_fn']}:{info['cost_fp']}).")

    st.markdown("### Global explanation (SHAP)")
    for name, caption in [("shap_summary.png", "Each dot is an applicant. Right = raises risk."),
                          ("shap_importance.png", "Average absolute impact per feature.")]:
        path = Path("reports/figures") / name
        if path.exists():
            st.image(str(path), caption=caption)

    st.markdown("### Known limitations")
    st.markdown(
        "- Trained on US LendingClub loans (issued 2012-2014) and tested on 2015 loans. "
        "It has **not** been validated on Indian borrowers, so treat it as an illustration.\n"
        "- Discrimination is moderate (AUC about 0.72), typical for application-time data.\n"
        "- On the 2015 test period the model predicted about 17.7% average default against "
        "20.2% actual, which is population drift and the reason monitoring exists.\n"
        "- SHAP explains what the model learned, not causes. Some features (for example revolving "
        "balance) behave counterintuitively once other features are held fixed, so reason wording "
        "needs compliance review before real use.\n"
        "- Lender-assigned grade, interest rate and payment features were excluded to prevent "
        "the model from copying an existing pricing model."
    )