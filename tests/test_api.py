from fastapi.testclient import TestClient

from src.api.main import app

GOOD = {
    "loan_amnt": 10000, "term": 36, "annual_inc": 90000, "dti": 8, "fico_score": 790,
    "open_acc": 10, "revol_bal": 8000, "revol_util": 15, "total_acc": 25, "mort_acc": 2,
    "pub_rec_bankruptcies": 0, "emp_length": 10, "earliest_cr_line": "2005-06-01",
    "home_ownership": "MORTGAGE", "verification_status": "Source Verified",
    "purpose": "credit_card",
}
RISKY = {
    "loan_amnt": 25000, "term": 60, "annual_inc": 40000, "dti": 35, "fico_score": 665,
    "delinq_2yrs": 1, "inq_last_6mths": 3, "open_acc": 14, "pub_rec": 1,
    "revol_bal": 15000, "revol_util": 85, "total_acc": 12, "mort_acc": 0,
    "pub_rec_bankruptcies": 1, "emp_length": 1, "earliest_cr_line": "2012-03-01",
    "home_ownership": "RENT", "verification_status": "Not Verified",
    "purpose": "small_business",
}


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_risky_applicant_scores_higher_and_decision_matches_threshold():
    with TestClient(app) as client:
        good = client.post("/predict", json=GOOD).json()
        risky = client.post("/predict", json=RISKY).json()
    assert 0 <= good["default_probability"] <= 1
    assert risky["default_probability"] > good["default_probability"]
    for res in (good, risky):
        assert (res["default_probability"] >= res["threshold"]) == (res["recommendation"] == "decline")
    assert all(f["shap"] > 0 for f in risky["risk_drivers"])


def test_missing_optional_fields_still_scores():
    payload = {k: v for k, v in GOOD.items() if k not in ("emp_length", "mort_acc", "revol_util")}
    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 200


def test_invalid_input_rejected():
    with TestClient(app) as client:
        assert client.post("/predict", json={**GOOD, "fico_score": 1000}).status_code == 422
        assert client.post("/predict", json={**GOOD, "earliest_cr_line": "2999-01-01"}).status_code == 422


def test_batch():
    with TestClient(app) as client:
        r = client.post("/predict/batch", json={"applications": [GOOD, RISKY]})
    body = r.json()
    assert r.status_code == 200 and body["count"] == 2 and len(body["predictions"]) == 2