from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.schemas import BatchRequest, BatchResponse, LoanApplication, Prediction
from src.models.explain import load_artifacts
from src.models.predict import score

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["artifacts"] = load_artifacts()  # load once at startup, not per request
    yield
    state.clear()


app = FastAPI(
    title="Credit Risk Intelligence API",
    description="Default-probability scoring with SHAP explanations. "
                "Output is a model recommendation; the final decision rests with a credit officer.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "artifacts" in state}


@app.get("/model-info")
def model_info():
    return state["artifacts"][2]  # metadata: threshold, AUC, KS, cost assumptions


@app.post("/predict", response_model=Prediction)
def predict(application: LoanApplication):
    return score([application.model_dump()], state["artifacts"])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    results = score([a.model_dump() for a in batch.applications], state["artifacts"])
    approved = sum(r["recommendation"] == "approve" for r in results)
    return {
        "count": len(results),
        "approval_rate": round(approved / len(results), 4),
        "mean_default_probability": round(sum(r["default_probability"] for r in results) / len(results), 4),
        "predictions": results,
    }