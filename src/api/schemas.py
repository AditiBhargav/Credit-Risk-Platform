from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class LoanApplication(BaseModel):
    loan_amnt: float = Field(gt=0, le=100_000)
    term: Literal[36, 60]
    annual_inc: float = Field(gt=0)
    dti: float = Field(ge=0, le=200, description="Debt-to-income ratio in percent")
    fico_score: int = Field(ge=300, le=850)
    delinq_2yrs: int = Field(ge=0, default=0)
    inq_last_6mths: int = Field(ge=0, default=0)
    open_acc: int = Field(ge=0)
    pub_rec: int = Field(ge=0, default=0)
    revol_bal: float = Field(ge=0)
    revol_util: Optional[float] = Field(default=None, ge=0)
    total_acc: int = Field(ge=0)
    mort_acc: Optional[int] = Field(default=None, ge=0)
    pub_rec_bankruptcies: Optional[int] = Field(default=None, ge=0)
    emp_length: Optional[int] = Field(default=None, ge=0, le=10,
                                      description="0 = under 1 year, 10 = 10+ years")
    earliest_cr_line: date
    home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    verification_status: Literal["Verified", "Source Verified", "Not Verified"]
    purpose: Literal[
        "debt_consolidation", "credit_card", "home_improvement", "other",
        "major_purchase", "small_business", "car", "medical", "moving",
        "vacation", "house", "wedding", "renewable_energy", "educational",
    ]
    application_type: Literal["Individual", "Joint App"] = "Individual"

    @field_validator("earliest_cr_line")
    @classmethod
    def not_in_future(cls, v):
        if v > date.today():
            raise ValueError("earliest_cr_line cannot be in the future")
        return v

    model_config = {"json_schema_extra": {"example": {
        "loan_amnt": 12000, "term": 36, "annual_inc": 65000, "dti": 18.5,
        "fico_score": 705, "delinq_2yrs": 0, "inq_last_6mths": 1, "open_acc": 9,
        "pub_rec": 0, "revol_bal": 11000, "revol_util": 52.0, "total_acc": 21,
        "mort_acc": 1, "pub_rec_bankruptcies": 0, "emp_length": 5,
        "earliest_cr_line": "2008-06-01", "home_ownership": "RENT",
        "verification_status": "Source Verified", "purpose": "debt_consolidation",
        "application_type": "Individual",
    }}}


class Factor(BaseModel):
    feature: str
    label: str
    value: Optional[float]
    shap: float


class Prediction(BaseModel):
    default_probability: float
    risk_band: Literal["low", "medium", "high", "very_high"]
    recommendation: Literal["approve", "decline"]
    threshold: float
    risk_drivers: list[Factor]
    protective_factors: list[Factor]
    model_version: str


class BatchRequest(BaseModel):
    applications: list[LoanApplication] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    count: int
    approval_rate: float
    mean_default_probability: float
    predictions: list[Prediction]