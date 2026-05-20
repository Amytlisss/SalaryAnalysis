from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class CorrelationResult(BaseModel):
    pearson: float
    spearman: float
    interpretation: str

class PartialCorrelationResult(BaseModel):
    coefficient: float
    p_value: float
    controlled_factors: List[str]

class RegressionResult(BaseModel):
    equation: str
    r_squared: float
    adj_r_squared: float
    std_error: float
    coefficients: Dict[str, float]
    p_values: Dict[str, float]

class AnalysisResponse(BaseModel):
    correlation: CorrelationResult
    partial_correlation: Optional[PartialCorrelationResult] = None
    regression: RegressionResult
    sample_size: int
    warnings: List[str]