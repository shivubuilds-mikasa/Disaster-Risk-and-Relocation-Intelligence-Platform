"""Validated contracts for the interactive PRISM workflow."""
from typing import Literal
from pydantic import BaseModel, Field


class ScenarioControls(BaseModel):
    scenario_type: Literal["Normal", "Heavy Rainfall", "Extreme Rainfall", "Landslide Scenario", "Flood Scenario", "Combined Disaster Scenario"] = "Heavy Rainfall"
    rainfall_24h_mm: float = Field(default=96, ge=0, le=300)
    duration_hours: float = Field(default=24, ge=1, le=72)
    vegetation_loss: float = Field(default=0.22, ge=0, le=1)
    slope_multiplier: float = Field(default=1, ge=0.5, le=2)
    severity: float = Field(default=1, ge=0.5, le=2)


class CopilotRequest(BaseModel):
    question: str = Field(default="Give an emergency response summary.", min_length=1, max_length=800)
    controls: ScenarioControls | None = None


class ReportRequest(BaseModel):
    controls: ScenarioControls | None = None
