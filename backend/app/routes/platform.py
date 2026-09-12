from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.settlement import Settlement
from app.models.site import CandidateRelocationSite
from app.services.platform_service import run_workflow
from app.services.ai_service import answer
from app.schemas.platform import ScenarioControls, CopilotRequest, ReportRequest

router = APIRouter(prefix="/api/platform", tags=["Platform Workflow"])

def _workflow(db: Session, controls: ScenarioControls | None = None):
    return run_workflow(db.query(Settlement).all(), db.query(CandidateRelocationSite).all(), controls.model_dump() if controls else None)

@router.get("/snapshot")
def snapshot(db: Session = Depends(get_db)):
    return _workflow(db)

@router.post("/simulate")
def simulate(payload: ScenarioControls, db: Session = Depends(get_db)):
    return _workflow(db, payload)

@router.post("/copilot")
def copilot(payload: CopilotRequest, db: Session = Depends(get_db)):
    workflow = _workflow(db, payload.controls)
    return {**answer(payload.question, workflow), "workflow": workflow}

@router.post("/report")
def report(payload: ReportRequest, db: Session = Depends(get_db)):
    workflow = _workflow(db, payload.controls)
    m = workflow["metrics"]
    return {"title": "Wayanad simulated disaster intelligence summary", "workflow": workflow, "summary": f"Scenario {workflow['scenario']['type']}: {m['affected_population']:,} people affected, {m['relocated_population']:,} allocated, {m['unmet_population']:,} unmet.", "limitations": workflow["disclaimer"]}
