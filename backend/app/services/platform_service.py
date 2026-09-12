"""Transparent, deterministic scenario and relocation workflow for the demo API.

This module deliberately treats its inputs as *synthetic demonstration data* and
does not make operational recommendations. AI-provider behavior lives separately.
"""
from __future__ import annotations

from math import hypot
from typing import Any


def _level(value: float) -> str:
    return "CRITICAL" if value >= 80 else "HIGH" if value >= 60 else "MEDIUM" if value >= 40 else "LOW"


def _site_score(site: Any) -> float:
    # Safety and service access matter more than capacity. Kept visible here so
    # the score can be inspected, reproduced, and challenged.
    return round(
        site.safety_score * .32 + (100 - site.future_risk) * .18 +
        site.accessibility_score * .18 + site.infrastructure_score * .17 +
        min(100, site.capacity / 20) * .15, 1
    )


def run_workflow(settlements: list[Any], sites: list[Any], controls: dict[str, Any] | None = None) -> dict[str, Any]:
    controls = controls or {}
    rainfall = max(0, min(300, float(controls.get("rainfall_24h_mm", 96))))
    duration = max(1, min(72, float(controls.get("duration_hours", 24))))
    vegetation_loss = max(0, min(1, float(controls.get("vegetation_loss", .22))))
    slope_multiplier = max(.5, min(2, float(controls.get("slope_multiplier", 1))))
    severity = max(.5, min(2, float(controls.get("severity", 1))))
    scenario_type = str(controls.get("scenario_type", "Heavy Rainfall"))
    rain_pressure = min(100, (rainfall / 2.2 + duration * .45) * severity)

    affected = []
    for s in settlements:
        hazard = min(100, (s.landslide_risk * .42 + s.rainfall_risk * .33 + s.slope_risk * .25 * slope_multiplier) + rain_pressure * .22 + vegetation_loss * 18)
        exposure = min(100, 22 + s.population / 45)
        vulnerability = min(100, s.housing_vulnerability * .7 + (s.priority_households / max(1, s.households)) * 30)
        risk = round(min(100, hazard * .55 + exposure * .2 + vulnerability * .25), 1)
        if risk >= 60:
            affected.append({"id": s.id, "name": s.name, "latitude": s.latitude, "longitude": s.longitude, "population": s.population, "households": s.households, "vulnerable_people": round(s.population * min(.45, .12 + vulnerability / 260)), "hazard_score": round(hazard, 1), "exposure_score": round(exposure, 1), "vulnerability_score": round(vulnerability, 1), "risk_score": risk, "risk_level": _level(risk)})
    affected.sort(key=lambda x: x["risk_score"], reverse=True)

    # Source data defines destination capacity in households, so allocation is
    # performed in households and only then converted back to people for the UI.
    inventories = [{"id": x.id, "name": x.name, "latitude": x.latitude, "longitude": x.longitude, "capacity_households": x.capacity, "available_households": x.capacity, "suitability_score": _site_score(x), "status": x.status, "accessibility": x.accessibility_score, "safety": x.safety_score} for x in sites if x.status != "REJECTED"]
    inventories.sort(key=lambda x: x["suitability_score"], reverse=True)
    assignments, routes = [], []
    for origin in affected:
        remaining_households = origin["households"]
        people_per_household = origin["population"] / max(1, origin["households"])
        for dest in inventories:
            if remaining_households <= 0 or dest["available_households"] <= 0:
                continue
            moved_households = min(remaining_households, dest["available_households"])
            moved = round(moved_households * people_per_household)
            distance = round(hypot(origin["latitude"] - dest["latitude"], origin["longitude"] - dest["longitude"]) * 111, 1)
            dest["available_households"] -= moved_households
            remaining_households -= moved_households
            assignments.append({"settlement_id": origin["id"], "settlement": origin["name"], "site_id": dest["id"], "site": dest["name"], "people": moved, "households": moved_households, "distance_km": distance})
            routes.append({"from": [origin["latitude"], origin["longitude"]], "to": [dest["latitude"], dest["longitude"]], "people": moved})
        origin["unmet_households"] = remaining_households
        origin["unmet_population"] = round(remaining_households * people_per_household)
    total_affected = sum(x["population"] for x in affected)
    relocated = sum(x["people"] for x in assignments)
    infrastructure_exposed = min(24, round(sum(x["hazard_score"] for x in affected) / 28))
    return {"disclaimer": "Synthetic/demo inputs and simulated outputs only. Human emergency authorities must validate every decision.", "scenario": {"type": scenario_type, "rainfall_24h_mm": rainfall, "duration_hours": duration, "vegetation_loss": vegetation_loss, "slope_multiplier": slope_multiplier, "severity": severity}, "affected_settlements": affected, "red_zones": [{"settlement_id": x["id"], "name": x["name"], "center": [x["latitude"], x["longitude"]], "radius_m": round(450 + x["hazard_score"] * 18), "intensity": x["hazard_score"]} for x in affected if x["risk_score"] >= 70], "sites": inventories, "assignments": assignments, "routes": routes, "metrics": {"settlements_assessed": len(settlements), "affected_population": total_affected, "affected_households": sum(x["households"] for x in affected), "vulnerable_population": round(sum(x["vulnerable_people"] for x in affected)), "relocated_population": relocated, "relocated_households": sum(x["households"] for x in assignments), "unmet_population": sum(x["unmet_population"] for x in affected), "unmet_households": sum(x["unmet_households"] for x in affected), "critical_zones": sum(1 for x in affected if x["risk_level"] == "CRITICAL"), "infrastructure_exposed": infrastructure_exposed, "available_household_capacity": sum(x["available_households"] for x in inventories), "average_route_km": round(sum(x["distance_km"] * x["people"] for x in assignments) / max(1, relocated), 1)}}


def local_copilot(question: str, workflow: dict[str, Any]) -> dict[str, str]:
    q = question.lower()
    m = workflow["metrics"]
    priority = workflow["affected_settlements"][0]["name"] if workflow["affected_settlements"] else "No settlement"
    if "bottleneck" in q or "unmet" in q:
        answer = f"The current capacity-aware allocation leaves {m['unmet_population']:,} people unassigned. Add verified capacity before treating this as an evacuation plan."
    elif "why" in q or "risk" in q:
        item = workflow["affected_settlements"][0] if workflow["affected_settlements"] else None
        answer = f"{priority} is prioritized because its simulated hazard score is {item['hazard_score']:.1f}, combined with exposure and vulnerability. These are synthetic scenario outputs, not a field warning." if item else "No settlement crosses the active scenario threshold."
    else:
        answer = f"{m['affected_population']:,} people are affected in this simulated scenario; {m['relocated_population']:,} fit the available verified-demo capacity. Priority review: {priority}."
    return {"mode": "local_fallback", "answer": answer, "grounding": "Response generated only from the current workflow payload; no external model or undisclosed data was used."}
