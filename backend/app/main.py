"""
Main FastAPI Application – Wayanad Disaster Risk & Relocation Intelligence Platform.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import engine, Base
from app.routes import dashboard, settlements, sites, scenarios, reports, model, live, platform

# Ensure DB tables exist
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.APP_TITLE,
    description=(
        "Smart India Hackathon Prototype: Decision-support backend for high-stakes disaster-risk "
        "assessment and transparent resettlement planning for the Wayanad 2024 landslide zone."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for React / Vite frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(dashboard.router)
app.include_router(settlements.router)
app.include_router(sites.router)
app.include_router(scenarios.router)
app.include_router(reports.router)
app.include_router(model.router)
app.include_router(live.router)
app.include_router(platform.router)


@app.get("/")
def root():
    return {
        "status": "online",
        "project": settings.APP_TITLE,
        "district": "Wayanad, Kerala",
        "disaster_case": "July 2024 Wayanad Landslide Resettlement Intelligence",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "version": "1.0.0",
        "core_workflow": [
            "GET /api/dashboard",
            "GET /api/risk-map",
            "GET /api/settlements",
            "GET /api/settlements/{id}",
            "GET /api/settlements/{id}/why",
            "GET /api/settlements/{id}/sites",
            "GET /api/sites/{id}",
            "POST /api/sites/{id}/revalidate",
            "POST /api/sites/compare",
            "POST /api/scenario",
            "GET /api/recommendation/{id}",
            "GET /api/report/{id}",
            "GET /api/geojson/wayanad",
            "GET /api/model/validation",
            "GET /api/live/weather",
            "GET /api/live/alerts",
            "GET /api/live/risk-map",
            "GET /api/live/status"
        ]
    }
