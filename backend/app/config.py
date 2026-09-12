"""
Application configuration. All tunable constants live here.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./wayanad.db"
    APP_TITLE: str = "Wayanad Disaster Risk & Relocation Intelligence API"
    DEBUG: bool = True

    # ── Risk-level thresholds (configurable in one place) ──────────
    RISK_CRITICAL: int = 80
    RISK_HIGH: int = 60
    RISK_MODERATE: int = 40

    # ── Risk engine weights ────────────────────────────────────────
    WEIGHT_HAZARD: float = 0.40
    WEIGHT_EXPOSURE: float = 0.25
    WEIGHT_VULNERABILITY: float = 0.20
    WEIGHT_TREND: float = 0.15

    # ── Destination revalidation weights ───────────────────────────
    DEST_WEIGHT_SAFETY: float = 0.40
    DEST_WEIGHT_FUTURE_RISK: float = 0.20
    DEST_WEIGHT_ACCESSIBILITY: float = 0.15
    DEST_WEIGHT_CAPACITY: float = 0.10
    DEST_WEIGHT_INFRASTRUCTURE: float = 0.10
    DEST_WEIGHT_COST: float = 0.05

    # ── Destination validation thresholds ──────────────────────────
    DEST_PASSED: int = 80
    DEST_CONDITIONAL: int = 60

    # ── Scenario bounds (expanded for Round-2 multi-variable engine) ──────
    SCENARIO_MIN_RAINFALL: int = -20
    SCENARIO_MAX_RAINFALL: int = 50

    # ── Hard constraint thresholds (Tier-1 disqualification) ───────────────
    HARD_CONSTRAINT_MAX_SLOPE: float = 12.0        # degrees — reject sites steeper than this
    HARD_CONSTRAINT_MIN_SCARP_BUFFER: float = 2.0  # km — reject sites closer to active scarp
    HARD_CONSTRAINT_MAX_HAZARD_SCORE: float = 30.0 # 0-100 — reject sites with hazard above this

    # ── CORS origins ───────────────────────────────────────────────
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # ── Live / near-real-time monitoring ────────────────────────────────
    # Which weather data source to use: "imd" (official IMD API), "weatherapi"
    # (WeatherAPI.com), or "demo" (simulated, default).
    # Secrets always come from environment variables / .env — never hard-coded.
    WEATHER_PROVIDER: str = "demo"
    IMD_API_KEY: str = ""
    IMD_API_BASE_URL: str = ""
    # WeatherAPI.com adapter (the key family used by the demo frontend) — lets the
    # real observation flow through this gateway -> risk engine -> GeoJSON.
    WEATHER_API_KEY: str = ""
    WEATHER_API_QUERY: str = "11.60,76.04"  # Kalpetta / Wayanad (lat,lng)

    # Optional OpenAI-compatible service. These values are consumed only by the
    # backend; a missing key intentionally selects the local deterministic mode.
    AI_PROVIDER: str = "local"
    AI_API_KEY: str = ""
    AI_MODEL: str = "gpt-4o-mini"
    AI_BASE_URL: str = "https://api.openai.com/v1"

    # Automatic refresh (seconds): external-fetch TTL on the backend AND the
    # cadence advertised to the frontend polling loop. One external call per window.
    LIVE_REFRESH_INTERVAL_SECONDS: int = 300
    # A feed is considered stale after this many minutes without a successful update.
    LIVE_STALE_AFTER_MINUTES: int = 15
    # When an external feed fails, fall back to the last good observation / demo.
    LIVE_FALLBACK_TO_DEMO: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
