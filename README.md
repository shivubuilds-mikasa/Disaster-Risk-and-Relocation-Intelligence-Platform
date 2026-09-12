# PRISM — Disaster Risk & Relocation Intelligence

PRISM is a Wayanad, Kerala decision-support demonstration: risk factors → scenario simulation → red zones → affected population → capacity-aware relocation → grounded local AI summary → report.

## Safety and data notice

All settlement attributes, capacities, ML samples, hazard zones, routes, and results are **synthetic demo data or simulated outputs**. They are not government records, satellite truth, field measurements, field-validated labels, or emergency instructions. Human authorities must validate every operational decision.

## Run

```powershell
cd backend
python -m pip install -r requirements.txt
python seed.py
uvicorn app.main:app --reload
```

In another terminal run `npm install` and `npm run dev`, then open `http://localhost:5173`.

## Key APIs

- `GET /api/platform/snapshot`
- `POST /api/platform/simulate`
- `POST /api/platform/copilot`
- `POST /api/platform/report`
- `GET /api/model/validation`

The Random Forest model is cached per process and trained on synthetic data. Its score is not calibrated or field validated.
