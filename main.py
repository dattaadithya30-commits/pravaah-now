"""
PRAVAAH-NOW backend — FastAPI service.

Serves nowcast predictions and the CAP/SACHET alert-dispatch endpoint
consumed by the React dashboard (see ../frontend).

Run: uvicorn backend.main:app --reload --port 8000

STATUS (matches README.md / PPT):
  - /nowcast currently returns MOCK data shaped exactly like what the
    trained model will eventually output, so the frontend can be
    developed/demoed against a stable contract before training
    finishes. Swap `generate_mock_forecast()` for a real
    `model.py` inference call once weights are trained.
  - /dispatch-alert SIMULATES a SACHET gateway response (generates a
    correctly-formatted CAP-XML payload, does not hit a live NDMA
    endpoint) — this matches the "CAP alert generation & dispatch
    flow fully functional; SACHET gateway connection simulated" note
    on our final slide.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="PRAVAAH-NOW API", version="0.1.0")

# Allow the React dev server (and your deployed frontend) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend origin before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


WARDS = [
    {"name": "Sinhgad Catchment", "pop": 41_000, "dist_km": 1.6},
    {"name": "Khadakwasla Dam Reach", "pop": 58_000, "dist_km": 3.9},
    {"name": "Dhayari-Narhe", "pop": 132_000, "dist_km": 7.8},
    {"name": "Warje-Malwadi", "pop": 187_000, "dist_km": 10.8},
    {"name": "Kothrud", "pop": 212_000, "dist_km": 13.6},
    {"name": "Katraj-Ambegaon", "pop": 164_000, "dist_km": 12.6},
    {"name": "Shivajinagar", "pop": 96_000, "dist_km": 17.6},
    {"name": "Hadapsar", "pop": 241_000, "dist_km": 20.9},
]


class NowcastRequest(BaseModel):
    lead_time_minutes: int  # 0, 15, 30, 60, 120, 240, 360


class DispatchRequest(BaseModel):
    cell_id: str
    lead_time_minutes: int
    peak_dbz: float
    rain_rate_mm_hr: float


def generate_mock_forecast(lead_time_minutes: int) -> dict:
    """
    Produces a forecast payload shaped exactly like real model output
    will be. Storm intensifies slightly with lead time up to +1h, then
    decays — a simple placeholder curve, NOT a real prediction.
    """
    # simple bell-ish curve peaking around +60min, purely for demo purposes
    t = lead_time_minutes
    intensity_factor = max(0.0, 1 - abs(t - 60) / 300)
    peak_dbz = round(50 + 10 * intensity_factor + random.uniform(-1, 1), 1)
    rain_rate = round(60 + 70 * intensity_factor + random.uniform(-3, 3), 1)

    wards_with_eta = []
    for w in WARDS:
        eta_min = max(0, int(w["dist_km"] * 4.3 - t))  # toy ETA model, not physical
        wards_with_eta.append(
            {
                **w,
                "peak_dbz": round(peak_dbz - w["dist_km"] * 0.15, 1),
                "eta_minutes": eta_min,
                "status": "IMPACT NOW" if eta_min <= 5 else ("CLEARED" if eta_min > 240 else f"ETA {eta_min} min"),
            }
        )

    return {
        "valid_at_minutes": t,
        "echo_top_km": round(14.5 + 1.5 * intensity_factor, 1),
        "vil_kg_m2": round(55 + 15 * intensity_factor, 1),
        "peak_reflectivity_dbz": peak_dbz,
        "instant_rain_rate_mm_hr": rain_rate,
        "cloudburst_criterion_met": rain_rate >= 100,
        "cell_speed_kmh": round(11 + random.uniform(-2, 2), 1),
        "heading_deg": 54,
        "shear_ms": round(37 + random.uniform(-2, 2), 1),
        "wards": wards_with_eta,
        "model_status": "MOCK — swap for real model.py inference once trained",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_status": "mock (see README for training status)"}


@app.post("/nowcast")
def nowcast(req: NowcastRequest):
    """Returns the forecast state for a given lead time (drives the timeline scrubber)."""
    return generate_mock_forecast(req.lead_time_minutes)


@app.post("/dispatch-alert")
def dispatch_alert(req: DispatchRequest):
    """
    Generates a real, spec-compliant CAP (Common Alerting Protocol)
    XML payload and returns a SIMULATED dispatch confirmation.

    This does NOT connect to the live NDMA SACHET gateway — see
    README.md "What's live vs. designed-for" for why, and say this
    plainly if a judge asks.
    """
    identifier = f"PRAVAAH-{uuid4().hex[:8].upper()}"
    sent_time = datetime.now(timezone.utc).isoformat()

    cap_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>{identifier}</identifier>
  <sender>pravaah-now@imd.gov.in</sender>
  <sent>{sent_time}</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>en-IN</language>
    <category>Met</category>
    <event>Flash Inundation &amp; Cloudburst Threat</event>
    <urgency>Immediate</urgency>
    <severity>Severe</severity>
    <certainty>Likely</certainty>
    <effective>{sent_time}</effective>
    <headline>Ward Alert: Cell {req.cell_id} — {req.peak_dbz} dBZ, {req.rain_rate_mm_hr} mm/hr</headline>
  </info>
</alert>"""

    return {
        "dispatch_id": identifier,
        "gateway": "SACHET (SIMULATED — not a live connection)",
        "dispatched_at": sent_time,
        "cap_xml": cap_xml,
        "note": "CAP payload is real/spec-compliant. Gateway transmission is simulated; "
        "live SACHET integration requires NDMA authorization (out of hackathon scope).",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
