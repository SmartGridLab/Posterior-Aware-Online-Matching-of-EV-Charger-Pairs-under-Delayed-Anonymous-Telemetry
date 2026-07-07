# -*- coding: utf-8 -*-
"""Evidence-informed cloud ingestion baseline for EV telemetry."""

from __future__ import annotations

from typing import Any, Dict


INGESTION_PROFILE_ID = "central_console_nominal_v1"
INGESTION_PROFILE_LABEL = "Central Console Nominal (Evidence-Informed)"

# Evidence-informed default (not site-calibrated):
# - Mean/Jitter are set to a low-seconds regime for online operation.
# - Max delay is capped at 10s to reflect practical message timeout behavior.
# - Loss is set to 0.5% as a conservative operational baseline.
DEFAULT_EV_INGEST_DELAY_MEAN_S = 2.0
DEFAULT_EV_INGEST_DELAY_JITTER_S = 1.0
DEFAULT_EV_INGEST_DELAY_MAX_S = 10.0
DEFAULT_EV_INGEST_LOSS_PROB = 0.005


INGESTION_SOURCES: list[dict[str, str]] = [
    {
        "id": "rfc6455",
        "title": "IETF RFC 6455 (WebSocket Protocol)",
        "url": "https://datatracker.ietf.org/doc/html/rfc6455",
        "evidence": "WebSocket protocol is layered over TCP, so transport delivery is reliable and ordered.",
    },
    {
        "id": "rfc6298",
        "title": "IETF RFC 6298 (TCP RTO)",
        "url": "https://datatracker.ietf.org/doc/html/rfc6298",
        "evidence": "Initial retransmission timeout lower bound is 1 second; Internet RTT is typically below 1 second.",
    },
    {
        "id": "fcc_mba_13th",
        "title": "FCC Measuring Broadband America, 13th Report",
        "url": "https://docs.fcc.gov/public/attachments/FCC-24-27A1.pdf",
        "evidence": "Reported latency spans tens of milliseconds, while packet-loss quality thresholds are around sub-1%.",
    },
    {
        "id": "zaptec_ocpp_keys",
        "title": "Zaptec OCPP Key-Value Settings",
        "url": "https://docs.zaptec.com/docs/zaptec-go2-ocpp-16j-supported-configuration-keys",
        "evidence": "Operational keys include MessageTimeout=10s and retry controls, supporting a practical 10s ingest cap.",
    },
    {
        "id": "abb_ocpp_setup",
        "title": "ABB Terra AC OCPP Configuration Guide",
        "url": "https://forum.iobroker.net/assets/uploads/files/1658735175644-9akk107992a1326_terraac-chargepointconfigurationocpp.pdf",
        "evidence": "Meter value sampling interval is configurable and commonly set to 30s in OCPP deployments.",
    },
]


def default_ingestion_values() -> Dict[str, float]:
    return {
        "ev_ingest_delay_mean_s": float(DEFAULT_EV_INGEST_DELAY_MEAN_S),
        "ev_ingest_delay_jitter_s": float(DEFAULT_EV_INGEST_DELAY_JITTER_S),
        "ev_ingest_delay_max_s": float(DEFAULT_EV_INGEST_DELAY_MAX_S),
        "ev_ingest_loss_prob": float(DEFAULT_EV_INGEST_LOSS_PROB),
    }


def ingestion_reference_payload() -> Dict[str, Any]:
    return {
        "profile_id": str(INGESTION_PROFILE_ID),
        "profile_label": str(INGESTION_PROFILE_LABEL),
        "model_type": "evidence_informed_baseline",
        "defaults": default_ingestion_values(),
        "sources": [dict(s) for s in INGESTION_SOURCES],
        "notes": [
            "This baseline is evidence-informed and intended for realistic online-operation emulation.",
            "It is not field-calibrated to one specific site; site logs should be used for final calibration.",
            "EVSE physical current dynamics (15s first-step, 5s subsequent-step, ramp/offset model) remain unchanged.",
        ],
    }

