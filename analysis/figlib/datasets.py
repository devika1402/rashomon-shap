"""
figlib.datasets: the canonical dataset registry.

This is the one source of truth. It maps each result directory to its dataset
and framework. It also holds the display order and the epsilon grid. Every
figure module imports it. No script redeclares this data.
"""
from __future__ import annotations

DATASETS = {
    "Electricity":  {"ag": "bq_electricity",  "h2o": "h2o_bq_electricity"},
    "ETTh1":        {"ag": "bq_etth1",         "h2o": "h2o_bq_etth1"},
    "ETTh2":        {"ag": "bq_etth2",         "h2o": "h2o_bq_etth2"},
    "ETTm1":        {"ag": "bq_ettm1",         "h2o": "h2o_bq_ettm1"},
    "M4 Monthly":   {"ag": "bq_m4_monthly",    "h2o": "h2o_bq_m4_monthly"},
    "Cable Demand": {"ag": "bq_cable_demand",  "h2o": "h2o_bq_cable_demand"},
}

DS_ORDER = ["Electricity", "ETTh1", "ETTh2", "ETTm1", "M4 Monthly", "Cable Demand"]

EPS_VALS = [0.02, 0.05, 0.10, 0.20, 0.30]
