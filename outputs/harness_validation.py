from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


# Report assumptions. These are illustrative campaign parameters, not observed costs.
INCENTIVE_COST_DELTA = 5.0
CONTACT_COST_KAPPA = 0.5
RETENTION_RATE_GAMMA = 0.25
CONFIDENCE_REVIEW_BAND = (0.45, 0.55)
RATE_LIMIT_DAYS = 30


@dataclass
class CustomerCase:
    customer_id: str
    churn_probability: float
    clv_proxy: float
    last_campaign_date: datetime | None = None


def neslin_expected_profit_per_customer(
    churn_probability: float,
    clv_proxy: float,
    incentive_cost: float = INCENTIVE_COST_DELTA,
    contact_cost: float = CONTACT_COST_KAPPA,
    retention_rate: float = RETENTION_RATE_GAMMA,
) -> float:
    """Per-customer version of Neslin-style campaign profit for a targeted customer."""
    beta = churn_probability
    return (
        beta * retention_rate * (clv_proxy - incentive_cost - contact_cost)
        + beta * (1.0 - retention_rate) * (-contact_cost)
        + (1.0 - beta) * (-incentive_cost - contact_cost)
    )


def decide_campaign_action(
    case: CustomerCase,
    now: datetime,
    rate_limit_days: int = RATE_LIMIT_DAYS,
) -> dict[str, Any]:
    audit_log: list[str] = []
    p = float(case.churn_probability)
    audit_log.append(f"input_score={p:.3f}, clv_proxy={case.clv_proxy:.2f}")

    lo, hi = CONFIDENCE_REVIEW_BAND
    if lo <= p <= hi:
        audit_log.append(f"confidence_gate=human_review because score in [{lo}, {hi}]")
        return {"customer_id": case.customer_id, "action": "human_review", "audit_log": audit_log}

    if case.last_campaign_date is not None:
        days_since = (now - case.last_campaign_date).days
        if days_since < rate_limit_days:
            audit_log.append(f"rate_limit=blocked because last campaign was {days_since} days ago")
            return {"customer_id": case.customer_id, "action": "block_rate_limited", "audit_log": audit_log}
        audit_log.append(f"rate_limit=passed because last campaign was {days_since} days ago")

    expected_profit = neslin_expected_profit_per_customer(p, case.clv_proxy)
    audit_log.append(f"expected_profit={expected_profit:.2f}")

    if expected_profit <= INCENTIVE_COST_DELTA:
        audit_log.append("profit_gate=blocked because expected profit does not exceed incentive cost")
        return {"customer_id": case.customer_id, "action": "block_low_profit", "audit_log": audit_log}

    if p >= hi:
        audit_log.append("campaign_gate=send because score is high and profit gate passed")
        return {"customer_id": case.customer_id, "action": "send_campaign", "audit_log": audit_log}

    audit_log.append("campaign_gate=no_action because churn risk is below action threshold")
    return {"customer_id": case.customer_id, "action": "no_action", "audit_log": audit_log}


def demo_cases(now: datetime | None = None) -> list[dict[str, Any]]:
    now = now or datetime(2011, 12, 10)
    cases = [
        CustomerCase("C001", 0.91, 420.0, None),
        CustomerCase("C002", 0.51, 300.0, None),
        CustomerCase("C003", 0.88, 24.0, None),
        CustomerCase("C004", 0.77, 280.0, now - timedelta(days=8)),
        CustomerCase("C005", 0.32, 900.0, None),
        CustomerCase("C006", 0.66, 160.0, now - timedelta(days=45)),
        CustomerCase("C007", 0.47, 80.0, now - timedelta(days=60)),
        CustomerCase("C008", 0.95, 75.0, now - timedelta(days=31)),
    ]
    return [decide_campaign_action(case, now) for case in cases]


if __name__ == "__main__":
    for decision in demo_cases():
        print(decision)
