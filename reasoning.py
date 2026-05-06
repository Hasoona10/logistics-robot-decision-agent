"""
CPSC 481 - Logistics Robot Decision Agent - Reasoning Module
Combines ML-predicted delay with symbolic rules over each candidate route to
produce a numeric score, a categorical decision, a list of human-readable
reasons, and a final route selection with explanation.

Public API expected by main.py:
    apply_rules_and_score(route) -> mutates route, adds 'decision',
                                    'reasons', and 'score'
    select_final_route(routes)   -> returns the chosen route with an
                                    'explanation' field added
"""
from __future__ import annotations

from typing import Any, Dict, List


# --------------------------------------------------------------------------- #
# Tunable weights
# Score starts at BASE_SCORE and penalties are subtracted. Highest score wins.
# --------------------------------------------------------------------------- #
BASE_SCORE = 100

# Penalty applied based on the ML decision tree's predicted delay level.
DELAY_PENALTY = {
    "low":    0,
    "medium": 15,
    "high":   35,
}

# Symbolic / rule-based penalties applied on top of the ML signal.
RESTRICTED_ZONE_PENALTY   = 25   # per restricted cell crossed
HIGH_TRAFFIC_ZONE_PENALTY = 8    # per high-traffic cell crossed
STEP_PENALTY              = 1    # per grid step (distance proxy)

# Time-of-day modifier: at night, restricted zones are extra risky because
# of low visibility / reduced supervision.
NIGHT_RESTRICTED_EXTRA = 10

# Time-of-day modifier: in the evening, high-traffic cells are worse because
# of shift changes and end-of-day surge.
EVENING_HIGH_TRAFFIC_EXTRA = 12

# Bonus for the ideal case: ML predicts low delay AND the route stays in
# normal zones the whole way.
SAFE_ROUTE_BONUS = 10


# --------------------------------------------------------------------------- #
# Decision categories (printed in the per-route report)
# --------------------------------------------------------------------------- #
DECISION_RECOMMEND  = "Recommend"
DECISION_ACCEPTABLE = "Acceptable"
DECISION_AVOID      = "Avoid"


def _classify(score: int) -> str:
    """Map a numeric score to a categorical decision label."""
    if score >= 80:
        return DECISION_RECOMMEND
    if score >= 55:
        return DECISION_ACCEPTABLE
    return DECISION_AVOID


def apply_rules_and_score(route: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score one candidate route by combining ML's predicted_delay with
    symbolic rules over the route features. Mutates and returns `route`.
    """
    features        = route["features"]
    predicted_delay = route["predicted_delay"]
    time_of_day     = features["time_of_day"]

    score: int = BASE_SCORE
    reasons: List[str] = []

    # 1) ML signal
    delay_pen = DELAY_PENALTY.get(predicted_delay, 0)
    if delay_pen > 0:
        score -= delay_pen
        reasons.append(f"ML predicts {predicted_delay} delay (-{delay_pen})")
    else:
        reasons.append(f"ML predicts {predicted_delay} delay (no penalty)")

    # 2) Restricted zones
    if route["passes_restricted"]:
        rcount = route["restricted_count"]
        pen = rcount * RESTRICTED_ZONE_PENALTY
        score -= pen
        reasons.append(f"Crosses {rcount} restricted cell(s) (-{pen})")
        if time_of_day == "night":
            score -= NIGHT_RESTRICTED_EXTRA
            reasons.append(
                f"Night-time restricted-zone risk (-{NIGHT_RESTRICTED_EXTRA})"
            )
    else:
        reasons.append("Avoids restricted zones")

    # 3) High-traffic zones
    if route["passes_high_traffic"]:
        hcount = route["high_traffic_count"]
        pen = hcount * HIGH_TRAFFIC_ZONE_PENALTY
        score -= pen
        reasons.append(f"Crosses {hcount} high-traffic cell(s) (-{pen})")
        if time_of_day == "evening":
            score -= EVENING_HIGH_TRAFFIC_EXTRA
            reasons.append(
                f"Evening high-traffic surge (-{EVENING_HIGH_TRAFFIC_EXTRA})"
            )
    else:
        reasons.append("Avoids high-traffic zones")

    # 4) Distance
    step_pen = route["steps"] * STEP_PENALTY
    score -= step_pen
    reasons.append(f"{route['steps']} steps long (-{step_pen})")

    # 5) Bonus: ML predicts low delay AND route stays in normal zones.
    if (predicted_delay == "low"
            and features["zone_type"] == "normal"):
        score += SAFE_ROUTE_BONUS
        reasons.append(
            f"Safe-route bonus: low delay + normal zone (+{SAFE_ROUTE_BONUS})"
        )

    # 6) Hard reject: high predicted delay on a long route is non-viable.
    hard_reject = (predicted_delay == "high"
                   and features["distance"] == "long")
    if hard_reject:
        reasons.append("REJECTED: high delay on long route")

    # 7) Final categorical decision
    score = max(score, 0)
    route["score"]    = score
    route["reasons"]  = reasons
    route["decision"] = DECISION_AVOID if hard_reject else _classify(score)
    return route


def select_final_route(routes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pick the highest-scoring route. On ties, prefer fewer steps, then lower
    predicted delay severity, then earlier id (deterministic). Adds an
    `explanation` field describing why this route won.
    """
    if not routes:
        raise ValueError("select_final_route called with no routes.")

    delay_rank = {"low": 0, "medium": 1, "high": 2}
    decision_rank = {
        DECISION_RECOMMEND:  0,
        DECISION_ACCEPTABLE: 1,
        DECISION_AVOID:      2,
    }

    def sort_key(r: Dict[str, Any]):
        return (
            decision_rank.get(r["decision"], 3),
            -r["score"],
            r["steps"],
            delay_rank.get(r["predicted_delay"], 3),
            r["id"],
        )

    ranked = sorted(routes, key=sort_key)
    best   = ranked[0]

    parts = [
        f"{best['id']} chosen with score {best['score']} ({best['decision']}).",
        f"Predicted delay = {best['predicted_delay']},",
        f"steps = {best['steps']},",
        f"zone = {best['features']['zone_type']}.",
    ]
    if len(ranked) > 1:
        runner_up = ranked[1]
        parts.append(
            f"Beat next-best {runner_up['id']} (score {runner_up['score']})."
        )
    best["explanation"] = " ".join(parts)
    return best


# --------------------------------------------------------------------------- #
# Manual smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    demo_routes = [
        {
            "id": "Route A",
            "path": [(0, 0), (0, 1), (1, 1)],
            "steps": 5,
            "passes_restricted": False,
            "passes_high_traffic": True,
            "high_traffic_count": 1,
            "restricted_count": 0,
            "features": {
                "time_of_day": "afternoon",
                "zone_type": "high_traffic",
                "congestion_level": "medium",
                "distance": "short",
            },
            "predicted_delay": "medium",
        },
        {
            "id": "Route B",
            "path": [(0, 0), (1, 0), (2, 0)],
            "steps": 8,
            "passes_restricted": True,
            "passes_high_traffic": False,
            "high_traffic_count": 0,
            "restricted_count": 2,
            "features": {
                "time_of_day": "afternoon",
                "zone_type": "restricted",
                "congestion_level": "low",
                "distance": "medium",
            },
            "predicted_delay": "high",
        },
    ]
    for r in demo_routes:
        apply_rules_and_score(r)
        print(r["id"], r["score"], r["decision"])
        for reason in r["reasons"]:
            print("  -", reason)
    chosen = select_final_route(demo_routes)
    print("\nFinal:", chosen["id"])
    print(chosen["explanation"])