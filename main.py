# CPSC 481 - Logistics Robot Decision Agent - Pipeline orchestrator
from __future__ import annotations

from typing import List, Tuple

from search import GRID, find_candidate_paths, print_grid
from ml import load_and_train


VALID_TIMES_OF_DAY = ("morning", "afternoon", "evening", "night")


def bucket_distance(steps: int) -> str:
    # short <= 6, medium 7-10, long > 10
    if steps <= 6:
        return "short"
    if steps <= 10:
        return "medium"
    return "long"


def bucket_zone_type(route: dict) -> str:
    # restricted dominates high_traffic dominates normal
    if route["passes_restricted"]:
        return "restricted"
    if route["passes_high_traffic"]:
        return "high_traffic"
    return "normal"


def bucket_congestion(route: dict) -> str:
    # 0 H -> low, 1 H -> medium, 2+ H -> high
    h = route["high_traffic_count"]
    if h >= 2:
        return "high"
    if h == 1:
        return "medium"
    return "low"


def extract_route_features(route: dict, time_of_day: str) -> Tuple[str, str, str, str]:
    return (
        time_of_day,
        bucket_zone_type(route),
        bucket_congestion(route),
        bucket_distance(route["steps"]),
    )


def prompt_time_of_day() -> str:
    options = ", ".join(VALID_TIMES_OF_DAY)
    while True:
        raw = input(f"Time of day ({options}): ").strip().lower()
        if raw in VALID_TIMES_OF_DAY:
            return raw
        print(f"  '{raw}' is not valid. Choose one of: {options}")


def evaluate_routes(routes: List[dict], time_of_day: str, predictor) -> List[dict]:
    enriched: List[dict] = []
    for route in routes:
        features = extract_route_features(route, time_of_day)
        time_, zone, cong, dist = features
        delay = predictor.predict_delay(time_, zone, cong, dist)

        new_route = dict(route)
        new_route["features"] = {
            "time_of_day": time_,
            "zone_type": zone,
            "congestion_level": cong,
            "distance": dist,
        }
        new_route["predicted_delay"] = delay
        enriched.append(new_route)
    return enriched


def print_route_evaluation(route: dict) -> None:
    f = route["features"]
    print(f"{route['id']}")
    print(f"  Path:             {route['path']}")
    print(f"  Steps:            {route['steps']}")
    print(f"  Distance:         {f['distance']}")
    print(f"  Zone type:        {f['zone_type']}")
    print(f"  Congestion level: {f['congestion_level']}")
    print(f"  Predicted delay:  {route['predicted_delay']}")
    if "decision" in route:
        print(f"  Decision:         {route['decision']}")
        if "reasons" in route:
            for r in route["reasons"]:
                print(f"    - {r}")
        if "score" in route:
            print(f"  Score:            {route['score']}")
    print()


def main() -> None:
    print("=" * 60)
    print("Logistics Robot Decision Agent")
    print("=" * 60)
    print("\nWarehouse grid:")
    print_grid(GRID)

    time_of_day = prompt_time_of_day()
    print(f"\nInput time_of_day: {time_of_day}\n")

    routes = find_candidate_paths(GRID, max_paths=3)
    if not routes:
        print("No routes found from S to G. Aborting.")
        return
    print(f"Search generated {len(routes)} candidate route(s).\n")

    predictor = load_and_train()
    enriched = evaluate_routes(routes, time_of_day, predictor)

    # TODO: wire in reasoning.py once it exposes apply_rules_and_score and select_final_route
    final = None

    print("Per-route evaluation:")
    print("-" * 60)
    for route in enriched:
        print_route_evaluation(route)

    if final is not None:
        print("=" * 60)
        print(f"Final Selected Route: {final['id']}")
        print(f"Why: {final.get('explanation', '(no explanation provided)')}")
        print("=" * 60)
    else:
        print("(Reasoning module not yet wired in - no final selection.)")


if __name__ == "__main__":
    main()