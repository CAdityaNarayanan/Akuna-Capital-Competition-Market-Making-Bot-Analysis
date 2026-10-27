"""Evaluate preselected inventory caps on a fresh synthetic sample, without tuning."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path

import stress_test_akuna as stress
from akuna_bot import AJARAI_UNDERLYING_ID
from quote_strategy_comparison import (
    MAX_CUSTOMER_QUANTITY,
    RFQS_PER_DAY,
    prepare_pricing_cache,
    run_episode,
)


ROOT = Path(__file__).resolve().parent
CAPS = (2, 3, 20)
FRACTIONS = (0.15, 0.30, 0.50)


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(episodes: int, seed: int, output_dir: Path) -> None:
    if episodes < 2:
        raise ValueError("Use at least two episodes")
    output_dir.mkdir(parents=True, exist_ok=True)
    truth = stress.make_true_parameters()
    results: dict[tuple[float, int], list[dict]] = {
        (fraction, cap): [] for fraction in FRACTIONS for cap in CAPS
    }

    for episode in range(episodes):
        history = stress.generate_history(truth, stress.TRAINING_DAYS, seed + episode)
        spot = history.values_by_underlying_id[AJARAI_UNDERLYING_ID][-1]
        option = stress.make_options(spot, episode * 100)[episode % 6]
        horizon = option.steps_until_expiry

        current = {uid: values[-1] for uid, values in history.values_by_underlying_id.items()}
        random.seed(seed + 40_000_000 + episode)
        future_states = [dict(current)]
        for _ in range(horizon):
            current = truth.advance_step(current)
            future_states.append(dict(current))
        pricing_cache = prepare_pricing_cache(history, option, future_states)

        event_rng = random.Random(seed + 50_000_000 + episode)
        customer_events = [
            [
                (event_rng.random(), event_rng.random() < 0.5,
                 event_rng.randint(1, MAX_CUSTOMER_QUANTITY), event_rng.random())
                for _ in range(RFQS_PER_DAY)
            ]
            for _ in range(horizon)
        ]

        for fraction in FRACTIONS:
            for cap in CAPS:
                result = run_episode(
                    episode, "symmetric_fair_mid", fraction, history, option,
                    future_states, customer_events, cap, pricing_cache,
                )
                results[(fraction, cap)].append(result)

        if (episode + 1) % 50 == 0:
            print(f"Completed {episode + 1}/{episodes} fresh episodes", flush=True)

    detail_rows = []
    summary_rows = []
    paired_rows = []
    for fraction in FRACTIONS:
        by_cap = {cap: results[(fraction, cap)] for cap in CAPS}
        for cap in CAPS:
            rows = by_cap[cap]
            pnls = [row["pnl"] for row in rows]
            mean = statistics.mean(pnls)
            sd = statistics.stdev(pnls)
            summary_rows.append({
                "seed": seed,
                "episodes": episodes,
                "informed_customer_fraction": fraction,
                "inventory_cap": cap,
                "mean_pnl": mean,
                "pnl_sd": sd,
                "mean_pnl_ci95_low_normal": mean - 1.96 * sd / math.sqrt(episodes),
                "mean_pnl_ci95_high_normal": mean + 1.96 * sd / math.sqrt(episodes),
                "positive_episode_fraction": sum(p > 0 for p in pnls) / episodes,
                "mean_peak_abs_inventory": statistics.mean(r["peak_abs_inventory"] for r in rows),
                "mean_ending_abs_inventory": statistics.mean(abs(r["ending_inventory"]) for r in rows),
                "fill_rate": sum(r["fills"] for r in rows) / sum(r["quotes"] for r in rows),
            })
            detail_rows.extend({
                "seed": seed,
                "informed_customer_fraction": fraction,
                "inventory_cap": cap,
                **row,
            } for row in rows)

        reference = by_cap[2]
        for cap in (3, 20):
            diffs = [other["pnl"] - base["pnl"] for base, other in zip(reference, by_cap[cap])]
            diff_mean = statistics.mean(diffs)
            se = statistics.stdev(diffs) / math.sqrt(episodes)
            paired_rows.append({
                "seed": seed,
                "informed_customer_fraction": fraction,
                "comparison": f"cap_{cap}_minus_cap_2",
                "mean_pnl_difference": diff_mean,
                "ci95_low_normal": diff_mean - 1.96 * se,
                "ci95_high_normal": diff_mean + 1.96 * se,
                "episodes": episodes,
            })

    save_csv(output_dir / "final_cap_validation_episodes.csv", detail_rows)
    save_csv(output_dir / "final_cap_validation_summary.csv", summary_rows)
    save_csv(output_dir / "final_cap_validation_paired.csv", paired_rows)

    print("\nFixed-cap fresh-sample validation (no cap selected from these results)")
    print(f"Seed: {seed}; episodes: {episodes}; caps: {', '.join(map(str, CAPS))}")
    print("Mean PnL per episode and mean peak inventory:")
    for row in summary_rows:
        print(
            f"  informed={row['informed_customer_fraction']:.0%} cap={row['inventory_cap']:2d} "
            f"PnL=${row['mean_pnl']:+.3f} "
            f"(95% CI {row['mean_pnl_ci95_low_normal']:+.3f} to {row['mean_pnl_ci95_high_normal']:+.3f}) "
            f"positive={row['positive_episode_fraction']:.1%} "
            f"peak inventory={row['mean_peak_abs_inventory']:.2f}"
        )
    print("Paired differences versus cap 2:")
    for row in paired_rows:
        print(
            f"  informed={row['informed_customer_fraction']:.0%} {row['comparison']}: "
            f"${row['mean_pnl_difference']:+.3f} "
            f"(95% CI {row['ci95_low_normal']:+.3f} to {row['ci95_high_normal']:+.3f})"
        )
    print(f"Saved CSV results under {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20304000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "trading_results")
    args = parser.parse_args()
    run(args.episodes, args.seed, args.output_dir)


if __name__ == "__main__":
    main()

