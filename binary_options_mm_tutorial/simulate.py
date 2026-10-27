"""A transparent toy experiment for binary-option market making."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


SPREAD = 0.04                 # Each quote is 4 cents from estimated fair value.
INVENTORY_LIMIT = 10          # Maximum long or short position in one contract.
INVENTORY_SKEW = 0.01         # Quote center moves 1 cent per contract held.
INFORMED_CUSTOMER_RATE = 0.25
NOISE_CUSTOMER_TRADE_RATE = 0.20
ROUNDS_PER_CONTRACT = 300


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def make_episode(seed: int) -> tuple[float, int, list[tuple[bool, bool, str]]]:
    """Create one contract's true probability, payout, and customer flow."""
    rng = random.Random(seed)
    true_probability = rng.uniform(0.10, 0.90)
    payout = int(rng.random() < true_probability)  # 1 pays; 0 does not.
    events = []
    for _ in range(ROUNDS_PER_CONTRACT):
        informed = rng.random() < INFORMED_CUSTOMER_RATE
        noise_customer_trades = rng.random() < NOISE_CUSTOMER_TRADE_RATE
        noise_side = rng.choice(("sells_to_mm", "buys_from_mm"))
        events.append((informed, noise_customer_trades, noise_side))
    return true_probability, payout, events


def make_quote(estimated_probability: float, inventory: int, policy: str) -> tuple[float, float]:
    """Return bid and ask. Positive inventory shifts shaded quotes downward."""
    skew = INVENTORY_SKEW * inventory if policy == "inventory_aware" else 0.0
    center = clamp(estimated_probability - skew, 0.02, 0.98)
    bid = round(max(0.01, center - SPREAD), 2)
    ask = round(min(0.99, center + SPREAD), 2)
    return bid, ask


def run_episode(
    true_probability: float,
    payout: int,
    events: list[tuple[bool, bool, str]],
    estimated_probability: float,
    policy: str,
) -> dict[str, float | list[float]]:
    """Replay the same customer flow for one quoting policy."""
    cash = 0.0
    inventory = 0
    fills = 0
    peak_inventory = 0
    pnl_path = []

    for informed, noise_trades, noise_side in events:
        bid, ask = make_quote(estimated_probability, inventory, policy)

        if informed:
            # An informed customer trades only when the true value beats our quote.
            if true_probability > ask and inventory > -INVENTORY_LIMIT:
                cash += ask             # We sold one option at the ask.
                inventory -= 1
                fills += 1
            elif true_probability < bid and inventory < INVENTORY_LIMIT:
                cash -= bid             # We bought one option at the bid.
                inventory += 1
                fills += 1
        elif noise_trades:
            # A noise customer randomly chooses a side, without using true value.
            if noise_side == "sells_to_mm" and inventory < INVENTORY_LIMIT:
                cash -= bid
                inventory += 1
                fills += 1
            elif noise_side == "buys_from_mm" and inventory > -INVENTORY_LIMIT:
                cash += ask
                inventory -= 1
                fills += 1

        peak_inventory = max(peak_inventory, abs(inventory))
        # Mark open contracts at their true expected payout for evaluation.
        pnl_path.append(cash + inventory * true_probability)

    # At expiry, each held contract pays `payout` dollars (0 or 1).
    pnl = cash + inventory * payout
    return {
        "pnl": pnl,
        "fills": float(fills),
        "peak_inventory": float(peak_inventory),
        "ending_inventory": float(inventory),
        "pnl_path": pnl_path,
    }


def run_experiment(
    episodes: int, base_seed: int
) -> tuple[
    list[dict[str, float | str]],
    dict[float, dict[str, dict[str, list[float]]]],
]:
    rows: list[dict[str, float | str]] = []
    pnl_curves: dict[float, dict[str, dict[str, list[float]]]] = {}
    # Positive bias means the bot overestimates payout probability.
    for probability_bias in (-0.10, 0.0, 0.10):
        outcomes = {"fixed": [], "inventory_aware": []}
        for episode_number in range(episodes):
            true_p, payout, events = make_episode(base_seed + episode_number)
            estimate = clamp(true_p + probability_bias, 0.01, 0.99)
            for policy in outcomes:
                result = run_episode(true_p, payout, events, estimate, policy)
                outcomes[policy].append(result)

        pnl_curves[probability_bias] = {}
        for policy, results in outcomes.items():
            paths = [result["pnl_path"] for result in results]
            summary = {key: [] for key in ("mean", "p05", "p95", "example")}
            for round_number in range(ROUNDS_PER_CONTRACT):
                sample = sorted(path[round_number] for path in paths)
                summary["mean"].append(sum(sample) / episodes)
                summary["p05"].append(sample[int(0.05 * (episodes - 1))])
                summary["p95"].append(sample[int(0.95 * (episodes - 1))])
                summary["example"].append(paths[0][round_number])
            pnl_curves[probability_bias][policy] = summary

        for policy, results in outcomes.items():
            peak_values = sorted(result["peak_inventory"] for result in results)
            p95_index = min(len(peak_values) - 1, int(0.95 * len(peak_values)))
            rows.append({
                "probability_bias": probability_bias,
                "policy": policy,
                "mean_pnl": sum(r["pnl"] for r in results) / episodes,
                "mean_fills": sum(r["fills"] for r in results) / episodes,
                "p95_peak_inventory": peak_values[p95_index],
                "mean_abs_ending_inventory": sum(abs(r["ending_inventory"]) for r in results) / episodes,
            })
    return rows, pnl_curves


def write_pnl_curve_csv(
    curves: dict[float, dict[str, dict[str, list[float]]]], output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "probability_bias", "round", "fixed_mean_pnl", "fixed_p05", "fixed_p95",
            "fixed_example_pnl", "inventory_aware_mean_pnl", "inventory_aware_p05",
            "inventory_aware_p95", "inventory_aware_example_pnl",
        ])
        for bias, policies in curves.items():
            for index in range(len(policies["fixed"]["mean"])):
                writer.writerow([
                    bias, index + 1,
                    policies["fixed"]["mean"][index],
                    policies["fixed"]["p05"][index],
                    policies["fixed"]["p95"][index],
                    policies["fixed"]["example"][index],
                    policies["inventory_aware"]["mean"][index],
                    policies["inventory_aware"]["p05"][index],
                    policies["inventory_aware"]["p95"][index],
                    policies["inventory_aware"]["example"][index],
                ])


def write_pnl_chart(
    curves: dict[float, dict[str, dict[str, list[float]]]], episodes: int, output: Path
) -> None:
    """Write mean lines, 5th-95th percentile bands, and one example path per policy."""
    width, height = 920, 870
    left, right = 92, 875
    panel_height, panel_gap, top = 190, 70, 86
    colors = {"fixed": "#2563eb", "inventory_aware": "#ea580c"}
    labels = {"fixed": "Fixed quotes", "inventory_aware": "Inventory-aware quotes"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.grid{stroke:#d9dee8;stroke-width:1}.axis{stroke:#475467;stroke-width:1.2}</style>',
        '<text x="460" y="32" text-anchor="middle" font-size="21" font-weight="700">PnL by RFQ round: average and variation</text>',
        f'<text x="460" y="55" text-anchor="middle" font-size="13">Mean and 5th-95th percentile across {episodes} contracts; dashed = one example contract</text>',
    ]
    # Legend: colors distinguish policies; line style distinguishes mean/example.
    legend_items = [
        ("fixed", "solid", "Mean: fixed"),
        ("inventory_aware", "solid", "Mean: inventory-aware"),
        ("fixed", "dash", "Example: fixed"),
        ("inventory_aware", "dash", "Example: inventory-aware"),
    ]
    for index, (policy, style, label) in enumerate(legend_items):
        x = 110 + index * 195
        dash = ' stroke-dasharray="6 4"' if style == "dash" else ""
        parts.append(f'<line x1="{x}" y1="72" x2="{x + 24}" y2="72" stroke="{colors[policy]}" stroke-width="2.5"{dash}/>')
        parts.append(f'<text x="{x + 30}" y="77" font-size="12">{label}</text>')

    for panel_index, (bias, policies) in enumerate(curves.items()):
        y_top = top + panel_index * (panel_height + panel_gap)
        y_bottom = y_top + panel_height
        all_values = [value for policy_data in policies.values() for series in policy_data.values() for value in series]
        low, high = min(all_values), max(all_values)
        if high - low < 0.01:
            low -= 0.5
            high += 0.5
        padding = (high - low) * 0.08
        low -= padding
        high += padding
        parts.append(f'<text x="{left}" y="{y_top - 13}" font-size="15" font-weight="700">Probability estimate bias: {bias:+.0%}</text>')

        for tick in range(5):
            value = low + (high - low) * tick / 4
            y = y_bottom - panel_height * tick / 4
            parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>')
            parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{value:.2f}</text>')
        parts.append(f'<line class="axis" x1="{left}" y1="{y_top}" x2="{left}" y2="{y_bottom}"/>')
        parts.append(f'<line class="axis" x1="{left}" y1="{y_bottom}" x2="{right}" y2="{y_bottom}"/>')

        for round_tick in (0, 75, 150, 225, 300):
            x = left + (right - left) * round_tick / (ROUNDS_PER_CONTRACT - 1)
            parts.append(f'<line class="axis" x1="{x:.1f}" y1="{y_bottom}" x2="{x:.1f}" y2="{y_bottom + 4}"/>')
            parts.append(f'<text x="{x:.1f}" y="{y_bottom + 17}" text-anchor="middle" font-size="11">{round_tick}</text>')

        for policy in ("fixed", "inventory_aware"):
            values = policies[policy]["mean"]
            upper = policies[policy]["p95"]
            lower = policies[policy]["p05"]
            band_points = []
            for index, value in enumerate(upper):
                x = left + (right - left) * index / (len(upper) - 1)
                y = y_bottom - (value - low) / (high - low) * panel_height
                band_points.append(f"{x:.1f},{y:.1f}")
            for index in range(len(lower) - 1, -1, -1):
                x = left + (right - left) * index / (len(lower) - 1)
                y = y_bottom - (lower[index] - low) / (high - low) * panel_height
                band_points.append(f"{x:.1f},{y:.1f}")
            parts.append(
                f'<polygon fill="{colors[policy]}" fill-opacity="0.13" stroke="none" points="{" ".join(band_points)}"/>'
            )

            points = []
            for index, value in enumerate(values):
                x = left + (right - left) * index / (len(values) - 1)
                y = y_bottom - (value - low) / (high - low) * panel_height
                points.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<polyline fill="none" stroke="{colors[policy]}" stroke-width="2.5" points="{" ".join(points)}"/>')

            example_points = []
            for index, value in enumerate(policies[policy]["example"]):
                x = left + (right - left) * index / (len(values) - 1)
                y = y_bottom - (value - low) / (high - low) * panel_height
                example_points.append(f"{x:.1f},{y:.1f}")
            parts.append(
                f'<polyline fill="none" stroke="{colors[policy]}" stroke-opacity="0.75" stroke-width="1.2" stroke-dasharray="5 4" points="{" ".join(example_points)}"/>'
            )

        if panel_index == len(curves) - 1:
            parts.append(f'<text x="{(left + right) / 2:.1f}" y="{y_bottom + 43}" text-anchor="middle" font-size="13">RFQ round</text>')

    parts.append('</svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    rows, pnl_curves = run_experiment(args.episodes, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    curve_csv = args.output.with_name("pnl_curve.csv")
    chart_svg = args.output.with_name("pnl_curve.svg")
    write_pnl_curve_csv(pnl_curves, curve_csv)
    write_pnl_chart(pnl_curves, args.episodes, chart_svg)

    for row in rows:
        print(
            f"bias={row['probability_bias']:+.2f} | {row['policy']:15s} | "
            f"mean PnL={row['mean_pnl']:.3f} | mean fills={row['mean_fills']:.1f} | "
            f"95th pct peak inventory={row['p95_peak_inventory']:.0f} | "
            f"mean ending inventory={row['mean_abs_ending_inventory']:.2f}"
        )
    print(f"\nSaved results to {args.output}")
    print(f"Saved PnL curve data to {curve_csv}")
    print(f"Saved PnL chart to {chart_svg}")


if __name__ == "__main__":
    main()
