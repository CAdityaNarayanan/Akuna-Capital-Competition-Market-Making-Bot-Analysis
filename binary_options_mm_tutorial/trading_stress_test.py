"""Evaluate the Akuna bot's RFQ quoting path under synthetic customer flow.

This is a deliberately small market simulator. It uses the bot's real quote()
and on_trade() methods, while customer arrivals and informedness are explicit
simulation assumptions rather than recovered Akuna competition data.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import replace
from pathlib import Path

import stress_test_akuna as stress
from akuna_bot import (
    AJARAI_UNDERLYING_ID,
    BinaryOption,
    MarketMaker,
    OptionLeg,
    Underlying,
)


NOISE_CUSTOMER_FILL_PROBABILITY = 0.30
RFQS_PER_DAY = 8
MAX_CUSTOMER_QUANTITY = 3
INFORMED_EDGE_THRESHOLD = 0.01


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_pnl_chart(metrics: list[dict], output: Path) -> None:
    fractions = sorted({row["informed_customer_fraction"] for row in metrics})
    width, height = 780, 500
    left, top, plot_width, plot_height = 90, 55, 620, 340
    max_value = max(row["mean_pnl"] for row in metrics)
    y_max = max(1.0, math.ceil((max_value + 1.0) / 2.0) * 2.0)
    colors = {"raw_estimates": "#2563eb", "drift_shrunk": "#ea580c"}
    labels = {"raw_estimates": "Original estimates", "drift_shrunk": "Drift shrunk to zero"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.grid{stroke:#e1e6ef;stroke-width:1}.axis{stroke:#475467;stroke-width:1.2}</style>',
        '<text x="390" y="28" text-anchor="middle" font-size="19" font-weight="700">Synthetic RFQ market-making PnL</text>',
        '<text x="390" y="46" text-anchor="middle" font-size="12">Held-out episodes; error bars show 95% normal intervals for mean PnL</text>',
    ]
    for tick in range(0, int(y_max) + 1, 2):
        y = top + plot_height * (1 - tick / y_max)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{tick}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>')
    parts.append(f'<text x="{left + plot_width / 2}" y="{top + plot_height + 38}" text-anchor="middle" font-size="13">Informed customer share</text>')
    parts.append(f'<text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20 {top + plot_height / 2})" text-anchor="middle" font-size="13">Mean episode PnL ($)</text>')
    by_key = {(row["informed_customer_fraction"], row["policy"]): row for row in metrics}
    for fraction_index, fraction in enumerate(fractions):
        center = left + plot_width * (fraction_index + 0.5) / len(fractions)
        parts.append(f'<text x="{center:.1f}" y="{top + plot_height + 20}" text-anchor="middle" font-size="11">{fraction:.0%}</text>')
        for offset, policy in ((-12, "raw_estimates"), (12, "drift_shrunk")):
            row = by_key[(fraction, policy)]
            x = center + offset
            mean = row["mean_pnl"]
            half = 1.96 * row["pnl_sd_across_episodes"] / math.sqrt(row["heldout_episodes"])
            y = top + plot_height * (1 - mean / y_max)
            y_hi = top + plot_height * (1 - min(y_max, mean + half) / y_max)
            y_lo = top + plot_height * (1 - max(0.0, mean - half) / y_max)
            parts.append(f'<line x1="{x}" y1="{y_hi:.1f}" x2="{x}" y2="{y_lo:.1f}" stroke="{colors[policy]}" stroke-width="2"/>')
            parts.append(f'<line x1="{x - 4}" y1="{y_hi:.1f}" x2="{x + 4}" y2="{y_hi:.1f}" stroke="{colors[policy]}" stroke-width="2"/>')
            parts.append(f'<line x1="{x - 4}" y1="{y_lo:.1f}" x2="{x + 4}" y2="{y_lo:.1f}" stroke="{colors[policy]}" stroke-width="2"/>')
            parts.append(f'<circle cx="{x}" cy="{y:.1f}" r="5" fill="{colors[policy]}"/>')
    for i, policy in enumerate(("raw_estimates", "drift_shrunk")):
        x, y = 540 + i * 115, 440
        parts.append(f'<circle cx="{x}" cy="{y - 4}" r="5" fill="{colors[policy]}"/>')
        parts.append(f'<text x="{x + 10}" y="{y}" font-size="11">{labels[policy]}</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def run_episode(
    episode: int,
    seed: int,
    training_days: int,
    drift_lambda: float,
    informed_fraction: float,
    option_template: BinaryOption,
    base_history,
    future_states: list[dict[int, float]],
    customer_events: list[list[tuple[float, bool, int, float]]],
) -> dict:
    option = option_template
    starting = {
        uid: values[-1]
        for uid, values in base_history.values_by_underlying_id.items()
    }
    bot = MarketMaker(stress.current_underlyings(base_history), [option], cash_balance=100_000.0)
    bot.warm_up(stress.recent_history(base_history, training_days))
    bot.starting_capital = 100_000.0

    def apply_drift_adjustment() -> None:
        if bot.estimated_params is not None:
            bot.estimated_params = replace(
                bot.estimated_params,
                ajarai_drift=drift_lambda * bot.estimated_params.ajarai_drift,
            )
            bot._theo_std_cache.clear()
            bot._greeks_cache.clear()

    apply_drift_adjustment()
    cash = 0.0
    fills = 0
    bought = 0
    sold = 0
    quote_count = 0
    absolute_inventory_sum = 0
    peak_abs_inventory = 0
    horizon = option.steps_until_expiry

    for day in range(horizon):
        current_values = future_states[day]
        option = replace(option_template, steps_until_expiry=horizon - day)
        bot.underlying_state = [
            Underlying(name, uid, value)
            for uid, name, value in (
                (1, "FED", current_values[1]),
                (2, "AJR", current_values[2]),
                (3, "THR", current_values[3]),
            )
        ]
        bot.active_option_state = [option]
        bot._greeks_cache.clear()
        bot._theo_std_cache.clear()

        for counterparty_id, (informed_draw, is_customer_buyer, quantity, noise_draw) in enumerate(
            customer_events[day]
        ):
            # Preserve the original strategy for reproducing the earlier
            # drift-ablation experiment after quote() became symmetric.
            quote = bot.quote_regime_based(option, counterparty_id)
            quote_count += 1

            if informed_draw < informed_fraction:
                # An informed customer trades only when the true-model fair
                # probability says the bot's quote is favorable to them.
                true_probability = bot.price_option_from_parameters(
                    stress.make_true_parameters(), option
                )
                if is_customer_buyer:
                    fills_this_event = true_probability > quote.offer_price + INFORMED_EDGE_THRESHOLD
                else:
                    fills_this_event = true_probability < quote.bid_price - INFORMED_EDGE_THRESHOLD
            else:
                # Noise customers trade randomly; this parameter is a declared
                # toy-market assumption, not an estimate from Akuna data.
                fills_this_event = noise_draw < NOISE_CUSTOMER_FILL_PROBABILITY

            if not fills_this_event:
                continue

            if is_customer_buyer:
                traded_quantity = min(quantity, quote.offer_quantity)
                trade_price = quote.offer_price
                signed_quantity = -traded_quantity  # bot sells
                sold += traded_quantity
            else:
                traded_quantity = min(quantity, quote.bid_quantity)
                trade_price = quote.bid_price
                signed_quantity = traded_quantity  # bot buys
                bought += traded_quantity

            cash -= trade_price * signed_quantity
            bot.cash_balance = 100_000.0 + cash
            bot.on_trade(option, trade_price, signed_quantity, counterparty_id)
            fills += 1
            absolute_inventory_sum += abs(bot.positions[option.option_id])
            peak_abs_inventory = max(peak_abs_inventory, abs(bot.positions[option.option_id]))

        if day < horizon - 1:
            next_values = future_states[day + 1]
            next_option = replace(option_template, steps_until_expiry=horizon - day - 1)
            new_underlyings = [
                Underlying(name, uid, next_values[uid])
                for uid, name in ((1, "FED"), (2, "AJR"), (3, "THR"))
            ]
            bot.on_step_advance(new_underlyings, [next_option])
            apply_drift_adjustment()

    final_values = future_states[horizon]
    payout = option_template.expiry_valuation(final_values)
    position = bot.positions[option_template.option_id]
    pnl = cash + position * payout
    return {
        "episode": episode,
        "pnl": pnl,
        "fills": fills,
        "quote_count": quote_count,
        "buy_contracts": bought,
        "sell_contracts": sold,
        "ending_inventory": position,
        "peak_abs_inventory": peak_abs_inventory,
        "mean_abs_inventory_at_fill": absolute_inventory_sum / fills if fills else 0.0,
        "expiry_payout": payout,
    }


def run(episodes: int, seed: int, informed_fractions: list[float], output_dir: Path) -> None:
    if episodes < 4 or episodes % 2:
        raise ValueError("episodes must be an even number of at least 4")

    true_params = stress.make_true_parameters()
    split = episodes // 2
    heldout_rows = {
        fraction: {"raw_estimates": [], "drift_shrunk": []}
        for fraction in informed_fractions
    }

    for episode in range(episodes):
        history_seed = seed + episode
        history = stress.generate_history(true_params, stress.TRAINING_DAYS, history_seed)
        spot = history.values_by_underlying_id[AJARAI_UNDERLYING_ID][-1]
        options = stress.make_options(spot, episode * 100)
        option_template = options[episode % len(options)]
        horizon = option_template.steps_until_expiry

        # Pre-generate market paths and customer behavior so both policies see
        # identical exogenous conditions (common random numbers).
        current = {
            uid: values[-1]
            for uid, values in history.values_by_underlying_id.items()
        }
        random.seed(seed + 40_000_000 + episode)
        future_states = [dict(current)]
        for _ in range(horizon):
            current = true_params.advance_step(current)
            future_states.append(dict(current))

        event_rng = random.Random(seed + 50_000_000 + episode)
        customer_events = []
        for _ in range(horizon):
            day_events = []
            for _ in range(RFQS_PER_DAY):
                informed_draw = event_rng.random()
                customer_buyer = event_rng.random() < 0.5
                quantity = event_rng.randint(1, MAX_CUSTOMER_QUANTITY)
                noise_draw = event_rng.random()
                day_events.append((informed_draw, customer_buyer, quantity, noise_draw))
            customer_events.append(day_events)

        if episode < split:
            continue  # First half is reserved for selecting the adjustment.

        for fraction in informed_fractions:
            heldout_rows[fraction]["raw_estimates"].append(run_episode(
                episode, seed, 30, 1.0, fraction, option_template, history,
                future_states, customer_events
            ))
            heldout_rows[fraction]["drift_shrunk"].append(run_episode(
                episode, seed, 30, 0.0, fraction, option_template, history,
                future_states, customer_events
            ))

    metrics = []
    episode_output = []
    comparison_rows = []
    for fraction, policies in heldout_rows.items():
        for policy, rows in policies.items():
            pnls = [row["pnl"] for row in rows]
            fill_total = sum(row["fills"] for row in rows)
            quote_total = sum(row["quote_count"] for row in rows)
            metrics.append({
                "informed_customer_fraction": fraction,
                "policy": policy,
                "heldout_episodes": len(rows),
                "mean_pnl": statistics.mean(pnls),
                "pnl_sd_across_episodes": statistics.pstdev(pnls),
                "positive_pnl_episode_fraction": sum(x > 0 for x in pnls) / len(pnls),
                "fill_events": fill_total,
                "quotes": quote_total,
                "fill_rate": fill_total / quote_total if quote_total else 0.0,
                "mean_ending_inventory": statistics.mean(row["ending_inventory"] for row in rows),
                "mean_peak_abs_inventory": statistics.mean(row["peak_abs_inventory"] for row in rows),
                "mean_abs_inventory_at_fill": statistics.mean(row["mean_abs_inventory_at_fill"] for row in rows),
            })
            episode_output.extend(
                {"informed_customer_fraction": fraction, "policy": policy, **row}
                for row in rows
            )

        raw = policies["raw_estimates"]
        adjusted = policies["drift_shrunk"]
        paired_differences = [a["pnl"] - b["pnl"] for a, b in zip(raw, adjusted)]
        mean_difference = statistics.mean(paired_differences)
        standard_error = statistics.stdev(paired_differences) / math.sqrt(len(paired_differences))
        comparison_rows.append({
            "informed_customer_fraction": fraction,
            "paired_mean_pnl_raw_minus_shrunk": mean_difference,
            "paired_difference_se": standard_error,
            "ci95_low_normal": mean_difference - 1.96 * standard_error,
            "ci95_high_normal": mean_difference + 1.96 * standard_error,
            "heldout_episodes": len(paired_differences),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "trading_metrics.csv", metrics)
    write_csv(output_dir / "trading_episode_pnl.csv", episode_output)
    write_csv(output_dir / "paired_pnl_comparison.csv", comparison_rows)
    write_pnl_chart(metrics, output_dir / "trading_pnl_by_informed_flow.svg")
    print(f"Episodes: {episodes}; calibration: {split}; held out: {episodes - split}")
    print(f"Option per episode cycles through 5/10-day expiries and three nearby strikes.")
    for row in metrics:
        print(
            f"informed={row['informed_customer_fraction']:.0%} | "
            f"{row['policy']:16s} | mean PnL=${row['mean_pnl']:.3f} | "
            f"positive episodes={row['positive_pnl_episode_fraction']:.1%} | "
            f"fill rate={row['fill_rate']:.1%} | mean peak |inventory|={row['mean_peak_abs_inventory']:.2f}"
        )
    print(f"Wrote trading results to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20300000)
    parser.add_argument("--informed-fractions", type=float, nargs="+", default=[0.15, 0.30, 0.50])
    parser.add_argument("--output-dir", type=Path, default=Path("trading_results"))
    args = parser.parse_args()
    if any(not 0.0 <= fraction <= 1.0 for fraction in args.informed_fractions):
        parser.error("informed fractions must be between 0 and 1")
    run(args.episodes, args.seed, args.informed_fractions, args.output_dir)


if __name__ == "__main__":
    main()
