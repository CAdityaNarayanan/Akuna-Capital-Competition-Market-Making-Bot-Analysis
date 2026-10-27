"""Compare the bot's regime-based RFQ quotes with a symmetric fair-value quote."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path

import stress_test_akuna as stress
from akuna_bot import (
    AJARAI_UNDERLYING_ID,
    BinaryOption,
    MarketMaker,
    Underlying,
)


RFQS_PER_DAY = 8
INFORMED_EDGE_THRESHOLD = 0.01
NOISE_FILL_PROBABILITY = 0.30
NOISE_SPREAD_ELASTICITY = 8.0
REFERENCE_SPREAD = 0.08
MAX_CUSTOMER_QUANTITY = 3
REFERENCE_INVENTORY_LIMIT = 12
SYMMETRIC_INVENTORY_CAPS = tuple(range(1, 21))


def make_bot(history, option: BinaryOption) -> MarketMaker:
    bot = MarketMaker(stress.current_underlyings(history), [option], cash_balance=100_000.0)
    bot.warm_up(stress.recent_history(history, 30))
    bot.starting_capital = 100_000.0
    return bot


def prepare_pricing_cache(history, option_template: BinaryOption, future_states: list[dict[int, float]]) -> list[dict]:
    """Precompute fair values once per episode for all inventory-cap variants."""
    bot = make_bot(history, option_template)
    horizon = option_template.steps_until_expiry
    pricing = []
    for day in range(horizon):
        option = BinaryOption(
            legs=option_template.legs,
            option_id=option_template.option_id,
            steps_until_expiry=horizon - day,
            strike=option_template.strike,
        )
        current = future_states[day]
        bot.underlying_state = [
            Underlying(name, uid, current[uid])
            for uid, name in ((1, "FED"), (2, "AJR"), (3, "THR"))
        ]
        bot.active_option_state = [option]
        bot._greeks_cache.clear()
        bot._theo_std_cache.clear()
        greeks = bot.compute_numerical_greeks(option)
        true_fair = bot.price_option_from_parameters(stress.make_true_parameters(), option)
        pricing.append({"greeks": dict(greeks), "true_fair": true_fair})
        if day < horizon - 1:
            next_values = future_states[day + 1]
            next_option = BinaryOption(
                legs=option_template.legs,
                option_id=option_template.option_id,
                steps_until_expiry=horizon - day - 1,
                strike=option_template.strike,
            )
            bot.on_step_advance(
                [Underlying(name, uid, next_values[uid]) for uid, name in ((1, "FED"), (2, "AJR"), (3, "THR"))],
                [next_option],
            )
    return pricing


def run_episode(
    episode: int,
    policy: str,
    informed_fraction: float,
    history,
    option_template: BinaryOption,
    future_states: list[dict[int, float]],
    customer_events: list[list[tuple[float, bool, int, float]]],
    inventory_cap: int = REFERENCE_INVENTORY_LIMIT,
    pricing_cache: list[dict] | None = None,
) -> dict:
    bot = make_bot(history, option_template)
    bot.MAX_NET_INVENTORY = inventory_cap
    cash = 0.0
    fills = quotes = 0
    bought = sold = 0
    midpoint_error_sum = abs_midpoint_error_sum = spread_sum = 0.0
    position_abs_sum = peak_abs_inventory = 0
    horizon = option_template.steps_until_expiry

    for day in range(horizon):
        current = future_states[day]
        option = option_template
        # Quote at the current time to expiry, preserving the original option ID.
        option = BinaryOption(
            legs=option_template.legs,
            option_id=option_template.option_id,
            steps_until_expiry=horizon - day,
            strike=option_template.strike,
        )
        bot.underlying_state = [
            Underlying(name, uid, current[uid])
            for uid, name in ((1, "FED"), (2, "AJR"), (3, "THR"))
        ]
        bot.active_option_state = [option]
        bot._greeks_cache.clear()
        bot._theo_std_cache.clear()
        if pricing_cache is not None:
            bot._greeks_cache[option.option_id] = dict(pricing_cache[day]["greeks"])

        for counterparty_id, (informed_draw, customer_buys, customer_qty, noise_draw) in enumerate(customer_events[day]):
            fair = bot.compute_numerical_greeks(option)["theo"]
            if policy == "regime_based":
                quote = bot.quote_regime_based(option, counterparty_id)
            else:
                quote = bot.quote(option, counterparty_id)
            quotes += 1
            midpoint = (quote.bid_price + quote.offer_price) / 2.0
            midpoint_error = midpoint - fair
            midpoint_error_sum += midpoint_error
            abs_midpoint_error_sum += abs(midpoint_error)
            spread_sum += quote.offer_price - quote.bid_price

            if informed_draw < informed_fraction:
                true_fair = (
                    pricing_cache[day]["true_fair"]
                    if pricing_cache is not None
                    else bot.price_option_from_parameters(stress.make_true_parameters(), option)
                )
                if customer_buys:
                    should_fill = true_fair > quote.offer_price + INFORMED_EDGE_THRESHOLD
                else:
                    should_fill = true_fair < quote.bid_price - INFORMED_EDGE_THRESHOLD
            else:
                spread = quote.offer_price - quote.bid_price
                fill_probability = min(
                    0.80,
                    NOISE_FILL_PROBABILITY
                    * math.exp(-NOISE_SPREAD_ELASTICITY * (spread - REFERENCE_SPREAD)),
                )
                should_fill = noise_draw < fill_probability
            if not should_fill:
                continue

            position = bot.positions[option.option_id]
            if customer_buys:
                # Customer buys at the offer; cap the bot's short position.
                max_qty = max(0, inventory_cap + position)
                qty = min(customer_qty, quote.offer_quantity, max_qty)
                if qty == 0:
                    continue
                price, signed_qty = quote.offer_price, -qty
                sold += qty
            else:
                # Customer sells at the bid; cap the bot's long position.
                max_qty = max(0, inventory_cap - position)
                qty = min(customer_qty, quote.bid_quantity, max_qty)
                if qty == 0:
                    continue
                price, signed_qty = quote.bid_price, qty
                bought += qty

            cash -= price * signed_qty
            bot.cash_balance = 100_000.0 + cash
            bot.on_trade(option, price, signed_qty, counterparty_id)
            fills += 1
            position = bot.positions[option.option_id]
            position_abs_sum += abs(position)
            peak_abs_inventory = max(peak_abs_inventory, abs(position))

        if day < horizon - 1:
            next_values = future_states[day + 1]
            next_option = BinaryOption(
                legs=option_template.legs,
                option_id=option_template.option_id,
                steps_until_expiry=horizon - day - 1,
                strike=option_template.strike,
            )
            bot.on_step_advance(
                [Underlying(name, uid, next_values[uid]) for uid, name in ((1, "FED"), (2, "AJR"), (3, "THR"))],
                [next_option],
            )

    payout = option_template.expiry_valuation(future_states[horizon])
    position = bot.positions[option_template.option_id]
    return {
        "episode": episode,
        "pnl": cash + position * payout,
        "fills": fills,
        "quotes": quotes,
        "buy_contracts": bought,
        "sell_contracts": sold,
        "ending_inventory": position,
        "peak_abs_inventory": peak_abs_inventory,
        "mean_abs_inventory_at_fill": position_abs_sum / fills if fills else 0.0,
        "mean_quote_midpoint_error": midpoint_error_sum / quotes if quotes else 0.0,
        "mean_abs_quote_midpoint_error": abs_midpoint_error_sum / quotes if quotes else 0.0,
        "mean_spread": spread_sum / quotes if quotes else 0.0,
        "expiry_payout": payout,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_chart(metrics: list[dict], output: Path) -> None:
    fractions = sorted({row["informed_customer_fraction"] for row in metrics})
    width, height = 780, 500
    left, top, plot_width, plot_height = 90, 55, 620, 340
    ymin = min(row["mean_pnl"] - 2.0 * row["pnl_sd_across_episodes"] / math.sqrt(row["heldout_episodes"]) for row in metrics)
    ymax = max(row["mean_pnl"] + 2.0 * row["pnl_sd_across_episodes"] / math.sqrt(row["heldout_episodes"]) for row in metrics)
    ymin = min(-1.0, math.floor(ymin))
    ymax = max(1.0, math.ceil(ymax))
    colors = {"regime_based": "#9333ea", "symmetric_fair_mid": "#2563eb"}
    labels = {"regime_based": "Original regime quotes", "symmetric_fair_mid": "Symmetric fair-mid quotes"}
    def y_pixel(value: float) -> float:
        return top + plot_height * (ymax - value) / (ymax - ymin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.grid{stroke:#e1e6ef;stroke-width:1}.axis{stroke:#475467;stroke-width:1.2}</style>',
        '<text x="390" y="28" text-anchor="middle" font-size="19" font-weight="700">Quote strategy PnL under informed flow</text>',
        '<text x="390" y="46" text-anchor="middle" font-size="12">150 held-out episodes per point; bars show 95% normal intervals</text>',
    ]
    tick = math.floor(ymin)
    while tick <= ymax:
        y = y_pixel(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{tick}</text>')
        tick += 1 if ymax - ymin <= 8 else 2
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>')
    zero = y_pixel(0.0)
    parts.append(f'<line class="axis" x1="{left}" y1="{zero:.1f}" x2="{left + plot_width}" y2="{zero:.1f}" stroke-dasharray="5 4"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>')
    parts.append(f'<text x="{left + plot_width / 2}" y="{top + plot_height + 38}" text-anchor="middle" font-size="13">Informed customer share</text>')
    parts.append(f'<text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20 {top + plot_height / 2})" text-anchor="middle" font-size="13">Mean episode PnL ($)</text>')
    by_key = {(row["informed_customer_fraction"], row["quote_policy"]): row for row in metrics}
    for fraction_index, fraction in enumerate(fractions):
        center = left + plot_width * (fraction_index + 0.5) / len(fractions)
        parts.append(f'<text x="{center:.1f}" y="{top + plot_height + 20}" text-anchor="middle" font-size="11">{fraction:.0%}</text>')
        for offset, policy in ((-12, "regime_based"), (12, "symmetric_fair_mid")):
            row = by_key[(fraction, policy)]
            x = center + offset
            half = 1.96 * row["pnl_sd_across_episodes"] / math.sqrt(row["heldout_episodes"])
            y_hi = y_pixel(row["mean_pnl"] + half)
            y_lo = y_pixel(row["mean_pnl"] - half)
            y_mean = y_pixel(row["mean_pnl"])
            parts.append(f'<line x1="{x}" y1="{y_hi:.1f}" x2="{x}" y2="{y_lo:.1f}" stroke="{colors[policy]}" stroke-width="2"/>')
            parts.append(f'<line x1="{x - 4}" y1="{y_hi:.1f}" x2="{x + 4}" y2="{y_hi:.1f}" stroke="{colors[policy]}" stroke-width="2"/>')
            parts.append(f'<line x1="{x - 4}" y1="{y_lo:.1f}" x2="{x + 4}" y2="{y_lo:.1f}" stroke="{colors[policy]}" stroke-width="2"/>')
            parts.append(f'<circle cx="{x}" cy="{y_mean:.1f}" r="5" fill="{colors[policy]}"/>')
    for i, policy in enumerate(("regime_based", "symmetric_fair_mid")):
        x, y = 480 + i * 150, 440
        parts.append(f'<circle cx="{x}" cy="{y - 4}" r="5" fill="{colors[policy]}"/>')
        parts.append(f'<text x="{x + 10}" y="{y}" font-size="11">{labels[policy]}</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def write_inventory_cap_chart(rows: list[dict], output: Path) -> None:
    """Plot held-out mean PnL by cap, with 95% normal intervals."""
    width, height = 900, 520
    left, top, plot_width, plot_height = 80, 70, 740, 340
    colors = {0.15: "#2563eb", 0.30: "#ea580c", 0.50: "#16a34a"}
    fractions = sorted({float(row["informed_customer_fraction"]) for row in rows})
    caps = sorted({int(row["inventory_cap"]) for row in rows})
    ymin = min(float(row["heldout_mean_pnl"]) - 1.96 * float(row["heldout_pnl_se"]) for row in rows)
    ymax = max(float(row["heldout_mean_pnl"]) + 1.96 * float(row["heldout_pnl_se"]) for row in rows)
    if ymin == ymax:
        ymin, ymax = ymin - 1.0, ymax + 1.0
    pad = (ymax - ymin) * 0.08
    ymin, ymax = ymin - pad, ymax + pad
    x = lambda cap: left + plot_width * (cap - min(caps)) / (max(caps) - min(caps))
    y = lambda value: top + plot_height * (ymax - value) / (ymax - ymin)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.grid{stroke:#e1e6ef;stroke-width:1}.axis{stroke:#475467;stroke-width:1.2}</style>',
        '<text x="450" y="30" text-anchor="middle" font-size="20" font-weight="700">Inventory cap versus held-out PnL</text>',
        '<text x="450" y="51" text-anchor="middle" font-size="12">Each point uses the same held-out episodes; bars show 95% normal intervals</text>',
    ]
    tick_count = 6
    for tick_index in range(tick_count):
        value = ymin + (ymax - ymin) * tick_index / (tick_count - 1)
        yp = y(value)
        parts.append(f'<line class="grid" x1="{left}" y1="{yp:.1f}" x2="{left + plot_width}" y2="{yp:.1f}"/>')
        parts.append(f'<text x="{left - 9}" y="{yp + 4:.1f}" text-anchor="end" font-size="10">{value:.2f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>')
    zero_y = y(0.0)
    parts.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_width}" y2="{zero_y:.1f}" stroke="#667085" stroke-dasharray="4 4"/>')
    for cap in caps:
        if cap == 1 or cap % 2 == 0:
            xp = x(cap)
            parts.append(f'<text x="{xp:.1f}" y="{top + plot_height + 18}" text-anchor="middle" font-size="10">{cap}</text>')
    for fraction in fractions:
        series = sorted((row for row in rows if float(row["informed_customer_fraction"]) == fraction), key=lambda row: int(row["inventory_cap"]))
        points = " ".join(f'{x(int(row["inventory_cap"])):.1f},{y(float(row["heldout_mean_pnl"])):.1f}' for row in series)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[fraction]}" stroke-width="2.5"/>')
        for row in series:
            xp = x(int(row["inventory_cap"]))
            mean = float(row["heldout_mean_pnl"])
            half = 1.96 * float(row["heldout_pnl_se"])
            yh, yl = y(mean + half), y(mean - half)
            parts.append(f'<line x1="{xp:.1f}" y1="{yh:.1f}" x2="{xp:.1f}" y2="{yl:.1f}" stroke="{colors[fraction]}" stroke-width="1"/>')
            parts.append(f'<circle cx="{xp:.1f}" cy="{y(mean):.1f}" r="3.2" fill="{colors[fraction]}"/>')
        selected_cap = next(int(row["selected_by_calibration"]) for row in series if row["selected_by_calibration"])
        chosen = next(row for row in series if int(row["inventory_cap"]) == selected_cap)
        parts.append(f'<circle cx="{x(selected_cap):.1f}" cy="{y(float(chosen["heldout_mean_pnl"])):.1f}" r="7" fill="none" stroke="{colors[fraction]}" stroke-width="2"/>')
    parts.append(f'<text x="{left + plot_width/2}" y="{top + plot_height + 42}" text-anchor="middle" font-size="12">Maximum absolute inventory (contracts)</text>')
    parts.append(f'<text x="20" y="{top + plot_height/2}" transform="rotate(-90 20 {top + plot_height/2})" text-anchor="middle" font-size="12">Mean episode PnL ($)</text>')
    for index, fraction in enumerate(fractions):
        x0 = 280 + index * 150
        parts.append(f'<line x1="{x0}" y1="480" x2="{x0 + 22}" y2="480" stroke="{colors[fraction]}" stroke-width="3"/>')
        parts.append(f'<text x="{x0 + 28}" y="484" font-size="11">{fraction:.0%} informed</text>')
    parts.append('<circle cx="705" cy="480" r="7" fill="none" stroke="#475467" stroke-width="2"/>')
    parts.append('<text x="718" y="484" font-size="11">Selected on calibration</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def run(episodes: int, seed: int, informed_fractions: list[float], output: Path) -> None:
    if episodes < 4 or episodes % 2:
        raise ValueError("episodes must be even and at least 4")
    heldout_start = episodes // 2
    rows_by_case = {
        (fraction, policy): []
        for fraction in informed_fractions
        for policy in ("regime_based", "symmetric_fair_mid")
    }
    cap_rows = {
        (fraction, cap, partition): []
        for fraction in informed_fractions
        for cap in SYMMETRIC_INVENTORY_CAPS
        for partition in ("calibration", "heldout")
    }

    truth = stress.make_true_parameters()
    for episode in range(episodes):
        history = stress.generate_history(truth, stress.TRAINING_DAYS, seed + episode)
        spot = history.values_by_underlying_id[AJARAI_UNDERLYING_ID][-1]
        option_template = stress.make_options(spot, episode * 100)[episode % 6]
        horizon = option_template.steps_until_expiry

        current = {uid: values[-1] for uid, values in history.values_by_underlying_id.items()}
        random.seed(seed + 40_000_000 + episode)
        future_states = [dict(current)]
        for _ in range(horizon):
            current = truth.advance_step(current)
            future_states.append(dict(current))
        pricing_cache = prepare_pricing_cache(history, option_template, future_states)

        event_rng = random.Random(seed + 50_000_000 + episode)
        customer_events = []
        for _ in range(horizon):
            day_events = []
            for _ in range(RFQS_PER_DAY):
                day_events.append((
                    event_rng.random(),
                    event_rng.random() < 0.5,
                    event_rng.randint(1, MAX_CUSTOMER_QUANTITY),
                    event_rng.random(),
                ))
            customer_events.append(day_events)

        for fraction in informed_fractions:
            for cap in SYMMETRIC_INVENTORY_CAPS:
                result = run_episode(
                    episode, "symmetric_fair_mid", fraction, history, option_template,
                    future_states, customer_events, cap, pricing_cache,
                )
                partition = "calibration" if episode < heldout_start else "heldout"
                cap_rows[(fraction, cap, partition)].append(result)
            if episode >= heldout_start:
                rows_by_case[(fraction, "regime_based")].append(run_episode(
                    episode, "regime_based", fraction, history, option_template,
                    future_states, customer_events, REFERENCE_INVENTORY_LIMIT,
                ))
                rows_by_case[(fraction, "symmetric_fair_mid")].append(
                    cap_rows[(fraction, REFERENCE_INVENTORY_LIMIT, "heldout")][-1]
                )

    summary_rows = []
    paired_rows = []
    cap_summary_rows = []
    cap_paired_rows = []
    cap_selection_rows = []
    episode_rows = []
    for fraction in informed_fractions:
        for policy in ("regime_based", "symmetric_fair_mid"):
            rows = rows_by_case[(fraction, policy)]
            pnls = [row["pnl"] for row in rows]
            summary_rows.append({
                "informed_customer_fraction": fraction,
                "quote_policy": policy,
                "heldout_episodes": len(rows),
                "mean_pnl": statistics.mean(pnls),
                "pnl_sd_across_episodes": statistics.pstdev(pnls),
                "positive_pnl_episode_fraction": sum(x > 0 for x in pnls) / len(pnls),
                "fill_rate": sum(row["fills"] for row in rows) / sum(row["quotes"] for row in rows),
                "mean_peak_abs_inventory": statistics.mean(row["peak_abs_inventory"] for row in rows),
                "mean_quote_midpoint_error": statistics.mean(row["mean_quote_midpoint_error"] for row in rows),
                "mean_abs_quote_midpoint_error": statistics.mean(row["mean_abs_quote_midpoint_error"] for row in rows),
                "mean_spread": statistics.mean(row["mean_spread"] for row in rows),
            })
            episode_rows.extend({"informed_customer_fraction": fraction, "quote_policy": policy, **row} for row in rows)

        old = rows_by_case[(fraction, "regime_based")]
        new = rows_by_case[(fraction, "symmetric_fair_mid")]
        differences = [b["pnl"] - a["pnl"] for a, b in zip(old, new)]
        mean_diff = statistics.mean(differences)
        se = statistics.stdev(differences) / math.sqrt(len(differences))
        paired_rows.append({
            "informed_customer_fraction": fraction,
            "paired_mean_pnl_symmetric_minus_regime": mean_diff,
            "paired_difference_se": se,
            "ci95_low_normal": mean_diff - 1.96 * se,
            "ci95_high_normal": mean_diff + 1.96 * se,
            "heldout_episodes": len(differences),
        })

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "quote_strategy_metrics.csv", summary_rows)
    write_csv(output / "quote_strategy_paired_comparison.csv", paired_rows)
    write_csv(output / "quote_strategy_episode_results.csv", episode_rows)
    aggregate_calibration_means = {
        cap: statistics.mean(
            row["pnl"]
            for fraction in informed_fractions
            for row in cap_rows[(fraction, cap, "calibration")]
        )
        for cap in SYMMETRIC_INVENTORY_CAPS
    }
    selected_cap = max(SYMMETRIC_INVENTORY_CAPS, key=lambda cap: aggregate_calibration_means[cap])
    cap_selection_rows.append({
        "selection_scope": "equal-weight average across informed-flow scenarios",
        "selection_criterion": "highest mean PnL on calibration episodes",
        "selected_cap": selected_cap,
        "selected_calibration_mean_pnl": aggregate_calibration_means[selected_cap],
        "calibration_episodes_per_scenario": heldout_start,
        "heldout_episodes_per_scenario": episodes - heldout_start,
    })
    for fraction in informed_fractions:
        calibration_means = {
            cap: statistics.mean(row["pnl"] for row in cap_rows[(fraction, cap, "calibration")])
            for cap in SYMMETRIC_INVENTORY_CAPS
        }
        for cap in SYMMETRIC_INVENTORY_CAPS:
            calibration = cap_rows[(fraction, cap, "calibration")]
            rows = cap_rows[(fraction, cap, "heldout")]
            pnls = [row["pnl"] for row in rows]
            cap_summary_rows.append({
                "informed_customer_fraction": fraction,
                "inventory_cap": cap,
                "selected_by_calibration": cap == selected_cap,
                "calibration_mean_pnl": calibration_means[cap],
                "calibration_pnl_sd": statistics.pstdev(row["pnl"] for row in calibration),
                "heldout_episodes": len(rows),
                "heldout_mean_pnl": statistics.mean(pnls),
                "heldout_pnl_sd": statistics.pstdev(pnls),
                "heldout_pnl_se": statistics.stdev(pnls) / math.sqrt(len(pnls)),
                "heldout_ci95_low_normal": statistics.mean(pnls) - 1.96 * statistics.stdev(pnls) / math.sqrt(len(pnls)),
                "heldout_ci95_high_normal": statistics.mean(pnls) + 1.96 * statistics.stdev(pnls) / math.sqrt(len(pnls)),
                "positive_pnl_episode_fraction": sum(x > 0 for x in pnls) / len(pnls),
                "fill_rate": sum(row["fills"] for row in rows) / sum(row["quotes"] for row in rows),
                "mean_peak_abs_inventory": statistics.mean(row["peak_abs_inventory"] for row in rows),
                "mean_ending_abs_inventory": statistics.mean(abs(row["ending_inventory"]) for row in rows),
            })
        reference = cap_rows[(fraction, REFERENCE_INVENTORY_LIMIT, "heldout")]
        for cap in SYMMETRIC_INVENTORY_CAPS:
            if cap == REFERENCE_INVENTORY_LIMIT:
                continue
            tested = cap_rows[(fraction, cap, "heldout")]
            diffs = [b["pnl"] - a["pnl"] for a, b in zip(reference, tested)]
            mean_diff = statistics.mean(diffs)
            se = statistics.stdev(diffs) / math.sqrt(len(diffs))
            cap_paired_rows.append({
                "informed_customer_fraction": fraction,
                "tested_cap": cap,
                "reference_cap": REFERENCE_INVENTORY_LIMIT,
                "paired_mean_pnl_difference": mean_diff,
                "ci95_low_normal": mean_diff - 1.96 * se,
                "ci95_high_normal": mean_diff + 1.96 * se,
                "heldout_episodes": len(diffs),
            })
    write_csv(output / "inventory_cap_metrics.csv", cap_summary_rows)
    write_csv(output / "inventory_cap_paired_comparison.csv", cap_paired_rows)
    write_csv(output / "inventory_cap_selection.csv", cap_selection_rows)
    write_chart(summary_rows, output / "quote_strategy_pnl.svg")
    write_inventory_cap_chart(cap_summary_rows, output / "inventory_cap_pnl_curve.svg")
    print(f"Episodes: {episodes}; held-out episodes: {episodes - heldout_start}")
    print(
        "Symmetric policy: 8-cent spread centered at nearest-cent fair value; inventory caps swept at "
        + " through ".join((str(min(SYMMETRIC_INVENTORY_CAPS)), str(max(SYMMETRIC_INVENTORY_CAPS))))
        + " contracts. The cap is selected on the first half of episodes and evaluated on the second half."
    )
    print(
        f"Noise fills: 30% at an 8-cent spread, decaying exponentially with elasticity {NOISE_SPREAD_ELASTICITY:g}."
    )
    for row in summary_rows:
        print(
            f"informed={row['informed_customer_fraction']:.0%} | {row['quote_policy']:18s} | "
            f"PnL=${row['mean_pnl']:.3f} | mid error={row['mean_quote_midpoint_error']:+.4f} | "
            f"fill rate={row['fill_rate']:.1%} | peak inventory={row['mean_peak_abs_inventory']:.2f}"
        )
    for row in paired_rows:
        print(
            f"  symmetric - regime PnL at {row['informed_customer_fraction']:.0%} informed: "
            f"${row['paired_mean_pnl_symmetric_minus_regime']:.3f} "
            f"(95% CI ${row['ci95_low_normal']:.3f} to ${row['ci95_high_normal']:.3f})"
        )
    print(f"Saved comparison to {output.resolve()}")
    print("Inventory-cap sweep; calibration selects the cap, held-out results are shown below:")
    for row in cap_summary_rows:
        print(
            f"  informed={row['informed_customer_fraction']:.0%} | cap={row['inventory_cap']:2d} | "
            f"cal PnL=${row['calibration_mean_pnl']:.3f} | test PnL=${row['heldout_mean_pnl']:.3f} "
            f"(95% CI ${row['heldout_ci95_low_normal']:.3f} to ${row['heldout_ci95_high_normal']:.3f}) | "
            f"fill={row['fill_rate']:.1%} | "
            f"peak inventory={row['mean_peak_abs_inventory']:.2f} | "
            f"ending |inventory|={row['mean_ending_abs_inventory']:.2f} | "
            f"selected={bool(row['selected_by_calibration'])}"
        )
    for row in cap_paired_rows:
        print(
            f"  cap {row['tested_cap']} - cap {row['reference_cap']} at "
            f"{row['informed_customer_fraction']:.0%} informed: "
            f"${row['paired_mean_pnl_difference']:.3f} "
            f"(95% CI ${row['ci95_low_normal']:.3f} to ${row['ci95_high_normal']:.3f})"
        )
    selected = [item for item in cap_summary_rows if item["inventory_cap"] == selected_cap]
    print(
        f"  Selected cap {selected_cap} from calibration average across flow scenarios; "
        f"held-out PnL by flow: "
        + ", ".join(f"{item['informed_customer_fraction']:.0%}=${item['heldout_mean_pnl']:.3f}" for item in selected)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20300000)
    parser.add_argument("--informed-fractions", type=float, nargs="+", default=[0.15, 0.30, 0.50])
    parser.add_argument("--output-dir", type=Path, default=Path("trading_results"))
    args = parser.parse_args()
    if any(not 0.0 <= value <= 1.0 for value in args.informed_fractions):
        parser.error("informed fractions must be between 0 and 1")
    run(args.episodes, args.seed, args.informed_fractions, args.output_dir)


if __name__ == "__main__":
    main()
