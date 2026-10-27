"""Evaluate the provided Akuna bot's AJR binary-option probabilities."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import replace
from pathlib import Path

from akuna_bot import (
    AJARAI_NAME,
    AJARAI_UNDERLYING_ID,
    FED_FUNDS_RATE_NAME,
    FED_FUNDS_RATE_UNDERLYING_ID,
    THERIODIC_NAME,
    THERIODIC_UNDERLYING_ID,
    BinaryOption,
    MarketHistory,
    MarketMaker,
    MarketParameters,
    OptionLeg,
    Underlying,
)


STARTING_VALUES = {
    FED_FUNDS_RATE_UNDERLYING_ID: 2.0,
    AJARAI_UNDERLYING_ID: 100.0,
    THERIODIC_UNDERLYING_ID: 100.0,
}
TRAINING_DAYS = 120
EXPIRIES = (5, 10)
STRIKE_OFFSETS = (-0.01, 0.0, 0.01)


def make_true_parameters() -> MarketParameters:
    """Choose a known synthetic market process with nonzero rate effects."""
    return replace(
        MarketMaker._default_params(),
        ajarai_drift=0.0008,
        theriodic_drift=0.0012,
        ajarai_rate_beta=-0.02,
        theriodic_rate_beta=0.012,
        sector_std_dev=0.015,
    )


def generate_history(
    parameters: MarketParameters, days: int, seed: int
) -> MarketHistory:
    """Generate past prices using the synthetic market's known parameters."""
    random.seed(seed)
    values = dict(STARTING_VALUES)
    history = {underlying_id: [value] for underlying_id, value in values.items()}
    for _ in range(days - 1):
        values = parameters.advance_step(values)
        for underlying_id, value in values.items():
            history[underlying_id].append(value)
    return MarketHistory({uid: tuple(prices) for uid, prices in history.items()})


def recent_history(history: MarketHistory, days: int) -> MarketHistory:
    return MarketHistory({
        uid: values[-days:]
        for uid, values in history.values_by_underlying_id.items()
    })


def current_underlyings(history: MarketHistory) -> list[Underlying]:
    names = {
        FED_FUNDS_RATE_UNDERLYING_ID: FED_FUNDS_RATE_NAME,
        AJARAI_UNDERLYING_ID: AJARAI_NAME,
        THERIODIC_UNDERLYING_ID: THERIODIC_NAME,
    }
    return [
        Underlying(names[uid], uid, values[-1])
        for uid, values in history.values_by_underlying_id.items()
    ]


def make_options(current_ajarai: float, first_option_id: int) -> list[BinaryOption]:
    options = []
    option_id = first_option_id
    for expiry in EXPIRIES:
        for offset in STRIKE_OFFSETS:
            options.append(BinaryOption(
                legs=(OptionLeg(AJARAI_UNDERLYING_ID, 1.0),),
                option_id=option_id,
                steps_until_expiry=expiry,
                strike=round(current_ajarai * (1.0 + offset), 2),
            ))
            option_id += 1
    return options


def simulate_payout(
    parameters: MarketParameters,
    starting_values: dict[int, float],
    option: BinaryOption,
    seed: int,
) -> int:
    """Simulate one unseen future and return 1 if the option pays, else 0."""
    random.seed(seed)
    values = dict(starting_values)
    for _ in range(option.steps_until_expiry):
        values = parameters.advance_step(values)
    return int(option.expiry_valuation(values))


def score_predictions(
    observations: list[tuple[float, int]], bins: int = 5
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    """Return Brier/calibration summaries and bins of predicted probabilities."""
    count = len(observations)
    mean_prediction = sum(p for p, _ in observations) / count
    payout_rate = sum(outcome for _, outcome in observations) / count
    brier = sum((prediction - outcome) ** 2 for prediction, outcome in observations) / count
    naive_half_brier = sum((0.5 - outcome) ** 2 for _, outcome in observations) / count
    summary = {
        "forecast_count": count,
        "mean_predicted_probability": mean_prediction,
        "observed_payout_rate": payout_rate,
        "calibration_gap": mean_prediction - payout_rate,
        "brier_score": brier,
        "always_0_5_brier": naive_half_brier,
    }

    bin_rows = []
    for bin_number in range(bins):
        lower = bin_number / bins
        upper = (bin_number + 1) / bins
        members = [
            (prediction, outcome)
            for prediction, outcome in observations
            if lower <= prediction < upper or (bin_number == bins - 1 and prediction == 1.0)
        ]
        if not members:
            continue
        bin_rows.append({
            "probability_range": f"{lower:.1f}-{upper:.1f}",
            "count": len(members),
            "mean_prediction": sum(p for p, _ in members) / len(members),
            "observed_payout_rate": sum(y for _, y in members) / len(members),
        })
    return summary, bin_rows


def fit_shrinkage_lambda(observations: list[tuple[int, float, int]]) -> float:
    """Choose a shrinkage strength using only the calibration episodes."""
    candidates = (0.0, 0.25, 0.5, 0.75, 1.0)
    return min(
        candidates,
        key=lambda strength: sum(
            (0.5 + strength * (probability - 0.5) - outcome) ** 2
            for _, probability, outcome in observations
        ) / len(observations),
    )


def write_calibration_chart(
    bins_by_scenario: dict[str, list[dict[str, float | int | str]]],
    output: Path,
) -> None:
    """Create an SVG reliability plot: predicted chance versus observed rate."""
    width, height = 820, 680
    left, top, size = 90, 70, 500
    colors = {
        "matched_120d": "#2563eb",
        "short_history_30d": "#ea580c",
        "volatility_x2": "#16a34a",
        "rate_beta_reversed": "#9333ea",
    }
    labels = {
        "matched_120d": "Matched model, 120-day history",
        "short_history_30d": "Matched model, 30-day history",
        "volatility_x2": "Future volatility doubled",
        "rate_beta_reversed": "Future rate betas reversed",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.grid{stroke:#d9dee8;stroke-width:1}.axis{stroke:#475467;stroke-width:1.2}</style>',
        '<text x="410" y="30" text-anchor="middle" font-size="20" font-weight="700">Akuna bot probability calibration</text>',
        '<text x="410" y="51" text-anchor="middle" font-size="13">Perfect calibration lies on the diagonal</text>',
    ]
    for tick in range(6):
        value = tick / 5
        x = left + size * value
        y = top + size * (1 - value)
        parts.append(f'<line class="grid" x1="{left}" y1="{y}" x2="{left + size}" y2="{y}"/>')
        parts.append(f'<line class="grid" x1="{x}" y1="{top}" x2="{x}" y2="{top + size}"/>')
        parts.append(f'<text x="{x}" y="{top + size + 20}" text-anchor="middle" font-size="11">{value:.1f}</text>')
        parts.append(f'<text x="{left - 12}" y="{y + 4}" text-anchor="end" font-size="11">{value:.1f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + size}" x2="{left + size}" y2="{top}" stroke-dasharray="5 4"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + size}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + size}" x2="{left + size}" y2="{top + size}"/>')
    parts.append(f'<text x="{left + size / 2}" y="{top + size + 48}" text-anchor="middle" font-size="13">Mean predicted probability</text>')
    parts.append(f'<text x="20" y="{top + size / 2}" transform="rotate(-90 20 {top + size / 2})" text-anchor="middle" font-size="13">Observed payout rate</text>')

    for scenario, rows in bins_by_scenario.items():
        color = colors[scenario]
        for row in rows:
            x = left + size * float(row["mean_prediction"])
            y = top + size * (1 - float(row["observed_payout_rate"]))
            radius = min(10, 4 + 0.4 * (int(row["count"]) ** 0.5))
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.78"/>')

    for index, (scenario, label) in enumerate(labels.items()):
        x = 625
        y = 105 + index * 38
        parts.append(f'<circle cx="{x}" cy="{y - 4}" r="5" fill="{colors[scenario]}"/>')
        parts.append(f'<text x="{x + 12}" y="{y}" font-size="11">{label}</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def run_experiment(episodes: int, paths_per_option: int, seed: int, output_dir: Path) -> None:
    base_parameters = make_true_parameters()
    volatility_shock = replace(
        base_parameters,
        ajarai_idio_std_dev=base_parameters.ajarai_idio_std_dev * 2,
        theriodic_idio_std_dev=base_parameters.theriodic_idio_std_dev * 2,
        sector_std_dev=base_parameters.sector_std_dev * 2,
    )
    beta_shock = replace(
        base_parameters,
        ajarai_rate_beta=-base_parameters.ajarai_rate_beta,
        theriodic_rate_beta=-base_parameters.theriodic_rate_beta,
    )
    scenarios = (
        ("matched_120d", TRAINING_DAYS, base_parameters),
        ("short_history_30d", 30, base_parameters),
        ("volatility_x2", TRAINING_DAYS, volatility_shock),
        ("rate_beta_reversed", TRAINING_DAYS, beta_shock),
    )
    observations = {name: [] for name, _, _ in scenarios}

    for episode in range(episodes):
        history_seed = seed + episode
        full_history = generate_history(base_parameters, TRAINING_DAYS, history_seed)
        current_values = {
            uid: values[-1]
            for uid, values in full_history.values_by_underlying_id.items()
        }

        for scenario_index, (name, training_days, future_parameters) in enumerate(scenarios):
            training_history = recent_history(full_history, training_days)
            options = make_options(
                current_values[AJARAI_UNDERLYING_ID],
                first_option_id=episode * 100 + scenario_index * 10,
            )
            bot = MarketMaker(current_underlyings(full_history), options, cash_balance=100_000.0)
            bot.warm_up(training_history)

            for option_index, option in enumerate(options):
                predicted_probability = bot.price_option(option)
                for path_number in range(paths_per_option):
                    future_seed = seed + 10_000_000 + episode * 10_000 + option_index * 100 + path_number
                    payout = simulate_payout(
                        future_parameters, current_values, option, future_seed
                    )
                    # Keep episode ID so calibration and final evaluation use
                    # separate histories, not randomly mixed paths.
                    observations[name].append((episode, predicted_probability, payout))

    summary_rows = []
    bin_rows = []
    bins_by_scenario = {}
    for name, pairs in observations.items():
        calibration = [row for row in pairs if row[0] < episodes // 2]
        test = [row for row in pairs if row[0] >= episodes // 2]
        if not calibration or not test:
            raise ValueError("Use at least 2 episodes to split calibration and test data")
        strength = fit_shrinkage_lambda(calibration)
        original_test = [(p, y) for _, p, y in test]
        adjusted_test = [(0.5 + strength * (p - 0.5), y) for _, p, y in test]
        summary, bins = score_predictions(original_test)
        adjusted_summary, _ = score_predictions(adjusted_test)
        summary["calibration_lambda"] = strength
        summary["calibrated_test_brier"] = adjusted_summary["brier_score"]
        summary_rows.append({"scenario": name, **summary})
        bins_by_scenario[name] = bins
        bin_rows.extend({"scenario": name, **row} for row in bins)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "akuna_probability_summary.csv"
    bins_path = output_dir / "akuna_calibration_bins.csv"
    chart_path = output_dir / "akuna_calibration.svg"
    for path, rows in ((summary_path, summary_rows), (bins_path, bin_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_calibration_chart(bins_by_scenario, chart_path)

    print(f"Episodes: {episodes}; paths per option: {paths_per_option}")
    print(f"Forecasts per scenario: {episodes * len(EXPIRIES) * len(STRIKE_OFFSETS) * paths_per_option}")
    for row in summary_rows:
        print(
            f"{row['scenario']:22s} | Brier={row['brier_score']:.4f} | "
            f"shrink lambda={row['calibration_lambda']:.2f} | "
            f"test Brier after shrink={row['calibrated_test_brier']:.4f} | "
            f"always-0.5 Brier={row['always_0_5_brier']:.4f} | "
            f"mean predicted={row['mean_predicted_probability']:.3f} | "
            f"payout rate={row['observed_payout_rate']:.3f} | "
            f"calibration gap={row['calibration_gap']:+.3f}"
        )
    print(f"Saved {summary_path}")
    print(f"Saved {bins_path}")
    print(f"Saved {chart_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--paths-per-option", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.episodes < 1 or args.paths_per_option < 1:
        parser.error("--episodes and --paths-per-option must be at least 1")
    run_experiment(args.episodes, args.paths_per_option, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
