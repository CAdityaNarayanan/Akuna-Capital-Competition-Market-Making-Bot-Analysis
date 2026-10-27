"""Reproduce parameter-estimation and drift-shrinkage diagnostics.

All price data and option outcomes in this experiment are synthetic. The script
uses independent histories as the evaluation units and holds out the second half
of episodes when selecting drift shrinkage.
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
from akuna_bot import AJARAI_UNDERLYING_ID, MarketMaker


WINDOWS = (30, 120)
PARAMETERS = (
    "ajarai_drift",
    "theriodic_drift",
    "ajarai_rate_beta",
    "theriodic_rate_beta",
    "sector_std_dev",
    "ajarai_idio_std_dev",
    "theriodic_idio_std_dev",
)


def pearson(xs: list[float], ys: list[float]) -> float:
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_lambda_chart(rows: list[dict], output: Path) -> None:
    """Plot calibration and held-out Brier scores over the full lambda grid."""
    width, height = 980, 520
    panel_lefts = (90, 535)
    panel_top, panel_width, panel_height = 80, 360, 330
    colors = {"calibration_brier": "#2563eb", "held_out_test_brier": "#ea580c"}
    labels = {"calibration_brier": "Calibration", "held_out_test_brier": "Held-out"}
    grouped = {
        days: sorted((row for row in rows if row["history_days"] == days), key=lambda r: r["drift_lambda"])
        for days in WINDOWS
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.grid{stroke:#e1e6ef;stroke-width:1}.axis{stroke:#475467;stroke-width:1.2}</style>',
        '<text x="490" y="30" text-anchor="middle" font-size="20" font-weight="700">Brier score versus drift shrinkage factor</text>',
        '<text x="490" y="52" text-anchor="middle" font-size="12">Lower is better; lambda is selected using calibration episodes only</text>',
    ]
    for panel_index, days in enumerate(WINDOWS):
        left = panel_lefts[panel_index]
        group = grouped[days]
        values = [float(row[key]) for row in group for key in colors]
        low, high = min(values + [0.25]), max(values + [0.25])
        pad = max((high - low) * 0.08, 0.002)
        low, high = low - pad, high + pad
        y = lambda value: panel_top + panel_height * (high - value) / (high - low)
        x = lambda value: left + panel_width * value
        parts.append(f'<text x="{left + panel_width/2:.1f}" y="72" text-anchor="middle" font-size="15" font-weight="700">{days}-day history</text>')
        for tick_index in range(5):
            val = low + (high - low) * tick_index / 4
            yp = y(val)
            parts.append(f'<line class="grid" x1="{left}" y1="{yp:.1f}" x2="{left + panel_width}" y2="{yp:.1f}"/>')
            parts.append(f'<text x="{left - 8}" y="{yp + 4:.1f}" text-anchor="end" font-size="10">{val:.3f}</text>')
        parts.append(f'<line class="axis" x1="{left}" y1="{panel_top}" x2="{left}" y2="{panel_top + panel_height}"/>')
        parts.append(f'<line class="axis" x1="{left}" y1="{panel_top + panel_height}" x2="{left + panel_width}" y2="{panel_top + panel_height}"/>')
        baseline_y = y(0.25)
        parts.append(f'<line x1="{left}" y1="{baseline_y:.1f}" x2="{left + panel_width}" y2="{baseline_y:.1f}" stroke="#667085" stroke-dasharray="4 4"/>')
        for tick_index in range(6):
            val = tick_index / 5
            xp = x(val)
            parts.append(f'<text x="{xp:.1f}" y="{panel_top + panel_height + 18}" text-anchor="middle" font-size="10">{val:.1f}</text>')
        for key, color in colors.items():
            points = " ".join(f'{x(float(row["drift_lambda"])):.1f},{y(float(row[key])):.1f}' for row in group)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        selected = min(group, key=lambda row: row["calibration_brier"])
        sx, sy = x(float(selected["drift_lambda"])), y(float(selected["calibration_brier"]))
        parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" fill="#2563eb" stroke="white" stroke-width="1.5"/>')
        parts.append(f'<text x="{left + panel_width/2:.1f}" y="{panel_top + panel_height + 42}" text-anchor="middle" font-size="12">Drift factor lambda</text>')
    parts.append('<text x="20" y="245" transform="rotate(-90 20 245)" text-anchor="middle" font-size="12">Mean Brier score</text>')
    for index, (key, color) in enumerate(colors.items()):
        x0 = 375 + index * 130
        parts.append(f'<line x1="{x0}" y1="475" x2="{x0 + 24}" y2="475" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x0 + 30}" y="479" font-size="11">{labels[key]}</text>')
    parts.append('<line x1="635" y1="475" x2="659" y2="475" stroke="#667085" stroke-dasharray="4 4"/>')
    parts.append('<text x="665" y="479" font-size="11">Brier score of 0.5 forecast</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def run(
    episodes: int,
    paths_per_option: int,
    seed: int,
    output_dir: Path,
    lambda_step: float = 0.01,
) -> None:
    if episodes < 4 or episodes % 2:
        raise ValueError("episodes must be an even number of at least 4")
    if paths_per_option < 1:
        raise ValueError("paths_per_option must be at least 1")
    grid_count = round(1.0 / lambda_step) if lambda_step > 0 else 0
    if lambda_step <= 0 or not math.isclose(grid_count * lambda_step, 1.0, abs_tol=1e-9):
        raise ValueError("lambda_step must be positive and divide 1 exactly (for example, 0.01 or 0.04)")
    drift_lambdas = tuple(round(index * lambda_step, 10) for index in range(grid_count + 1))

    truth = stress.make_true_parameters()
    parameter_samples = {
        days: {name: [] for name in PARAMETERS} for days in WINDOWS
    }
    # One value per independent history, averaged over its six options and
    # future paths. This avoids treating repeated paths as separate forecasts.
    episode_scores = {
        days: {lam: [] for lam in drift_lambdas} for days in WINDOWS
    }
    beta_errors = {days: [] for days in WINDOWS}
    drift_errors = {days: [] for days in WINDOWS}

    for episode in range(episodes):
        history = stress.generate_history(truth, stress.TRAINING_DAYS, seed + episode)
        current = {
            uid: values[-1]
            for uid, values in history.values_by_underlying_id.items()
        }

        for days in WINDOWS:
            bot = MarketMaker(stress.current_underlyings(history), [], cash_balance=100_000.0)
            bot.warm_up(stress.recent_history(history, days))
            estimated = bot.estimated_params
            if estimated is None:
                raise RuntimeError(f"Bot did not estimate parameters for {days} days")

            for field in PARAMETERS:
                parameter_samples[days][field].append(getattr(estimated, field))
            beta_errors[days].append(
                abs(estimated.ajarai_rate_beta - truth.ajarai_rate_beta)
            )
            drift_errors[days].append(
                abs(estimated.ajarai_drift - truth.ajarai_drift)
            )

            options = stress.make_options(
                current[AJARAI_UNDERLYING_ID], episode * 100 + days
            )
            losses_by_lambda = {lam: [] for lam in drift_lambdas}
            for option_index, option in enumerate(options):
                outcomes = []
                for path_number in range(paths_per_option):
                    future_seed = (
                        seed + 10_000_000 + episode * 10_000
                        + option_index * 100 + path_number
                    )
                    outcomes.append(
                        stress.simulate_payout(truth, current, option, future_seed)
                    )

                for lam in drift_lambdas:
                    bot.estimated_params = replace(
                        estimated, ajarai_drift=lam * estimated.ajarai_drift
                    )
                    # price_option caches its calculation by option ID. Clear
                    # caches because this experiment deliberately changes drift.
                    bot._theo_std_cache.clear()
                    bot._greeks_cache.clear()
                    # This experiment needs only the fair probability. Calling
                    # price_option() also calculates delta, gamma, and theta,
                    # which are unrelated to the Brier score and much slower.
                    probability = bot.price_option_from_parameters(
                        bot.estimated_params, option
                    )
                    losses_by_lambda[lam].extend(
                        (probability - outcome) ** 2 for outcome in outcomes
                    )

            for lam, losses in losses_by_lambda.items():
                episode_scores[days][lam].append((episode, statistics.mean(losses)))

    parameter_rows = []
    for days in WINDOWS:
        for field in PARAMETERS:
            values = parameter_samples[days][field]
            actual = getattr(truth, field)
            parameter_rows.append({
                "history_days": days,
                "parameter": field,
                "true_value": actual,
                "mean_estimate": statistics.mean(values),
                "estimate_sd_across_histories": statistics.pstdev(values),
                "mean_absolute_error": statistics.mean(abs(x - actual) for x in values),
                "history_count": episodes,
            })

    split = episodes // 2
    drift_rows = []
    summary_rows = []
    correlation_rows = []
    for days in WINDOWS:
        calibration_scores = {}
        test_scores = {}
        for lam in drift_lambdas:
            rows = episode_scores[days][lam]
            calibration = [score for episode, score in rows if episode < split]
            test = [score for episode, score in rows if episode >= split]
            calibration_scores[lam] = statistics.mean(calibration)
            test_scores[lam] = statistics.mean(test)
            drift_rows.append({
                "history_days": days,
                "drift_lambda": lam,
                "calibration_brier": calibration_scores[lam],
                "held_out_test_brier": test_scores[lam],
                "calibration_episodes": split,
                "test_episodes": episodes - split,
            })

        selected = min(drift_lambdas, key=lambda lam: calibration_scores[lam])
        summary_rows.append({
            "history_days": days,
            "lambda_step": lambda_step,
            "lambda_grid_size": len(drift_lambdas),
            "selected_lambda_from_calibration": selected,
            "selected_lambda_test_brier": test_scores[selected],
            "raw_drift_test_brier_lambda_1": test_scores[1.0],
            "always_0_5_test_brier": 0.25,
            "calibration_histories": split,
            "held_out_histories": episodes - split,
        })

        raw_scores = [score for _, score in episode_scores[days][1.0]]
        correlation_rows.extend((
            {
                "history_days": days,
                "parameter_error": "absolute_ajarai_drift_error",
                "correlation_with_raw_episode_brier": pearson(drift_errors[days], raw_scores),
                "interpretation_note": "Exploratory association; does not establish causation.",
            },
            {
                "history_days": days,
                "parameter_error": "absolute_ajarai_rate_beta_error",
                "correlation_with_raw_episode_brier": pearson(beta_errors[days], raw_scores),
                "interpretation_note": "Exploratory association; does not establish causation.",
            },
        ))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "parameter_estimates.csv", parameter_rows)
    write_csv(output_dir / "drift_shrinkage.csv", drift_rows)
    write_csv(output_dir / "drift_shrinkage_summary.csv", summary_rows)
    write_csv(output_dir / "parameter_error_correlations.csv", correlation_rows)
    write_lambda_chart(drift_rows, output_dir / "drift_lambda_brier_curve.svg")

    print(f"Episodes: {episodes}; future paths per option: {paths_per_option}; lambda step: {lambda_step:g} ({len(drift_lambdas)} values)")
    print(f"Calibration histories: {split}; held-out histories: {episodes - split}")
    for row in summary_rows:
        print(
            f"{row['history_days']:3d}d | selected lambda={row['selected_lambda_from_calibration']:.2f} | "
            f"held-out Brier={row['selected_lambda_test_brier']:.5f} | "
            f"raw drift={row['raw_drift_test_brier_lambda_1']:.5f} | baseline=0.25000"
        )
    print(f"Wrote diagnostics to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--paths-per-option", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20300000)
    parser.add_argument("--output-dir", type=Path, default=Path("parameter_results"))
    parser.add_argument("--lambda-step", type=float, default=0.01)
    args = parser.parse_args()
    run(args.episodes, args.paths_per_option, args.seed, args.output_dir, args.lambda_step)


if __name__ == "__main__":
    main()
