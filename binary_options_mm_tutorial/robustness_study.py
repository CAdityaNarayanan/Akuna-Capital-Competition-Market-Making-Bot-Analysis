"""Repeat drift-shrinkage and inventory-cap sweeps across independent seeds."""

from __future__ import annotations

import argparse
import csv
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARAMETER_DIR = ROOT / "parameter_results"
TRADING_DIR = ROOT / "trading_results"
DEFAULT_SEEDS = (20301000, 20302000, 20303000)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def preserve(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)


def aggregate_cap_curves(seeds: list[int]) -> list[dict]:
    """Pool calibration summaries across seeds and plot held-out PnL by cap."""
    rows_by_seed: dict[int, list[dict[str, str]]] = {}
    for seed in seeds:
        rows_by_seed[seed] = read_csv(
            TRADING_DIR / f"robust_seed_{seed}_inventory_cap_metrics.csv"
        )
    caps = sorted({int(row["inventory_cap"]) for rows in rows_by_seed.values() for row in rows})
    aggregate_rows = []
    for cap in caps:
        all_rows = [row for rows in rows_by_seed.values() for row in rows if int(row["inventory_cap"]) == cap]
        per_seed_test = {
            seed: statistics.mean(
                float(row["heldout_mean_pnl"])
                for row in rows_by_seed[seed]
                if int(row["inventory_cap"]) == cap
            )
            for seed in seeds
        }
        aggregate_rows.append({
            "inventory_cap": cap,
            "calibration_mean_pnl_across_seeds_and_flows": statistics.mean(
                float(row["calibration_mean_pnl"]) for row in all_rows
            ),
            "heldout_mean_pnl_across_seeds_and_flows": statistics.mean(
                float(row["heldout_mean_pnl"]) for row in all_rows
            ),
            "heldout_mean_pnl_sd_across_seeds": statistics.pstdev(per_seed_test.values()),
            "heldout_positive_episode_fraction_average": statistics.mean(
                float(row["positive_pnl_episode_fraction"]) for row in all_rows
            ),
            "mean_peak_inventory_average": statistics.mean(
                float(row["mean_peak_abs_inventory"]) for row in all_rows
            ),
            "selected_by_pooled_calibration": False,
            **{f"heldout_mean_pnl_seed_{seed}": per_seed_test[seed] for seed in seeds},
        })
    selected = max(
        aggregate_rows,
        key=lambda row: row["calibration_mean_pnl_across_seeds_and_flows"],
    )
    selected["selected_by_pooled_calibration"] = True
    write_csv(ROOT / "robustness_cap_across_seeds.csv", aggregate_rows)
    write_cap_curve(aggregate_rows, seeds, ROOT / "robustness_cap_curve.svg")
    print(
        f"Pooled calibration selected cap {selected['inventory_cap']} "
        f"(mean calibration PnL ${selected['calibration_mean_pnl_across_seeds_and_flows']:.3f}); "
        f"held-out mean PnL ${selected['heldout_mean_pnl_across_seeds_and_flows']:.3f}."
    )
    return aggregate_rows


def write_cap_curve(rows: list[dict], seeds: list[int], output: Path) -> None:
    width, height = 850, 500
    left, top, plot_width, plot_height = 90, 65, 680, 330
    colors = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#0891b2"]
    caps = [int(row["inventory_cap"]) for row in rows]
    values = [float(row[f"heldout_mean_pnl_seed_{seed}"]) for row in rows for seed in seeds]
    low, high = min(values), max(values)
    pad = max((high - low) * 0.1, 0.05)
    low, high = low - pad, high + pad
    x = lambda cap: left + plot_width * (cap - min(caps)) / (max(caps) - min(caps))
    y = lambda value: top + plot_height * (high - value) / (high - low)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.grid{stroke:#e1e6ef;stroke-width:1}.axis{stroke:#475467;stroke-width:1.2}</style>',
        '<text x="425" y="28" text-anchor="middle" font-size="19" font-weight="700">Inventory cap robustness across seeds</text>',
        '<text x="425" y="48" text-anchor="middle" font-size="12">Each line averages held-out PnL equally across 15%, 30%, and 50% informed flow</text>',
    ]
    for index in range(5):
        value = low + (high - low) * index / 4
        yp = y(value)
        parts.append(f'<line class="grid" x1="{left}" y1="{yp:.1f}" x2="{left + plot_width}" y2="{yp:.1f}"/>')
        parts.append(f'<text x="{left - 8}" y="{yp + 4:.1f}" text-anchor="end" font-size="10">{value:.2f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>')
    for cap in caps:
        if cap == 1 or cap % 2 == 0:
            parts.append(f'<text x="{x(cap):.1f}" y="{top + plot_height + 18}" text-anchor="middle" font-size="10">{cap}</text>')
    for index, seed in enumerate(seeds):
        points = " ".join(
            f'{x(int(row["inventory_cap"])):.1f},{y(float(row[f"heldout_mean_pnl_seed_{seed}"])):.1f}'
            for row in rows
        )
        color = colors[index % len(colors)]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.6"/>')
        parts.append(f'<line x1="{340 + index*120}" y1="435" x2="{362 + index*120}" y2="435" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{368 + index*120}" y="439" font-size="10">Seed {seed}</text>')
    average_points = " ".join(
        f'{x(int(row["inventory_cap"])):.1f},{y(float(row["heldout_mean_pnl_across_seeds_and_flows"])):.1f}'
        for row in rows
    )
    parts.append(f'<polyline points="{average_points}" fill="none" stroke="#111827" stroke-width="3"/>')
    selected_row = next(row for row in rows if row["selected_by_pooled_calibration"])
    sx = x(int(selected_row["inventory_cap"]))
    sy = y(float(selected_row["heldout_mean_pnl_across_seeds_and_flows"]))
    parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="7" fill="none" stroke="#111827" stroke-width="2"/>')
    parts.append('<line x1="35" y1="435" x2="58" y2="435" stroke="#111827" stroke-width="3"/>')
    parts.append('<text x="64" y="439" font-size="10">Mean across seeds</text>')
    parts.append(f'<text x="{left + plot_width/2}" y="{top + plot_height + 42}" text-anchor="middle" font-size="12">Inventory cap (contracts)</text>')
    parts.append(f'<text x="20" y="{top + plot_height/2}" transform="rotate(-90 20 {top + plot_height/2})" text-anchor="middle" font-size="12">Held-out mean PnL per episode ($)</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def run(episodes: int, paths_per_option: int, lambda_step: float, seeds: list[int]) -> None:
    if episodes < 4 or episodes % 2:
        raise ValueError("episodes must be even and at least 4")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("provide one or more distinct seeds")

    for seed in seeds:
        prefix = f"robust_seed_{seed}"
        run_command([
            sys.executable, "parameter_diagnostic.py",
            "--episodes", str(episodes),
            "--paths-per-option", str(paths_per_option),
            "--seed", str(seed),
            "--lambda-step", str(lambda_step),
            "--output-dir", "parameter_results",
        ])
        for source_name, suffix in (
            ("drift_shrinkage.csv", "drift_shrinkage.csv"),
            ("drift_shrinkage_summary.csv", "drift_shrinkage_summary.csv"),
            ("drift_lambda_brier_curve.svg", "drift_lambda_brier_curve.svg"),
        ):
            preserve(PARAMETER_DIR / source_name, PARAMETER_DIR / f"{prefix}_{suffix}")

        run_command([
            sys.executable, "quote_strategy_comparison.py",
            "--episodes", str(episodes),
            "--seed", str(seed),
            "--output-dir", "trading_results",
        ])
        for source_name, suffix in (
            ("inventory_cap_metrics.csv", "inventory_cap_metrics.csv"),
            ("inventory_cap_selection.csv", "inventory_cap_selection.csv"),
            ("inventory_cap_pnl_curve.svg", "inventory_cap_pnl_curve.svg"),
        ):
            preserve(TRADING_DIR / source_name, TRADING_DIR / f"{prefix}_{suffix}")

    seed_rows = []
    selected_caps = []
    selected_lambdas: dict[int, list[float]] = {30: [], 120: []}
    selected_cap_per_seed: dict[int, int] = {}

    for seed in seeds:
        prefix = f"robust_seed_{seed}"
        summaries = read_csv(PARAMETER_DIR / f"{prefix}_drift_shrinkage_summary.csv")
        lambda_detail = read_csv(PARAMETER_DIR / f"{prefix}_drift_shrinkage.csv")
        summary_by_days = {int(row["history_days"]): row for row in summaries}
        row: dict[str, object] = {"seed": seed}
        for days in (30, 120):
            summary = summary_by_days[days]
            selected_lambda = float(summary["selected_lambda_from_calibration"])
            selected_lambdas[days].append(selected_lambda)
            selected_detail = next(
                item for item in lambda_detail
                if int(item["history_days"]) == days
                and abs(float(item["drift_lambda"]) - selected_lambda) < 1e-10
            )
            row.update({
                f"lambda_{days}d": selected_lambda,
                f"lambda_{days}d_calibration_brier": float(selected_detail["calibration_brier"]),
                f"lambda_{days}d_heldout_brier": float(summary["selected_lambda_test_brier"]),
                f"lambda_{days}d_raw_heldout_brier": float(summary["raw_drift_test_brier_lambda_1"]),
            })

        selection = read_csv(TRADING_DIR / f"{prefix}_inventory_cap_selection.csv")[0]
        cap = int(selection["selected_cap"])
        selected_caps.append(cap)
        selected_cap_per_seed[seed] = cap
        cap_metrics = read_csv(TRADING_DIR / f"{prefix}_inventory_cap_metrics.csv")
        chosen = [item for item in cap_metrics if int(item["inventory_cap"]) == cap]
        chosen_by_flow = {int(round(float(item["informed_customer_fraction"]) * 100)): item for item in chosen}
        row.update({
            "selected_cap": cap,
            "cap_calibration_mean_pnl_across_flows": float(selection["selected_calibration_mean_pnl"]),
            "cap_mean_heldout_pnl_across_flows": statistics.mean(float(item["heldout_mean_pnl"]) for item in chosen),
            "cap_mean_heldout_positive_episode_fraction": statistics.mean(float(item["positive_pnl_episode_fraction"]) for item in chosen),
            "cap_mean_peak_inventory_across_flows": statistics.mean(float(item["mean_peak_abs_inventory"]) for item in chosen),
            "cap_heldout_pnl_15pct": float(chosen_by_flow[15]["heldout_mean_pnl"]),
            "cap_heldout_pnl_30pct": float(chosen_by_flow[30]["heldout_mean_pnl"]),
            "cap_heldout_pnl_50pct": float(chosen_by_flow[50]["heldout_mean_pnl"]),
        })
        seed_rows.append(row)

    write_csv(ROOT / "robustness_seed_summary.csv", seed_rows)
    frequency_rows = []
    for days in (30, 120):
        counts: dict[float, int] = {}
        for value in selected_lambdas[days]:
            counts[value] = counts.get(value, 0) + 1
        frequency_rows.extend({
            "parameter": f"lambda_{days}d",
            "selected_value": value,
            "selection_count": count,
            "number_of_seeds": len(seeds),
        } for value, count in sorted(counts.items()))
    cap_counts: dict[int, int] = {}
    for cap in selected_caps:
        cap_counts[cap] = cap_counts.get(cap, 0) + 1
    frequency_rows.extend({
        "parameter": "inventory_cap",
        "selected_value": cap,
        "selection_count": count,
        "number_of_seeds": len(seeds),
    } for cap, count in sorted(cap_counts.items()))
    write_csv(ROOT / "robustness_selection_frequency.csv", frequency_rows)
    aggregate_cap_curves(seeds)

    print("\nRobustness summary")
    for row in seed_rows:
        print(
            f"seed={row['seed']} | lambda30={row['lambda_30d']:.2f} "
            f"(test Brier {row['lambda_30d_heldout_brier']:.4f}) | "
            f"lambda120={row['lambda_120d']:.2f} "
            f"(test Brier {row['lambda_120d_heldout_brier']:.4f}) | "
            f"cap={row['selected_cap']} | cap held-out mean PnL="
            f"${row['cap_mean_heldout_pnl_across_flows']:.3f}"
        )
    print(f"Saved per-seed summary to {ROOT / 'robustness_seed_summary.csv'}")
    print(f"Saved selection frequencies to {ROOT / 'robustness_selection_frequency.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--paths-per-option", type=int, default=20)
    parser.add_argument("--lambda-step", type=float, default=0.04)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--aggregate-only", action="store_true", help="summarize already preserved seed runs without rerunning experiments")
    args = parser.parse_args()
    if args.aggregate_only:
        aggregate_cap_curves(args.seeds)
        return
    run(args.episodes, args.paths_per_option, args.lambda_step, args.seeds)


if __name__ == "__main__":
    main()
