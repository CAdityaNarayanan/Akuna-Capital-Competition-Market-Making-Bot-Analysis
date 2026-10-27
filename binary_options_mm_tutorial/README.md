# Akuna Binary Options Bot: Synthetic Stress Study

This project studies a provided binary-options market-making bot in a small simulated market. It asks two questions:

1. How accurately can the bot estimate a market from 30 or 120 days of prices?
2. How do its actual RFQ quotes perform when customers sometimes have better information than the bot?
3. Does a symmetric quote centered on fair value behave differently from the bot's original asymmetric regimes?

The data and customer behavior are synthetic. This is a research exercise, not a reconstruction of Akuna Capital's competition or evidence of live trading profitability.

## Quick start

Requires Python 3.11+ and uses only the standard library.

From this directory, run:

```powershell
python parameter_diagnostic.py
python trading_stress_test.py
python quote_strategy_comparison.py
python robustness_study.py
```

The experiments use fixed random seeds and default to 300 episodes. The first half chooses drift shrinkage and inventory cap; the second half is held out for evaluation. Drift shrinkage tests λ from 0 to 1 in 0.01 increments. The trading simulation tests inventory caps from 1 to 20 and selects one cap using average calibration PnL across the three assumed informed-customer shares.

To change run size or seed:

```powershell
python parameter_diagnostic.py --episodes 500 --seed 20300000
python trading_stress_test.py --episodes 500 --seed 20300000 --informed-fractions 0.15 0.30 0.50
```

Episode count must be even and at least four. The experiments write CSVs and a chart into `parameter_results/` and `trading_results/` by default.

## Files

- `akuna_bot.py`: local copy of the bot code supplied for the project.
- `stress_test_akuna.py`: original binary-option probability and calibration experiment.
- `parameter_diagnostic.py`: compares estimated parameters with the known simulation parameters and plots Brier score across AJR drift shrinkage factors on calibration and held-out histories.
- `trading_stress_test.py`: sends synthetic RFQs through the preserved original `quote_regime_based()` and `on_trade()` methods, settles positions at expiry, and compares raw estimates with drift-shrunk estimates.
- `quote_strategy_comparison.py`: compares the new symmetric fair-mid `quote()` with the preserved original `quote_regime_based()` on identical held-out paths and customer requests.
- `PROJECT_REPORT.md`: full explanation of methods, results, assumptions, and limitations.
- `parameter_results/`: parameter errors, drift-shrinkage scores, and exploratory correlations.
- `trading_results/`: per-policy PnL and inventory metrics, paired comparisons, episode-level outputs, and PnL chart.
- `trading_results/quote_strategy_metrics.csv`, `quote_strategy_paired_comparison.csv`, and `quote_strategy_pnl.svg`: fair-mid quote comparison results.
- `parameter_results/drift_lambda_brier_curve.svg` and `drift_shrinkage.csv`: Brier score curve and all 101 λ values.
- `trading_results/inventory_cap_metrics.csv`, `inventory_cap_selection.csv`, and `inventory_cap_pnl_curve.svg`: full 1–20 cap sweep, selected cap, and held-out PnL curves.
- `robustness_study.py`: repeats the parameter sweeps across independent seeds and summarizes selection stability.
- `final_cap_validation.py`: compares fixed caps 2, 3, and 20 on a fresh seed without tuning.
- `robustness_seed_summary.csv`, `robustness_selection_frequency.csv`, `robustness_cap_across_seeds.csv`, and `robustness_cap_curve.svg`: cross-seed robustness results.
- `simulate.py`, `results.csv`, `pnl_curve.csv`, `pnl_curve.svg`: separate beginner simulator comparing fixed and inventory-aware quoting policies; it does not call the Akuna bot.

## Main result so far

In the synthetic probability study, 30-day histories produced much noisier drift estimates than 120-day histories. Shrinking AJR drift improved held-out Brier scores, but on the short-history case the adjusted bot still did not beat the 0.5 probability baseline.

The original RFQ quote was strongly asymmetric; the default `quote()` is now an 8-cent spread centered on fair value rounded to the nearest cent. In the quote-policy study, this kept the midpoint close to fair value and filled more often than the old regime-based quote. Neither policy had a statistically clear PnL advantage under the spread-sensitive customer model. Results remain sensitive to assumed informed flow and must not be described as real-market performance.

The initial fine-grid parameter sweep selected drift-shrinkage factors of λ=0.05 for 30-day histories and λ=0.18 for 120-day histories. In a three-seed robustness check with a coarser λ grid, λ=0.08 and λ=0.16 were selected most often for 30- and 120-day histories. The pooled cap selection chose 2 contracts; this remains provisional because one seed selected 20 and performed poorly out of sample. All conclusions are specific to the synthetic simulator.

## Important scope limits

The RFQ studies use the bot's real quoting and trade-position update methods, but they do not simulate the original Akuna environment. In particular, they do not model FOK order flow, competition scoring, fees, multiple simultaneous options, or data-driven customer behavior. Their informed and noise customer rules are declared assumptions. See `PROJECT_REPORT.md` before interpreting or presenting the results.


## Fresh fixed-cap validation

Run `python final_cap_validation.py` to compare preselected caps 2, 3, and 20 on 300 fresh episodes (seed 20304000). The run does not select a cap from its results. It writes per-episode records, summary metrics, and paired comparisons to `trading_results/final_cap_validation_*.csv`, plus a PnL chart at `trading_results/final_cap_validation_pnl.svg`. Results and interpretation are documented in `PROJECT_REPORT.md`.
