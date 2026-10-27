import math
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Final

# ============================================================================
# CONSTANTS & DATACLASSES
# ============================================================================

AJARAI_NAME: Final[str] = "AJR"
AJARAI_UNDERLYING_ID: Final[int] = 2
FED_FUNDS_RATE_NAME: Final[str] = "FED"
FED_FUNDS_RATE_UNDERLYING_ID: Final[int] = 1
RATE_STRIKE_GRID: Final[float] = 0.25
THERIODIC_NAME: Final[str] = "THR"
THERIODIC_UNDERLYING_ID: Final[int] = 3

UNDERLYING_NAME_BY_ID: Final[dict[int, str]] = {
    AJARAI_UNDERLYING_ID: AJARAI_NAME,
    FED_FUNDS_RATE_UNDERLYING_ID: FED_FUNDS_RATE_NAME,
    THERIODIC_UNDERLYING_ID: THERIODIC_NAME,
}


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class BinaryOption:
    legs: "tuple[OptionLeg, ...]"
    option_id: int
    steps_until_expiry: int
    strike: float

    def __post_init__(self) -> None:
        if self.steps_until_expiry < 0:
            raise ValueError("Steps until expiry must be non-negative")

        if not self.legs:
            raise ValueError("Binary option must have at least one leg")

        underlying_ids: list[int] = [leg.underlying_id for leg in self.legs]
        if len(underlying_ids) != len(set(underlying_ids)):
            raise ValueError("Binary option legs must reference distinct underlyings")

        if any(leg.weight == 0 for leg in self.legs):
            raise ValueError("Binary option leg weights must be non-zero")

    def __str__(self) -> str:
        terms: list[str] = []
        for index, leg in enumerate(self.legs):
            name: str = UNDERLYING_NAME_BY_ID.get(leg.underlying_id, str(leg.underlying_id))
            magnitude: float = abs(leg.weight)
            magnitude_str: str = "" if magnitude == 1 else f"{magnitude:.2f}*"
            if index == 0:
                sign: str = "-" if leg.weight < 0 else ""
            else:
                sign = " - " if leg.weight < 0 else " + "
            terms.append(f"{sign}{magnitude_str}{name}")
        observable_expression: str = "".join(terms)
        return f"{self.option_id} ({self.steps_until_expiry}d {observable_expression} >= {self.strike:.2f})"

    def advance_step(self) -> "BinaryOption":
        if self.steps_until_expiry == 0:
            return self

        return replace(self, steps_until_expiry=self.steps_until_expiry - 1)

    def contract_matches(self, other: "BinaryOption") -> bool:
        return replace(other, option_id=self.option_id) == self

    def expiry_valuation(self, value_by_underlying_id: dict[int, float]) -> float:
        return 1.0 if self.observable_value(value_by_underlying_id) >= self.strike else 0.0

    def observable_value(self, value_by_underlying_id: dict[int, float]) -> float:
        return sum(leg.weight * value_by_underlying_id[leg.underlying_id] for leg in self.legs)


@dataclass(frozen=True)
class FokOrder:
    counterparty_id: int
    option_id: int
    order_type: "OrderType"
    price: float
    quantity: int

    def __post_init__(self) -> None:
        if self.price < 0:
            raise ValueError("FOK order price must be non-negative")

        if self.quantity <= 0:
            raise ValueError("FOK order quantity must be positive")


@dataclass(frozen=True)
class MarketHistory:
    values_by_underlying_id: dict[int, tuple[float, ...]]

    def __post_init__(self) -> None:
        lengths: set[int] = {len(values) for values in self.values_by_underlying_id.values()}
        if len(lengths) > 1:
            raise ValueError("All underlyings must have the same number of historical days")

        if lengths and next(iter(lengths)) <= 0:
            raise ValueError("Market history must contain at least one day")

    @property
    def num_days(self) -> int:
        if not self.values_by_underlying_id:
            return 0
        return len(next(iter(self.values_by_underlying_id.values())))


@dataclass(frozen=True)
class MarketParameters:
    ajarai_drift: float
    ajarai_idio_std_dev: float
    ajarai_rate_beta: float
    ajarai_sector_beta: float
    rate_down_probability: float
    rate_reversion_strength: float
    rate_up_probability: float
    sector_std_dev: float
    theriodic_drift: float
    theriodic_idio_std_dev: float
    theriodic_rate_beta: float
    theriodic_sector_beta: float

    rate_step: float = 0.25
    rate_target: float = 2.0

    def __post_init__(self) -> None:
        if self.rate_step <= 0:
            raise ValueError("Rate step must be positive")

        if self.rate_up_probability <= 0 or self.rate_down_probability <= 0:
            raise ValueError("Rate up/down probabilities must both be positive")

        if self.rate_up_probability + self.rate_down_probability > 1:
            raise ValueError("Rate up/down probabilities must not sum to more than 1")

        if self.rate_target < 0:
            raise ValueError("Rate target must be non-negative")

        if not (0 <= self.rate_reversion_strength <= 1):
            raise ValueError("Rate reversion strength must be between 0 and 1")

        if self.ajarai_idio_std_dev < 0 or self.theriodic_idio_std_dev < 0 or self.sector_std_dev < 0:
            raise ValueError("Standard deviations must be non-negative")

    def advance_company_value(
        self,
        current_value: float,
        rate_change: float,
        sector_shock: float,
        *,
        drift: float,
        rate_beta: float,
        sector_beta: float,
        idio_std_dev: float,
    ) -> float:
        idiosyncratic_shock: float = random.gauss(mu=0.0, sigma=idio_std_dev)
        log_return: float = drift + (rate_beta * rate_change) + (sector_beta * sector_shock) + idiosyncratic_shock
        return round(current_value * math.exp(log_return), 2)

    def advance_rate(self, rate_value: float) -> float:
        up_probability, down_probability = self.tilted_rate_probabilities(rate_value)
        draw: float = random.random()
        if draw < up_probability:
            return self.next_rate_value(rate_value, 1)

        if draw < up_probability + down_probability:
            return self.next_rate_value(rate_value, -1)

        return rate_value

    def advance_step(self, value_by_underlying_id: dict[int, float]) -> dict[int, float]:
        current_rate_value: float = value_by_underlying_id[FED_FUNDS_RATE_UNDERLYING_ID]
        rate_value: float = self.advance_rate(current_rate_value)
        rate_change: float = round(rate_value - current_rate_value, 2)
        sector_shock: float = random.gauss(mu=0.0, sigma=self.sector_std_dev)
        return {
            FED_FUNDS_RATE_UNDERLYING_ID: rate_value,
            AJARAI_UNDERLYING_ID: self.advance_company_value(
                value_by_underlying_id[AJARAI_UNDERLYING_ID],
                rate_change,
                sector_shock,
                drift=self.ajarai_drift,
                rate_beta=self.ajarai_rate_beta,
                sector_beta=self.ajarai_sector_beta,
                idio_std_dev=self.ajarai_idio_std_dev,
            ),
            THERIODIC_UNDERLYING_ID: self.advance_company_value(
                value_by_underlying_id[THERIODIC_UNDERLYING_ID],
                rate_change,
                sector_shock,
                drift=self.theriodic_drift,
                rate_beta=self.theriodic_rate_beta,
                sector_beta=self.theriodic_sector_beta,
                idio_std_dev=self.theriodic_idio_std_dev,
            ),
        }

    def next_rate_value(self, rate_value: float, num_grid_steps: int) -> float:
        return max(round(rate_value + num_grid_steps * self.rate_step, 2), 0.0)

    def tilted_rate_probabilities(self, rate_value: float) -> tuple[float, float]:
        tilt: float = self.rate_reversion_strength * (self.rate_target - rate_value)
        up_probability: float = min(max(self.rate_up_probability + tilt, 0.0), 1.0)
        down_probability: float = min(max(self.rate_down_probability - tilt, 0.0), 1.0 - up_probability)
        return up_probability, down_probability


@dataclass(frozen=True)
class OptionLeg:
    underlying_id: int
    weight: float


class OrderType(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Position:
    def __init__(self) -> None:
        self.option_quantity_by_option_id: dict[int, int] = defaultdict(int)

    def add_option_quantity(self, option_id: int, quantity: int) -> None:
        self.option_quantity_by_option_id[option_id] += quantity


@dataclass(frozen=True)
class Quote:
    bid_price: float
    bid_quantity: int
    offer_price: float
    offer_quantity: int

    def __post_init__(self) -> None:
        if self.bid_quantity <= 0 or self.offer_quantity <= 0:
            raise ValueError("Quote quantities must be positive")

        if not (0.0 <= self.bid_price <= 1.0 and 0.0 <= self.offer_price <= 1.0):
            raise ValueError("Quote prices must be between 0 and 1")

        if self.bid_price >= self.offer_price:
            raise ValueError("Quote bid price must be less than offer price")

        if any(abs(round(price * 100) - price * 100) > 1e-6 for price in (self.bid_price, self.offer_price)):
            raise ValueError("Quote prices must be in whole pennies (multiples of 0.01)")


@dataclass(frozen=True)
class Underlying:
    name: str
    underlying_id: int
    value: float

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Underlying):
            return False
        return self.underlying_id == other.underlying_id


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def norm_cdf(x: float) -> float:
    """Standard normal CDF helper."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def clamp(val: float, min_val: float = 0.0001, max_val: float = 0.9999) -> float:
    """Clamps a float value between standard bounds."""
    return max(min_val, min(max_val, val))


# ============================================================================
# MARKET MAKER CLASS
# ============================================================================

class MarketMaker:
    MAX_NET_INVENTORY: int = 2  # Pooled calibration choice from the multi-seed cap sweep

    def __init__(
        self,
        underlying_initial_state: list,
        option_initial_state: list,
        cash_balance: float,
    ) -> None:
        self.underlying_state = underlying_initial_state
        self.active_option_state = option_initial_state
        self.cash_balance = cash_balance
        self.starting_capital = cash_balance
        self.estimated_params: "MarketParameters | None" = None
        self.positions: dict[int, int] = defaultdict(int)
        self.committed_margin: float = 0.0
        self.net_delta_by_underlying: dict[int, float] = defaultdict(float)
        self.param_std_errors: dict[str, float] | None = None
        self._theo_std_cache: dict[int, float] = {}
        self._history_buffer: dict[int, list[float]] = {}
        self._steps_since_recalib: int = 0
        self.recalib_alpha: float = 0.30  # EMA weight given to fresh estimate
        self.flow_imbalance: dict[int, float] = defaultdict(float)
        self._last_quote: dict[int, tuple] = {}

        # RFQ feedback counters
        self._recent_quotes_issued: float = 0.0
        self._recent_rfq_fills: float = 0.0
        self.competitiveness_multiplier: float = 1.0
        
        # Per-step cache for compute_numerical_greeks
        self._greeks_cache: dict[int, dict] = {}
        self.risk_aversion: float = 0.008

        # Directional trade edge histories for FOK theo-shifting
        self.recent_long_pnl: list[float] = []
        self.recent_short_pnl: list[float] = []

    # ------------------------------------------------------------------
    # 1. Parameter Estimation & Warm Up
    # ------------------------------------------------------------------
    def warm_up(self, market_history: "MarketHistory") -> None:
        self._history_buffer = {
            uid: list(vals) for uid, vals in market_history.values_by_underlying_id.items()
        }
        self._steps_since_recalib = 0
        self._recalibrate()

    def _recalibrate(self) -> None:
        buf = self._history_buffer
        if not buf or FED_FUNDS_RATE_UNDERLYING_ID not in buf:
            return

        fed_prices = buf[FED_FUNDS_RATE_UNDERLYING_ID]
        ajr_prices = buf[AJARAI_UNDERLYING_ID]
        thr_prices = buf[THERIODIC_UNDERLYING_ID]

        fresh_params, fresh_std_errors = self._estimate_from_prices(fed_prices, ajr_prices, thr_prices)
        if fresh_params is None:
            return

        if self.estimated_params is None or self.param_std_errors is None:
            self.estimated_params = fresh_params
        else:
            self.estimated_params = self._blend_params(self.estimated_params, fresh_params, self.recalib_alpha)
        self.param_std_errors = fresh_std_errors

        self._theo_std_cache = {}
        self._greeks_cache = {}

    @staticmethod
    def _blend_params(old: "MarketParameters", new: "MarketParameters", alpha: float) -> "MarketParameters":
        fields_to_blend = [
            "ajarai_drift", "ajarai_idio_std_dev", "ajarai_rate_beta",
            "rate_down_probability", "rate_reversion_strength", "rate_up_probability",
            "sector_std_dev", "theriodic_drift", "theriodic_idio_std_dev",
            "theriodic_rate_beta", "rate_target",
        ]
        kwargs = {}
        for f in fields_to_blend:
            old_v = getattr(old, f)
            new_v = getattr(new, f)
            kwargs[f] = (1.0 - alpha) * old_v + alpha * new_v

        kwargs["rate_up_probability"] = max(0.01, min(0.9, kwargs["rate_up_probability"]))
        kwargs["rate_down_probability"] = max(
            0.01, min(0.9 - kwargs["rate_up_probability"], kwargs["rate_down_probability"])
        )
        kwargs["rate_reversion_strength"] = max(0.0, min(1.0, kwargs["rate_reversion_strength"]))
        kwargs["rate_target"] = max(0.0, kwargs["rate_target"])
        kwargs["ajarai_idio_std_dev"] = max(1e-6, kwargs["ajarai_idio_std_dev"])
        kwargs["theriodic_idio_std_dev"] = max(1e-6, kwargs["theriodic_idio_std_dev"])
        kwargs["sector_std_dev"] = max(1e-6, kwargs["sector_std_dev"])

        return replace(old, **kwargs)

    @staticmethod
    def _estimate_from_prices(
        fed_prices: list, ajr_prices: list, thr_prices: list
    ) -> tuple["MarketParameters", dict | None]:
        n_days = len(fed_prices)
        if n_days < 3:
            return MarketMaker._default_params(), None

        rate_step = RATE_STRIKE_GRID
        rate_diffs = [round(fed_prices[i] - fed_prices[i - 1], 4) for i in range(1, n_days)]
        rate_changes = [round(d / rate_step) for d in rate_diffs]

        levels = [fed_prices[i] for i in range(n_days - 1)]
        steps = [float(rate_changes[i]) for i in range(n_days - 1)]

        mean_level = sum(levels) / len(levels)
        mean_step = sum(steps) / len(steps)
        cov_ls = sum((levels[i] - mean_level) * (steps[i] - mean_step) for i in range(len(levels)))
        var_l = sum((l - mean_level) ** 2 for l in levels)

        if var_l > 1e-9:
            slope = cov_ls / var_l
            reversion_strength = max(0.0, min(1.0, -slope))
        else:
            slope = 0.0
            reversion_strength = 0.1

        if reversion_strength > 1e-6:
            intercept = mean_step - slope * mean_level
            rate_target = intercept / reversion_strength if reversion_strength > 1e-9 else 2.0
            rate_target = max(0.0, rate_target)
        else:
            rate_target = 2.0

        n_up = sum(1 for s in rate_changes if s >= 1)
        n_down = sum(1 for s in rate_changes if s <= -1)
        total = len(rate_changes)
        empirical_up = n_up / total
        empirical_down = n_down / total
        avg_tilt = reversion_strength * (rate_target - mean_level)
        base_up = max(0.01, min(0.9, empirical_up - avg_tilt))
        base_down = max(0.01, min(0.9 - base_up, empirical_down + avg_tilt))

        ajr_returns = [math.log(ajr_prices[i] / ajr_prices[i - 1]) for i in range(1, n_days)]
        thr_returns = [math.log(thr_prices[i] / thr_prices[i - 1]) for i in range(1, n_days)]

        rate_change_vals = rate_diffs
        ajarai_rate_beta = MarketMaker._ols_beta(rate_change_vals, ajr_returns)
        theriodic_rate_beta = MarketMaker._ols_beta(rate_change_vals, thr_returns)

        ajr_resid = [ajr_returns[i] - ajarai_rate_beta * rate_change_vals[i] for i in range(len(ajr_returns))]
        thr_resid = [thr_returns[i] - theriodic_rate_beta * rate_change_vals[i] for i in range(len(thr_returns))]

        mean_ajr = sum(ajr_resid) / len(ajr_resid)
        mean_thr = sum(thr_resid) / len(thr_resid)

        var_ajr = sum((x - mean_ajr) ** 2 for x in ajr_resid) / max(1, len(ajr_resid) - 1)
        var_thr = sum((x - mean_thr) ** 2 for x in thr_resid) / max(1, len(thr_resid) - 1)
        cov_ajr_thr = sum(
            (ajr_resid[i] - mean_ajr) * (thr_resid[i] - mean_thr) for i in range(len(ajr_resid))
        ) / max(1, len(ajr_resid) - 1)

        sector_var = max(1e-8, cov_ajr_thr)
        idio_ajr_var = max(1e-8, var_ajr - sector_var)
        idio_thr_var = max(1e-8, var_thr - sector_var)

        params = MarketParameters(
            rate_up_probability=base_up,
            rate_down_probability=base_down,
            rate_step=rate_step,
            rate_target=rate_target,
            rate_reversion_strength=reversion_strength,
            ajarai_drift=mean_ajr,
            theriodic_drift=mean_thr,
            sector_std_dev=math.sqrt(sector_var),
            ajarai_idio_std_dev=math.sqrt(idio_ajr_var),
            theriodic_idio_std_dev=math.sqrt(idio_thr_var),
            ajarai_sector_beta=1.0,
            theriodic_sector_beta=1.0,
            ajarai_rate_beta=ajarai_rate_beta,
            theriodic_rate_beta=theriodic_rate_beta,
        )

        sector_std = math.sqrt(sector_var)
        idio_ajr_std = math.sqrt(idio_ajr_var)
        idio_thr_std = math.sqrt(idio_thr_var)
        se_sector_var = sector_var * math.sqrt(2.0 / n_days)
        se_idio_ajr_var = idio_ajr_var * math.sqrt(2.0 / n_days)
        se_idio_thr_var = idio_thr_var * math.sqrt(2.0 / n_days)
        std_errors = {
            "ajarai_drift": math.sqrt(var_ajr / max(1, len(ajr_resid))),
            "theriodic_drift": math.sqrt(var_thr / max(1, len(thr_resid))),
            "sector_std_dev": se_sector_var / (2 * sector_std) if sector_std > 1e-9 else 0.0,
            "ajarai_idio_std_dev": se_idio_ajr_var / (2 * idio_ajr_std) if idio_ajr_std > 1e-9 else 0.0,
            "theriodic_idio_std_dev": se_idio_thr_var / (2 * idio_thr_std) if idio_thr_std > 1e-9 else 0.0,
            "rate_up_probability": math.sqrt(max(1e-9, base_up * (1 - base_up) / total)),
            "rate_down_probability": math.sqrt(max(1e-9, base_down * (1 - base_down) / total)),
            "ajarai_rate_beta": MarketMaker._ols_beta_se(rate_change_vals, var_ajr),
            "theriodic_rate_beta": MarketMaker._ols_beta_se(rate_change_vals, var_thr),
        }
        return params, std_errors

    @staticmethod
    def _ols_beta(x: list, y: list) -> float:
        n = len(x)
        if n < 2:
            return 0.0
        mx = sum(x) / n
        my = sum(y) / n
        var_x = sum((xi - mx) ** 2 for xi in x)
        if var_x < 1e-9:
            return 0.0
        cov_xy = sum((x[i] - mx) * (y[i] - my) for i in range(n))
        return cov_xy / var_x

    @staticmethod
    def _ols_beta_se(x: list, residual_var: float) -> float:
        n = len(x)
        if n < 3:
            return 0.0
        mx = sum(x) / n
        var_x = sum((xi - mx) ** 2 for xi in x)
        if var_x < 1e-9:
            return 0.0
        return math.sqrt(max(0.0, residual_var) / var_x)

    @staticmethod
    def _default_params() -> "MarketParameters":
        return MarketParameters(
            rate_up_probability=0.25,
            rate_down_probability=0.20,
            rate_step=0.25,
            rate_target=2.0,
            rate_reversion_strength=0.1,
            ajarai_drift=0.001,
            theriodic_drift=0.0015,
            sector_std_dev=0.02,
            ajarai_idio_std_dev=0.01,
            theriodic_idio_std_dev=0.012,
            ajarai_sector_beta=1.0,
            theriodic_sector_beta=1.0,
            ajarai_rate_beta=0.0,
            theriodic_rate_beta=0.0,
        )

    # ------------------------------------------------------------------
    # 2. Distribution & Monte Carlo Helpers
    # ------------------------------------------------------------------
    def _rate_distribution(self, market_parameters: "MarketParameters", current_rate: float, steps: int) -> dict:
        dist = {round(current_rate, 4): 1.0}
        for _ in range(steps):
            new_dist = defaultdict(float)
            for rate_value, prob in dist.items():
                if prob <= 0.0:
                    continue
                up_p, down_p = market_parameters.tilted_rate_probabilities(rate_value)
                stay_p = max(0.0, 1.0 - up_p - down_p)
                up_val = market_parameters.next_rate_value(rate_value, 1)
                down_val = market_parameters.next_rate_value(rate_value, -1)
                new_dist[round(up_val, 4)] += prob * up_p
                new_dist[round(down_val, 4)] += prob * down_p
                new_dist[round(rate_value, 4)] += prob * stay_p
            dist = new_dist
        return dist

    def _simulate_payoff_prob(
        self, market_parameters: "MarketParameters", option: "BinaryOption", spots: dict, n_sims: int = 3000
    ) -> float:
        days = option.steps_until_expiry
        hits = 0.0
        for _ in range(n_sims):
            vals = dict(spots)
            for _ in range(days):
                vals = market_parameters.advance_step(vals)
            hits += option.expiry_valuation(vals)
        return hits / n_sims

    # ------------------------------------------------------------------
    # 3. Option Valuation Logic
    # ------------------------------------------------------------------
    def price_option_from_parameters(
        self, market_parameters: "MarketParameters", option: "BinaryOption", spot_overrides: dict | None = None
    ) -> float:
        spots = {u.underlying_id: u.value for u in self.underlying_state}
        if spot_overrides:
            spots.update(spot_overrides)

        days = option.steps_until_expiry
        strike = option.strike
        underlying_ids = {leg.underlying_id for leg in option.legs}

        if days == 0:
            return option.expiry_valuation(spots)

        current_rate = spots[FED_FUNDS_RATE_UNDERLYING_ID]

        if underlying_ids == {FED_FUNDS_RATE_UNDERLYING_ID}:
            dist = self._rate_distribution(market_parameters, current_rate, days)
            return clamp(sum(p for r, p in dist.items() if r >= strike - 1e-9))

        rate_dist = self._rate_distribution(market_parameters, current_rate, days)

        total_prob = 0.0
        for r_final, branch_prob in rate_dist.items():
            if branch_prob <= 0.0:
                continue
            rate_change_total = r_final - current_rate

            log_ret_mean = {}
            log_ret_var = {}
            for uid, drift, rate_beta, sector_beta, idio_std in (
                (AJARAI_UNDERLYING_ID, market_parameters.ajarai_drift, market_parameters.ajarai_rate_beta,
                 market_parameters.ajarai_sector_beta, market_parameters.ajarai_idio_std_dev),
                (THERIODIC_UNDERLYING_ID, market_parameters.theriodic_drift, market_parameters.theriodic_rate_beta,
                 market_parameters.theriodic_sector_beta, market_parameters.theriodic_idio_std_dev),
            ):
                log_ret_mean[uid] = drift * days + rate_beta * rate_change_total
                log_ret_var[uid] = (sector_beta ** 2) * (market_parameters.sector_std_dev ** 2) * days \
                    + (idio_std ** 2) * days

            sector_var = market_parameters.sector_std_dev ** 2 * days

            if len(option.legs) == 1:
                leg = option.legs[0]
                uid = leg.underlying_id
                if uid == FED_FUNDS_RATE_UNDERLYING_ID:
                    val = 1.0 if r_final * leg.weight >= strike else 0.0
                    total_prob += branch_prob * val
                    continue
                spot = spots[uid]
                mu = log_ret_mean[uid]
                var = max(1e-12, log_ret_var[uid])
                std = math.sqrt(var)
                threshold = strike / (leg.weight * spot) if leg.weight != 0 else float("inf")
                if leg.weight > 0:
                    if threshold <= 0:
                        branch_val = 1.0
                    else:
                        d = (math.log(threshold) - mu) / std
                        branch_val = 1.0 - norm_cdf(d)
                else:
                    d = (math.log(threshold) - mu) / std if threshold > 0 else float("-inf")
                    branch_val = norm_cdf(d)
                total_prob += branch_prob * branch_val
            else:
                ajr_leg = next((l for l in option.legs if l.underlying_id == AJARAI_UNDERLYING_ID), None)
                thr_leg = next((l for l in option.legs if l.underlying_id == THERIODIC_UNDERLYING_ID), None)

                if strike == 0.0 and ajr_leg is not None and thr_leg is not None and \
                        ((thr_leg.weight > 0 > ajr_leg.weight) or (ajr_leg.weight > 0 > thr_leg.weight)):
                    mu_diff = log_ret_mean[THERIODIC_UNDERLYING_ID] - log_ret_mean[AJARAI_UNDERLYING_ID]
                    var_diff = log_ret_var[THERIODIC_UNDERLYING_ID] + log_ret_var[AJARAI_UNDERLYING_ID] - 2 * sector_var
                    var_diff = max(1e-12, var_diff)
                    std_diff = math.sqrt(var_diff)
                    log_spot_diff = math.log(spots[THERIODIC_UNDERLYING_ID] / spots[AJARAI_UNDERLYING_ID])
                    d = (log_spot_diff + mu_diff) / std_diff
                    branch_val = norm_cdf(d)
                else:
                    branch_spots = dict(spots)
                    branch_spots[FED_FUNDS_RATE_UNDERLYING_ID] = r_final
                    branch_val = self._simulate_payoff_prob(
                        market_parameters, option, branch_spots,
                        n_sims=max(200, 3000 // max(1, len(rate_dist))),
                    )

                total_prob += branch_prob * branch_val

        return clamp(total_prob)

    # ------------------------------------------------------------------
    # 4. Numerical Greeks Engine
    # ------------------------------------------------------------------
    def _per_leg_deltas(self, option: "BinaryOption", params: "MarketParameters", spots: dict) -> dict:
        underlying_ids = {leg.underlying_id for leg in option.legs}
        if underlying_ids == {FED_FUNDS_RATE_UNDERLYING_ID} and option.steps_until_expiry <= 1:
            return {FED_FUNDS_RATE_UNDERLYING_ID: 0.0}
        deltas = {}
        for leg in option.legs:
            uid = leg.underlying_id
            s_0 = spots[uid]
            h = max(abs(s_0) * 1e-4, 1e-6)
            v_plus = self.price_option_from_parameters(params, option, spot_overrides={uid: s_0 + h})
            v_minus = self.price_option_from_parameters(params, option, spot_overrides={uid: s_0 - h})
            deltas[uid] = (v_plus - v_minus) / (2.0 * h)
        return deltas

    def _option_theta(self, params: "MarketParameters", option: "BinaryOption") -> float:
        if option.steps_until_expiry < 1:
            return 0.0
        v0 = self.price_option_from_parameters(params, option)
        v1 = self.price_option_from_parameters(params, option.advance_step())
        return v1 - v0

    def compute_numerical_greeks(self, option: "BinaryOption") -> dict:
        if option.option_id in self._greeks_cache:
            return self._greeks_cache[option.option_id]

        params = self.estimated_params or self._default_params()
        spots = {u.underlying_id: u.value for u in self.underlying_state}
        v_0 = self.price_option_from_parameters(params, option)

        underlying_ids = {leg.underlying_id for leg in option.legs}
        is_pure_fed_1d = underlying_ids == {FED_FUNDS_RATE_UNDERLYING_ID} and option.steps_until_expiry <= 1
        theta = self._option_theta(params, option)

        if is_pure_fed_1d:
            result = {"theo": v_0, "delta": 0.0, "gamma": 4.0, "theta": theta}
            self._greeks_cache[option.option_id] = result
            return result

        leg_deltas = self._per_leg_deltas(option, params, spots)
        total_delta = sum(leg_deltas.values())

        total_gamma = 0.0
        for leg in option.legs:
            uid = leg.underlying_id
            s_0 = spots[uid]
            h = max(abs(s_0) * 1e-4, 1e-6)
            v_plus = self.price_option_from_parameters(params, option, spot_overrides={uid: s_0 + h})
            v_minus = self.price_option_from_parameters(params, option, spot_overrides={uid: s_0 - h})
            total_gamma += abs((v_plus - 2.0 * v_0 + v_minus) / (h ** 2))

        result = {"theo": v_0, "delta": total_delta, "gamma": total_gamma, "theta": theta}
        self._greeks_cache[option.option_id] = result
        return result

    def price_option(self, option: "BinaryOption") -> float:
        return self.compute_numerical_greeks(option)["theo"]

    # ------------------------------------------------------------------
    # 4b. Confidence & Margin Calculations
    # ------------------------------------------------------------------
    @staticmethod
    def _clip_param(field_name: str, value: float) -> float:
        if field_name in ("rate_up_probability", "rate_down_probability"):
            return max(0.01, min(0.9, value))
        if field_name in ("ajarai_idio_std_dev", "theriodic_idio_std_dev", "sector_std_dev"):
            return max(1e-6, value)
        return value

    def _theo_std(self, option: "BinaryOption") -> float:
        if option.option_id in self._theo_std_cache:
            return self._theo_std_cache[option.option_id]

        if self.estimated_params is None or self.param_std_errors is None:
            self._theo_std_cache[option.option_id] = 0.15
            return 0.15

        params = self.estimated_params
        v0 = self.price_option_from_parameters(params, option)
        total_var = 0.0
        for field_name, se in self.param_std_errors.items():
            if se <= 0:
                continue
            bumped_val = self._clip_param(field_name, getattr(params, field_name) + se)
            try:
                bumped_params = replace(params, **{field_name: bumped_val})
            except ValueError:
                continue
            v_bump = self.price_option_from_parameters(bumped_params, option)
            total_var += (v_bump - v0) ** 2

        result = math.sqrt(total_var)
        self._theo_std_cache[option.option_id] = result
        return result

    def _confidence(self, theo_std: float) -> float:
        return 1.0 / (1.0 + 40.0 * theo_std)

    def _max_loss_per_contract(self, price: float, side_is_buy: bool) -> float:
        return price if side_is_buy else (1.0 - price)

    def _available_margin(self) -> float:
        buffer_frac = 0.15
        return max(0.0, self.cash_balance - self.committed_margin - buffer_frac * self.starting_capital)

    def _max_affordable_qty(self, price: float, side_is_buy: bool, cap: int = 25) -> int:
        loss_per = self._max_loss_per_contract(price, side_is_buy)
        if loss_per <= 1e-9:
            return cap
        affordable = int(self._available_margin() // loss_per)
        return max(0, min(cap, affordable))

    @staticmethod
    def _clamp_greek(value: float, bound: float = 3.0) -> float:
        return max(-bound, min(bound, value))

    def _recompute_portfolio_deltas(self) -> None:
        params = self.estimated_params or self._default_params()
        spots = {u.underlying_id: u.value for u in self.underlying_state}
        option_by_id = {o.option_id: o for o in self.active_option_state}
        net: dict[int, float] = defaultdict(float)
        for opt_id, qty in self.positions.items():
            if qty == 0:
                continue
            option = option_by_id.get(opt_id)
            if option is None:
                continue
            leg_deltas = self._per_leg_deltas(option, params, spots)
            for uid, d in leg_deltas.items():
                net[uid] += qty * self._clamp_greek(d)
        self.net_delta_by_underlying = net

    # ------------------------------------------------------------------
    # 5. Quoting Engine (RFQ)
    # ------------------------------------------------------------------
    def quote(self, option: "BinaryOption", counterparty_id: int) -> "Quote":
        """Quote a fixed-width market centered on the rounded fair probability."""
        theo = self.compute_numerical_greeks(option)["theo"]
        opt_id = option.option_id
        # Quotes trade in pennies. Center on the nearest valid penny, then keep
        # four ticks on either side (an 8-cent total spread), except at 0/1.
        center_tick = int(round(theo * 100))
        bid_tick = max(0, center_tick - 4)
        offer_tick = min(100, center_tick + 4)
        if offer_tick <= bid_tick:
            if bid_tick == 0:
                offer_tick = 1
            else:
                bid_tick = offer_tick - 1
        bid, offer = bid_tick / 100.0, offer_tick / 100.0

        inventory = self.positions[opt_id]
        bid_room = max(1, self.MAX_NET_INVENTORY - inventory)
        offer_room = max(1, self.MAX_NET_INVENTORY + inventory)
        bid_qty = max(1, min(
            self._max_affordable_qty(bid, side_is_buy=True, cap=bid_room),
            bid_room,
        ))
        offer_qty = max(1, min(
            self._max_affordable_qty(offer, side_is_buy=False, cap=offer_room),
            offer_room,
        ))

        self._last_quote[opt_id] = (bid, offer)
        return Quote(bid_price=bid, bid_quantity=bid_qty, offer_price=offer, offer_quantity=offer_qty)

    def quote_regime_based(self, option: "BinaryOption", counterparty_id: int) -> "Quote":
        """Original asymmetric regime-based RFQ strategy, retained for comparison."""
        greeks = self.compute_numerical_greeks(option)
        theo = greeks["theo"]

        opt_id = option.option_id
        inventory = self.positions[opt_id]

        theo_std = self._theo_std(option)
        confidence = self._confidence(theo_std)
        effective_cap = max(5, int(self.MAX_NET_INVENTORY * (0.5 + 1.5 * confidence)))
        inv_room = max(1, effective_cap - abs(inventory))

        if 0.0 < theo < 0.3:
            # LONG BIAS REGIME
            bid = round(max(0.0, theo - 0.002), 2)
            offer = round(min(theo + 0.4, 1.0), 2)
            self.MAX_NET_INVENTORY = 20

        elif 0.7 <= theo <= 1.0:
            # SHORT BIAS REGIME
            bid = round(max(theo - 0.99, 0.0), 2)
            offer = round(min(theo + 0.006, 1.0), 2)
            self.MAX_NET_INVENTORY = 9

        else:
            # NEUTRAL REGIME (0.3 <= theo < 0.7)
            bid = round(max(0.0, theo - 0.035), 2)
            offer = round(min(theo + 0.035, 1.0), 2)
            self.MAX_NET_INVENTORY = 12

        if offer <= bid:
            offer = round(min(1.0, bid + 0.01), 2)

        bid_qty = max(1, min(self._max_affordable_qty(bid, side_is_buy=True, cap=inv_room), inv_room))
        offer_qty = max(1, min(self._max_affordable_qty(offer, side_is_buy=False, cap=inv_room), inv_room))

        self._last_quote[opt_id] = (bid, offer)
        return Quote(bid_price=bid, bid_quantity=bid_qty, offer_price=offer, offer_quantity=offer_qty)

    # ------------------------------------------------------------------
    # 6. FOK Order Handler
    # ------------------------------------------------------------------
    def respond_to_fok(self, option: "BinaryOption", fok_order: "FokOrder") -> bool:
        greeks = self.compute_numerical_greeks(option)
        base_theo = greeks["theo"]

        # OrderType is the COUNTERPARTY'S side:
        # Counterparty SELL -> Bot BUY (Long)
        # Counterparty BUY  -> Bot SELL (Short)
        is_bot_buying = fok_order.order_type == OrderType.SELL

        # --- DIRECTIONAL THEO ADJUSTMENT ---
        relevant_history = self.recent_long_pnl if is_bot_buying else self.recent_short_pnl
        theo_adjustment = 0.0

        if len(relevant_history) == 5:
            total_pnl = sum(relevant_history)
            shift_magnitude = min(0.05, 0.01 + abs(total_pnl) * 0.1)

            if is_bot_buying:
                # Evaluating a BUY:
                # Past buys profited -> shift theo UP to bid higher and buy more.
                # Past buys lost     -> shift theo DOWN to bid lower and back away.
                theo_adjustment = shift_magnitude if total_pnl > 0 else -shift_magnitude
            else:
                # Evaluating a SELL:
                # Past sells profited -> shift theo DOWN to offer lower and sell more.
                # Past sells lost     -> shift theo UP to offer higher and back away.
                theo_adjustment = -shift_magnitude if total_pnl > 0 else shift_magnitude

        theo = clamp(base_theo + theo_adjustment)

        # --- SURVIVAL & MARGIN CHECKS ---
        current_inv = self.positions[option.option_id]
        projected_inv = current_inv + fok_order.quantity if is_bot_buying else current_inv - fok_order.quantity

        if abs(projected_inv) > self.MAX_NET_INVENTORY:
            return False

        affordable = self._max_affordable_qty(fok_order.price, side_is_buy=is_bot_buying, cap=fok_order.quantity)
        if affordable < fok_order.quantity:
            return False

        # --- REGIME EVALUATION (Using Directionally Adjusted Theo) ---
        if 0.0 < theo < 0.3:
            if not is_bot_buying:
                return False
            target_bid = round(max(0.0, theo - 0.002), 2)
            return fok_order.price <= target_bid

        elif 0.7 <= theo <= 1.0:
            if is_bot_buying:
                return False
            target_offer = round(min(theo + 0.006, 1.0), 2)
            return fok_order.price >= target_offer

        else:
            target_bid = round(max(0.0, theo - 0.035), 2)
            target_offer = round(min(theo + 0.035, 1.0), 2)

            if is_bot_buying:
                return fok_order.price <= target_bid
            else:
                return fok_order.price >= target_offer

    # ------------------------------------------------------------------
    # 7. Inventory Tracking & State Advancement
    # ------------------------------------------------------------------
    def on_trade(self, option: "BinaryOption", price: float, quantity: int, counterparty_id: int) -> None:
        opt_id = option.option_id
        self.positions[opt_id] += quantity

        # Calculate edge/PnL relative to fair value at execution
        greeks = self.compute_numerical_greeks(option)
        base_theo = greeks["theo"]
        
        # Positive quantity means bot bought; negative means bot sold
        trade_pnl = (base_theo - price) * quantity

        # Route PnL to the matching directional history list
        if quantity > 0:
            self.recent_long_pnl.append(trade_pnl)
            if len(self.recent_long_pnl) > 5:
                self.recent_long_pnl.pop(0)
        elif quantity < 0:
            self.recent_short_pnl.append(trade_pnl)
            if len(self.recent_short_pnl) > 5:
                self.recent_short_pnl.pop(0)

        # Track realized RFQ wins specifically
        last = self._last_quote.get(opt_id)
        if last is not None:
            last_bid, last_offer = last
            tick = 0.01
            hit_offer = abs(price - last_offer) <= tick
            hit_bid = abs(price - last_bid) <= tick
            if hit_offer or hit_bid:
                self._recent_rfq_fills += 1

        self._recompute_portfolio_deltas()

    def on_step_advance(self, new_underlying_state, new_option_state) -> None:
        self.underlying_state = new_underlying_state
        self.active_option_state = new_option_state
        self.committed_margin = 0.0
        self._greeks_cache = {}

        for u in new_underlying_state:
            if u.underlying_id in self._history_buffer:
                self._history_buffer[u.underlying_id].append(u.value)

        self._steps_since_recalib += 1
        RECALIBRATION_INTERVAL = 3
        if self._steps_since_recalib >= RECALIBRATION_INTERVAL and self._history_buffer:
            self._recalibrate()
            self._steps_since_recalib = 0
        else:
            self._theo_std_cache = {}

        for opt_id in list(self.flow_imbalance.keys()):
            self.flow_imbalance[opt_id] *= 0.85

        if self._recent_quotes_issued > 3:
            win_rate = self._recent_rfq_fills / max(1e-6, self._recent_quotes_issued)
            if win_rate < 0.12:
                self.competitiveness_multiplier *= 0.94
            elif win_rate > 0.55:
                self.competitiveness_multiplier *= 1.04
            self.competitiveness_multiplier = max(0.35, min(1.6, self.competitiveness_multiplier))

        self._recent_quotes_issued *= 0.85
        self._recent_rfq_fills *= 0.85

        self._recompute_portfolio_deltas()

    @property
    def name(self) -> str:
        return "Quant_Greeks_MM"
