#!/usr/bin/env python3
"""
mayhem_analysis_plan_backtester_v2.py

Extends mayhem_analysis_plan_backtester.py with a few additional "reporting" subcommands
so you can answer the questions in "Mayhem Mode Analysis: Dataset & Strategy Guide"
without storing gigantic per-(mint,param) result tables.

What this script does (engineering only; no dataset conclusions are produced):
- summarize:  one row per mint (bot response + peak/trough timing)
- grid:       aggregated strategy metrics over a TP/SL/timeout grid
- simulate:   per-mint results for ONE concrete strategy (so you can inspect distributions)
- reach:      "does it ever reach TP%" table grouped by observed dev initial buy buckets

Core modeling assumption (from the guide):
- "Dev position value at time t" is proxied by a pool series (see --pool-mode).
  - full: pool_sol_after(t)
  - mayhem_only: entry baseline + cumulative MAYHEM pool_sol_change after entry
- Gross PnL for an exit at time t is pool_series(t) - pool_series(entry).
- Net PnL subtracts the creation fee (default 0.021 SOL). Trade/network fees are not modeled
  unless you incorporate them separately.

Notes for large datasets (1M+ rows):
- Only required columns are loaded by default.
- Data is sorted by (mint, slot) once, then processed mint-by-mint using slices.

"""

from __future__ import annotations

import argparse
import heapq
import itertools
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


DEV_ACTOR = "😈 DEV"
MAYHEM_ACTOR = "🤖 MAYHEM"

DEFAULT_CREATION_FEE_SOL = 0.021
DEFAULT_SLOT_MS = 400.0  # 1 slot ≈ 400ms (approx, from your guide)


@dataclass(frozen=True)
class StrategyParams:
    buy_sol: float
    tp_pct: float
    sl_pct: float
    timeout_slots: int
    soft_stops: str = ""
    soft_a_slots: int = 0
    soft_b_window_slots: int = 0
    soft_b_ratio: float = 0.0


@dataclass
class OnlineMoments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def pvariance(self) -> float:
        return self.m2 / self.n if self.n else 0.0

    @property
    def pstdev(self) -> float:
        return math.sqrt(self.pvariance)


@dataclass
class RollingSumWindow:
    size: int
    values: Deque[float] = field(default_factory=deque)
    total: float = 0.0
    min_total: Optional[float] = None

    def update(self, x: float) -> None:
        self.values.append(x)
        self.total += x
        if len(self.values) > self.size:
            self.total -= self.values.popleft()
        if len(self.values) == self.size:
            if self.min_total is None or self.total < self.min_total:
                self.min_total = self.total


@dataclass
class ExtendedAgg:
    start_sol: float
    order_by: str
    worst_windows: Sequence[int]
    scale_mode: str
    slot_ms: float

    n: int = 0
    wins: int = 0
    losses: int = 0
    tp: int = 0
    sl: int = 0
    to: int = 0
    end: int = 0
    sum_gross: float = 0.0
    sum_net_raw: float = 0.0
    sum_net: float = 0.0
    slippage_cost_total: float = 0.0
    sum_hold_slots: float = 0.0
    sum_mae_pct: float = 0.0
    sum_mfe_pct: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    denom_sol_slots: float = 0.0

    pnl_moments: OnlineMoments = field(default_factory=OnlineMoments)
    ret_moments: OnlineMoments = field(default_factory=OnlineMoments)
    downside_sq_sum: float = 0.0

    roll_windows: Dict[int, RollingSumWindow] = field(init=False)
    pending: List[Tuple[int, str, float]] = field(default_factory=list)

    trade_index: int = 0
    equity: float = field(init=False)
    peak_equity: float = field(init=False)
    peak_trade_index: int = 0
    max_drawdown_sol: float = 0.0
    max_dd_peak_equity: float = field(init=False)
    max_dd_peak_trade_index: int = 0
    max_dd_trough_equity: float = field(init=False)
    max_dd_trough_trade_index: int = 0
    max_dd_recovery_trade_index: Optional[int] = None

    in_dd: bool = False
    ep_peak_equity: float = field(init=False)
    ep_peak_trade_index: int = 0
    ep_trough_equity: float = field(init=False)
    ep_trough_trade_index: int = 0
    ep_depths_sol_sum: float = 0.0
    ep_depths_pct_sum: float = 0.0
    ep_depths_count: int = 0

    underwater_trades: int = 0
    underwater_periods_sum: int = 0
    underwater_periods_count: int = 0
    current_underwater_len: int = 0
    max_underwater_length: int = 0

    dd_sq_sum: float = 0.0
    longest_losing_streak: int = 0
    current_losing_streak: int = 0

    def __post_init__(self) -> None:
        self.equity = float(self.start_sol)
        self.peak_equity = float(self.start_sol)
        self.max_dd_peak_equity = float(self.start_sol)
        self.max_dd_trough_equity = float(self.start_sol)
        self.ep_peak_equity = float(self.start_sol)
        self.ep_trough_equity = float(self.start_sol)
        self.roll_windows = {int(w): RollingSumWindow(int(w)) for w in self.worst_windows if int(w) > 0}

    def update_trade(
        self,
        *,
        net_pnl: float,
        net_pnl_raw: float,
        slippage_cost: float,
        gross_pnl: float,
        hold_slots: int,
        mae_pct: float,
        mfe_pct: float,
        exit_reason: str,
        buy_sol: float,
        entry_slot: int,
        exit_slot: int,
        mint: str,
    ) -> None:
        self.n += 1
        self.sum_gross += gross_pnl
        self.sum_net_raw += net_pnl_raw
        self.sum_net += net_pnl
        self.slippage_cost_total += slippage_cost
        self.sum_hold_slots += float(hold_slots)
        self.sum_mae_pct += mae_pct
        self.sum_mfe_pct += mfe_pct

        if net_pnl > 0:
            self.wins += 1
            self.gross_profit += net_pnl
        elif net_pnl < 0:
            self.losses += 1
            self.gross_loss += net_pnl

        if exit_reason == "TP":
            self.tp += 1
        elif exit_reason == "SL":
            self.sl += 1
        elif exit_reason == "TIMEOUT":
            self.to += 1
        else:
            self.end += 1

        if self.scale_mode == "linear":
            if hold_slots > 0 and buy_sol > 0 and math.isfinite(buy_sol):
                self.denom_sol_slots += float(buy_sol) * float(hold_slots)

        self.pnl_moments.update(net_pnl)
        if self.start_sol > 0:
            ret = net_pnl / float(self.start_sol)
            self.ret_moments.update(ret)
            if ret < 0:
                self.downside_sq_sum += ret * ret

        if self.order_by == "exit_slot":
            heapq.heappush(self.pending, (int(exit_slot), str(mint), float(net_pnl)))
            self.flush_pending(int(entry_slot))
        else:
            self._process_sequence_trade(float(net_pnl))

    def flush_pending(self, up_to_slot: Optional[int]) -> None:
        if self.order_by != "exit_slot":
            return
        limit = float("inf") if up_to_slot is None else int(up_to_slot)
        while self.pending and self.pending[0][0] <= limit:
            _exit_slot, _mint, pnl = heapq.heappop(self.pending)
            self._process_sequence_trade(float(pnl))

    def _record_underwater_period(self, length: int) -> None:
        self.underwater_periods_sum += int(length)
        self.underwater_periods_count += 1
        if length > self.max_underwater_length:
            self.max_underwater_length = int(length)

    def _process_sequence_trade(self, pnl: float) -> None:
        self.trade_index += 1
        equity = self.equity + pnl
        self.equity = equity

        if pnl < 0:
            self.current_losing_streak += 1
            if self.current_losing_streak > self.longest_losing_streak:
                self.longest_losing_streak = self.current_losing_streak
        else:
            self.current_losing_streak = 0

        if equity > self.peak_equity:
            self.peak_equity = equity
            self.peak_trade_index = self.trade_index
            if self.current_underwater_len:
                self._record_underwater_period(self.current_underwater_len)
                self.current_underwater_len = 0
            dd_pct = 0.0
        else:
            self.underwater_trades += 1
            self.current_underwater_len += 1
            dd_pct = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0

        self.dd_sq_sum += dd_pct * dd_pct

        if not self.in_dd and equity < self.peak_equity:
            self.in_dd = True
            self.ep_peak_equity = self.peak_equity
            self.ep_peak_trade_index = self.peak_trade_index
            self.ep_trough_equity = equity
            self.ep_trough_trade_index = self.trade_index
        elif self.in_dd:
            if equity < self.ep_trough_equity:
                self.ep_trough_equity = equity
                self.ep_trough_trade_index = self.trade_index
            if equity >= self.ep_peak_equity:
                depth = self.ep_peak_equity - self.ep_trough_equity
                self.ep_depths_sol_sum += depth
                self.ep_depths_pct_sum += (depth / self.ep_peak_equity) if self.ep_peak_equity > 0 else 0.0
                self.ep_depths_count += 1
                self.in_dd = False

        drawdown_sol = self.peak_equity - equity
        if drawdown_sol > self.max_drawdown_sol:
            self.max_drawdown_sol = drawdown_sol
            self.max_dd_peak_equity = self.peak_equity
            self.max_dd_peak_trade_index = self.peak_trade_index
            self.max_dd_trough_equity = equity
            self.max_dd_trough_trade_index = self.trade_index
            self.max_dd_recovery_trade_index = None

        if (
            self.max_drawdown_sol > 0
            and self.max_dd_recovery_trade_index is None
            and self.trade_index > self.max_dd_trough_trade_index
            and equity >= self.max_dd_peak_equity
        ):
            self.max_dd_recovery_trade_index = self.trade_index

        for window in self.roll_windows.values():
            window.update(pnl)

    def finalize(self, launches_per_day: Sequence[int]) -> Dict[str, object]:
        if self.order_by == "exit_slot":
            self.flush_pending(None)

        if self.current_underwater_len:
            self._record_underwater_period(self.current_underwater_len)
            self.current_underwater_len = 0

        if self.in_dd and self.ep_peak_equity > self.ep_trough_equity:
            depth = self.ep_peak_equity - self.ep_trough_equity
            self.ep_depths_sol_sum += depth
            self.ep_depths_pct_sum += (depth / self.ep_peak_equity) if self.ep_peak_equity > 0 else 0.0
            self.ep_depths_count += 1
            self.in_dd = False

        n = self.n
        if n == 0:
            return {}

        avg_hold_slots = self.sum_hold_slots / n
        avg_hold_seconds = avg_hold_slots * (float(self.slot_ms) / 1000.0)

        avg_net_pnl_raw = self.sum_net_raw / n
        avg_net_pnl = self.sum_net / n
        avg_slippage_cost = self.slippage_cost_total / n

        breakeven = n - self.wins - self.losses
        win_rate = self.wins / n if n else None

        profit_factor = None
        if self.gross_loss < 0:
            profit_factor = self.gross_profit / (-self.gross_loss)
        elif self.gross_profit > 0:
            profit_factor = float("inf")
        omega_ratio_0 = profit_factor

        avg_win_sol = (self.gross_profit / self.wins) if self.wins else None
        avg_loss_sol = (self.gross_loss / self.losses) if self.losses else None
        avg_loss_sol_abs = (-avg_loss_sol) if avg_loss_sol is not None else None
        win_loss_ratio = (
            (avg_win_sol / avg_loss_sol_abs) if avg_win_sol is not None and avg_loss_sol_abs and avg_loss_sol_abs > 0 else None
        )

        percent_time_underwater = (self.underwater_trades / n) if n else None
        ulcer_index = math.sqrt(self.dd_sq_sum / n) if n else None
        avg_underwater_length = (
            (self.underwater_periods_sum / self.underwater_periods_count) if self.underwater_periods_count else None
        )
        max_underwater_length = self.max_underwater_length if self.underwater_periods_count else None

        max_drawdown_pct = (self.max_drawdown_sol / self.max_dd_peak_equity) if self.max_dd_peak_equity > 0 else 0.0
        max_drawdown_recovery_trades = (
            (self.max_dd_recovery_trade_index - self.max_dd_trough_trade_index)
            if self.max_dd_recovery_trade_index is not None
            else None
        )

        avg_drawdown_sol = (self.ep_depths_sol_sum / self.ep_depths_count) if self.ep_depths_count else None
        avg_drawdown_pct = (self.ep_depths_pct_sum / self.ep_depths_count) if self.ep_depths_count else None

        sharpe_ratio = None
        sortino_ratio = None
        sharpe_annualized: Dict[int, float] = {}
        sortino_annualized: Dict[int, float] = {}
        if n and self.start_sol > 0:
            mean_r = self.ret_moments.mean
            std_r = self.ret_moments.pstdev if self.ret_moments.n > 1 else 0.0
            if std_r > 0:
                sharpe_ratio = mean_r / std_r
            downside = math.sqrt(self.downside_sq_sum / n) if n else 0.0
            if downside > 0:
                sortino_ratio = mean_r / downside
            for lpd in launches_per_day:
                tpy = float(lpd) * 365.0
                scale = math.sqrt(tpy) if tpy > 0 else 0.0
                if sharpe_ratio is not None and math.isfinite(sharpe_ratio):
                    sharpe_annualized[int(lpd)] = sharpe_ratio * scale
                if sortino_ratio is not None and math.isfinite(sortino_ratio):
                    sortino_annualized[int(lpd)] = sortino_ratio * scale

        calmar_by_lpd: Dict[int, float] = {}
        avg_net_pnl_per_launch = avg_net_pnl if n else None
        if avg_net_pnl_per_launch is not None and max_drawdown_pct > 0:
            for lpd in launches_per_day:
                annual_net = avg_net_pnl_per_launch * float(int(lpd)) * 365.0
                annual_return = annual_net / float(self.start_sol)
                calmar_by_lpd[int(lpd)] = annual_return / max_drawdown_pct
        elif avg_net_pnl_per_launch is not None and max_drawdown_pct == 0 and self.sum_net > 0:
            for lpd in launches_per_day:
                calmar_by_lpd[int(lpd)] = float("inf")

        recovery_time_by_lpd: Dict[int, Dict[str, float]] = {}
        if max_drawdown_recovery_trades is not None:
            rt = float(max_drawdown_recovery_trades)
            for lpd in launches_per_day:
                lpd_f = float(int(lpd))
                days = (rt / lpd_f) if lpd_f > 0 else 0.0
                recovery_time_by_lpd[int(lpd)] = {"days": days, "hours": days * 24.0}

        ruin_prob_brownian = None
        bankroll_for_ruin_prob_1pct_brownian = None
        bankroll_for_ruin_prob_0p1pct_brownian = None
        if n:
            mu = self.pnl_moments.mean
            var = self.pnl_moments.pvariance if self.pnl_moments.n > 1 else 0.0
            if var == 0.0:
                ruin_prob_brownian = 1.0 if mu <= 0 else 0.0
            elif mu <= 0:
                ruin_prob_brownian = 1.0
            else:
                b = float(self.start_sol)
                ruin_prob_brownian = math.exp(-2.0 * mu * b / var)
                bankroll_for_ruin_prob_1pct_brownian = -(var / (2.0 * mu)) * math.log(0.01)
                bankroll_for_ruin_prob_0p1pct_brownian = -(var / (2.0 * mu)) * math.log(0.001)

        kelly_binary = None
        kelly_mean_var = None
        if avg_win_sol is not None and avg_loss_sol_abs is not None and avg_loss_sol_abs > 0 and n:
            p = self.wins / n
            q = 1.0 - p
            b = float(avg_win_sol / avg_loss_sol_abs)
            kelly_binary = max(0.0, min(1.0, p - (q / b))) if b > 0 else 0.0
        if n:
            mu = self.pnl_moments.mean
            var = self.pnl_moments.pvariance if self.pnl_moments.n > 1 else 0.0
            if var > 0:
                f = (mu * float(self.start_sol)) / var
                kelly_mean_var = max(0.0, min(1.0, f))

        worst_rolling_pnl = {w: window.min_total for w, window in self.roll_windows.items()}

        throughput = None
        if self.scale_mode == "linear" and self.denom_sol_slots > 0:
            slot_minutes = float(self.slot_ms) / 60000.0
            denom_sol_minutes = self.denom_sol_slots * slot_minutes
            throughput = {
                "pnl_per_sol_slot_total": self.sum_net / self.denom_sol_slots,
                "pnl_per_sol_minute_total": (self.sum_net / denom_sol_minutes) if denom_sol_minutes > 0 else None,
                "denom_sol_slots": self.denom_sol_slots,
            }

        projections: Dict[int, Dict[str, float]] = {}
        if avg_net_pnl_per_launch is not None:
            for lpd in launches_per_day:
                per_day = avg_net_pnl_per_launch * float(int(lpd))
                projections[int(lpd)] = {"per_day": per_day, "per_hour": per_day / 24.0}

        return {
            "n_trades": n,
            "wins": self.wins,
            "losses": self.losses,
            "breakeven": breakeven,
            "win_rate": win_rate,
            "tp_rate": (self.tp / n) if n else None,
            "sl_rate": (self.sl / n) if n else None,
            "timeout_rate": (self.to / n) if n else None,
            "avg_net_pnl_raw": avg_net_pnl_raw,
            "avg_net_pnl": avg_net_pnl,
            "total_net_pnl_raw": self.sum_net_raw,
            "total_net_pnl": self.sum_net,
            "avg_gross_pnl": (self.sum_gross / n),
            "avg_hold_slots": avg_hold_slots,
            "avg_hold_seconds": avg_hold_seconds,
            "avg_mae_pct": (self.sum_mae_pct / n),
            "avg_mfe_pct": (self.sum_mfe_pct / n),
            "slippage_cost_total": self.slippage_cost_total,
            "avg_slippage_cost": avg_slippage_cost,
            "profit_factor": profit_factor,
            "omega_ratio_0": omega_ratio_0,
            "avg_win_sol": avg_win_sol,
            "avg_loss_sol": avg_loss_sol,
            "avg_loss_sol_abs": avg_loss_sol_abs,
            "win_loss_ratio": win_loss_ratio,
            "longest_losing_streak": self.longest_losing_streak,
            "percent_time_underwater": percent_time_underwater,
            "avg_underwater_length": avg_underwater_length,
            "max_underwater_length": max_underwater_length,
            "ulcer_index": ulcer_index,
            "max_drawdown_sol": self.max_drawdown_sol,
            "max_drawdown_pct": max_drawdown_pct,
            "max_drawdown_recovery_trades": max_drawdown_recovery_trades,
            "avg_drawdown_sol": avg_drawdown_sol,
            "avg_drawdown_pct": avg_drawdown_pct,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "sharpe_annualized": sharpe_annualized,
            "sortino_annualized": sortino_annualized,
            "calmar_by_lpd": calmar_by_lpd,
            "recovery_time_by_lpd": recovery_time_by_lpd,
            "worst_rolling_pnl": worst_rolling_pnl,
            "throughput": throughput,
            "projections": projections,
            "ruin_prob_brownian": ruin_prob_brownian,
            "bankroll_for_ruin_prob_1pct_brownian": bankroll_for_ruin_prob_1pct_brownian,
            "bankroll_for_ruin_prob_0p1pct_brownian": bankroll_for_ruin_prob_0p1pct_brownian,
            "kelly_binary": kelly_binary,
            "kelly_mean_var": kelly_mean_var,
        }

# ----------------------------
# IO + mint slicing
# ----------------------------

def load_events_csv(path: str, extra_cols: Sequence[str] = ()) -> pd.DataFrame:
    """
    Load only the columns needed for analysis.

    Expected base columns (from your export script):
      slot, mint, actor, action_type, pool_sol_before, pool_sol_after, pool_sol_change,
      tx_pnl_excl_network_fee

    Pass extra_cols if you want e.g. block_time or tx_pnl_incl_network_fee.
    """
    base_usecols = [
        "slot",
        "mint",
        "actor",
        "action_type",
        "pool_sol_before",
        "pool_sol_after",
        "pool_sol_change",
        "tx_pnl_excl_network_fee",
    ]
    usecols = list(dict.fromkeys([*base_usecols, *extra_cols]))  # stable unique

    dtypes = {
        "slot": "int64",
        "mint": "string",
        "actor": "string",
        "action_type": "string",
        "pool_sol_before": "float64",
        "pool_sol_after": "float64",
        "pool_sol_change": "float64",
        "tx_pnl_excl_network_fee": "float64",
    }
    # best-effort dtype assignment for extras
    for c in extra_cols:
        if c == "tx_pnl_incl_network_fee":
            dtypes[c] = "float64"
        else:
            dtypes.setdefault(c, "string")

    df = pd.read_csv(path, usecols=usecols, dtype=dtypes, encoding="utf-8")
    df["action_type"] = df["action_type"].str.upper()
    return df


def sort_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure grouping by mint and within-mint time ordering by slot.
    """
    return df.sort_values(["mint", "slot"], kind="mergesort").reset_index(drop=True)


def iter_mint_slices(df_sorted: pd.DataFrame) -> Iterator[Tuple[str, slice]]:
    """Yield (mint, slice) for df sorted by mint then slot."""
    mints = df_sorted["mint"].to_numpy()
    if len(mints) == 0:
        return
    boundaries = np.flatnonzero(mints[1:] != mints[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(mints)]])
    for s, e in zip(starts, ends):
        yield str(mints[int(s)]), slice(int(s), int(e))


# ----------------------------
# Entry detection + bot/dev helpers
# ----------------------------

def find_launch_index(mdf: pd.DataFrame, eps: float = 1e-9) -> int:
    """
    Heuristic: "launch" = first DEV BUY with pool_sol_before ~ 0.
    If missing, fallback to first row.
    """
    is_dev_buy = (
        (mdf["actor"] == DEV_ACTOR)
        & (mdf["action_type"] == "BUY")
        & (mdf["pool_sol_before"].abs() <= eps)
    )
    idxs = np.flatnonzero(is_dev_buy.to_numpy())
    return int(idxs[0]) if len(idxs) else 0


def find_first_dev_buy_sol(mdf: pd.DataFrame) -> float:
    """
    Robust dev initial buy size estimator:
    - Find the earliest DEV BUY in the mint slice.
    - Use -tx_pnl_excl_network_fee (buys are negative pnl) as the SOL amount spent.
    """
    mask = (mdf["actor"] == DEV_ACTOR) & (mdf["action_type"] == "BUY")
    idxs = np.flatnonzero(mask.to_numpy())
    if len(idxs) == 0:
        return float("nan")
    i = int(idxs[0])
    try:
        return float(-mdf.iloc[i]["tx_pnl_excl_network_fee"])
    except Exception:
        return float("nan")


# ----------------------------
# Pool-mode helpers
# ----------------------------

def compute_retrievable_pool_mayhem_only(mdf: pd.DataFrame, entry_i: int) -> np.ndarray:
    """
    Returns an array same length as mdf with the DEV-retrievable pool value at each event:
    - Baseline at entry_i is the actual pool_sol_after at entry_i (DEV launch buy pool).
    - After entry_i, ONLY apply pool_sol_change from actor == MAYHEM_ACTOR.
    - Ignore actor == "🙂 OTHER" entirely (both buys and sells).
    - Also ignore any DEV actions after entry_i (since we are simulating an alternate DEV path).
    - Clamp at >= 0.0 because pool cannot go negative in a "no-other-liquidity" world.
    """
    pool_full = mdf["pool_sol_after"].to_numpy(np.float64)
    if len(pool_full) == 0:
        return pool_full
    changes = mdf["pool_sol_change"].to_numpy(np.float64)
    actors = mdf["actor"].to_numpy()

    mayhem_changes = np.where(actors == MAYHEM_ACTOR, changes, 0.0)
    mayhem_cum = np.cumsum(mayhem_changes)

    baseline = float(pool_full[entry_i])
    offset = float(mayhem_cum[entry_i])  # remove any pre-entry mayhem activity (rare but safe)

    retrievable = pool_full.copy()
    retrievable[entry_i:] = baseline + (mayhem_cum[entry_i:] - offset)
    retrievable[entry_i:] = np.maximum(retrievable[entry_i:], 0.0)
    return retrievable


# Sanity check: /mnt/data/out_sample.csv mint 14m3biDuXQULX9aG5SDdRoTRxFj7ZJMsxKTxo3dNZkmX
# retrievable after the MAYHEM BUY == entry_pool + mayhem pool_sol_change, excluding OTHER pool_sol_change.


# ----------------------------
# Prefix arrays / first-passage utilities
# ----------------------------

def compute_prefix_arrays(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    values: 1D float array, from entry onward.
    Returns:
      prefix_max: max-so-far (monotone non-decreasing)
      prefix_min: min-so-far (monotone non-increasing)
      prefix_max_neg: max-so-far of (-values) (monotone non-decreasing)
    """
    prefix_max = np.maximum.accumulate(values)
    prefix_min = np.minimum.accumulate(values)
    prefix_max_neg = np.maximum.accumulate(-values)
    return prefix_max, prefix_min, prefix_max_neg


def first_idx_ge_prefix(prefix_max: np.ndarray, target: float) -> Optional[int]:
    j = int(np.searchsorted(prefix_max, target, side="left"))
    return None if j >= len(prefix_max) else j


def first_idx_le_via_prefix(prefix_max_neg: np.ndarray, target: float) -> Optional[int]:
    # first t where values[t] <= target  <=>  (-values[t]) >= (-target)
    j = int(np.searchsorted(prefix_max_neg, -target, side="left"))
    return None if j >= len(prefix_max_neg) else j


def timeout_index(offsets: np.ndarray, timeout_slots: int) -> Optional[int]:
    j = int(np.searchsorted(offsets, timeout_slots, side="left"))
    return None if j >= len(offsets) else j


# ----------------------------
# Core simulation primitives
# ----------------------------

def simulate_series_first_hit(
    slots_rel: np.ndarray,
    values_rel: np.ndarray,
    entry_value: float,
    tp_pct: float,
    sl_pct: float,
    timeout_slots: int,
    timeout_mode: str = "next_tick",
    soft_stops: Optional[Set[str]] = None,
    soft_a_slots: int = 0,
    soft_b_window_slots: int = 0,
    soft_b_ratio: float = 0.0,
) -> Tuple[int, int, str]:
    """
    Given series relative to entry, return (exit_value_j, hold_slots, reason).

    Trigger selection is based on the *time reached* (slot offset), not the event index:
      - TP/SL time is slots_rel[j]
      - TIMEOUT time is timeout_slots (exact) or slots_rel[j] (next_tick)

    Tie-breaks are deterministic; optional soft stops can be included with their own priorities.
    """
    tp_value = entry_value * (1.0 + float(tp_pct))
    sl_value = entry_value * (1.0 - float(sl_pct))

    prefix_max, prefix_min, prefix_max_neg = compute_prefix_arrays(values_rel)

    tp_j = first_idx_ge_prefix(prefix_max, tp_value)
    sl_j = first_idx_le_via_prefix(prefix_max_neg, sl_value)
    to_j = timeout_index(slots_rel, int(timeout_slots))

    active_soft_stops = set(soft_stops or ())
    if not active_soft_stops:
        cand: List[Tuple[int, int, int, str]] = []  # (time_reached, priority, exit_value_j, reason)
        if tp_j is not None:
            cand.append((int(slots_rel[tp_j]), 0, int(tp_j), "TP"))
        if sl_j is not None:
            cand.append((int(slots_rel[sl_j]), 1, int(sl_j), "SL"))
        if to_j is not None:
            if timeout_mode == "next_tick":
                cand.append((int(slots_rel[to_j]), 2, int(to_j), "TIMEOUT"))
            elif timeout_mode == "exact":
                # Exact mode exits at entry_slot + timeout_slots, using the last observed value at or before that slot.
                k = int(np.searchsorted(slots_rel, int(timeout_slots), side="right")) - 1
                k = max(k, 0)
                cand.append((int(timeout_slots), 2, k, "TIMEOUT"))
            else:
                return len(values_rel) - 1, int(slots_rel[-1]), f"unknown_timeout_mode:{timeout_mode}"

        if cand:
            hold_slots, _prio, exit_j, reason = min(cand)
            return int(exit_j), int(hold_slots), str(reason)
        exit_j = len(values_rel) - 1
        return int(exit_j), int(slots_rel[exit_j]), "END"

    cand: List[Tuple[int, int, int, str]] = []  # (time_reached, priority, exit_value_j, reason)
    if tp_j is not None:
        cand.append((int(slots_rel[tp_j]), 0, int(tp_j), "TP"))
    if "b" in active_soft_stops:
        end = int(np.searchsorted(slots_rel, int(soft_b_window_slots), side="right"))
        if end > 0:
            hits = np.flatnonzero(values_rel[:end] <= entry_value * float(soft_b_ratio))
            if len(hits):
                b_j = int(hits[0])
                cand.append((int(slots_rel[b_j]), 1, b_j, "SOFT_B"))
    if sl_j is not None:
        cand.append((int(slots_rel[sl_j]), 2, int(sl_j), "SL"))
    if "a" in active_soft_stops:
        a_j = timeout_index(slots_rel, int(soft_a_slots))
        if a_j is not None:
            if timeout_mode == "next_tick":
                cand.append((int(slots_rel[a_j]), 3, int(a_j), "SOFT_A"))
            elif timeout_mode == "exact":
                k = int(np.searchsorted(slots_rel, int(soft_a_slots), side="right")) - 1
                k = max(k, 0)
                cand.append((int(soft_a_slots), 3, k, "SOFT_A"))
            else:
                return len(values_rel) - 1, int(slots_rel[-1]), f"unknown_timeout_mode:{timeout_mode}"
    if to_j is not None:
        if timeout_mode == "next_tick":
            cand.append((int(slots_rel[to_j]), 4, int(to_j), "TIMEOUT"))
        elif timeout_mode == "exact":
            # Exact mode exits at entry_slot + timeout_slots, using the last observed value at or before that slot.
            k = int(np.searchsorted(slots_rel, int(timeout_slots), side="right")) - 1
            k = max(k, 0)
            cand.append((int(timeout_slots), 4, k, "TIMEOUT"))
        else:
            return len(values_rel) - 1, int(slots_rel[-1]), f"unknown_timeout_mode:{timeout_mode}"

    if cand:
        hold_slots, _prio, exit_j, reason = min(cand)
        return int(exit_j), int(hold_slots), str(reason)
    exit_j = len(values_rel) - 1
    return int(exit_j), int(slots_rel[exit_j]), "END"


def simulate_one_mint(
    slots: np.ndarray,
    pool_full: np.ndarray,
    pool: np.ndarray,
    entry_i: int,
    params: StrategyParams,
    creation_fee_sol: float = DEFAULT_CREATION_FEE_SOL,
    scale_mode: str = "linear",
    timeout_mode: str = "next_tick",
    soft_stops: Optional[Set[str]] = None,
    soft_a_slots: int = 0,
    soft_b_window_slots: int = 0,
    soft_b_ratio: float = 0.0,
) -> Dict[str, object]:
    """
    One-mint simulation:
    - entry at entry_i
    - exit on first TP/SL/timeout trigger (plus optional soft stops)

    Returns a dict with per-mint metrics (including MAE).
    """
    entry_slot = int(slots[entry_i])
    pool_entry = float(pool[entry_i])
    pool_entry_full = float(pool_full[entry_i])

    if not (pool_entry > 0 and math.isfinite(pool_entry)):
        return {"ok": False, "reason": "bad_entry_pool"}

    if params.buy_sol <= 0 or not math.isfinite(params.buy_sol):
        return {"ok": False, "reason": "bad_buy_sol"}

    if scale_mode == "linear":
        scale = float(params.buy_sol) / pool_entry
    elif scale_mode == "none":
        scale = 1.0
    else:
        return {"ok": False, "reason": f"unknown_scale_mode:{scale_mode}"}

    values_rel = pool[entry_i:] * scale
    slots_rel = slots[entry_i:] - entry_slot  # >=0

    entry_value = float(values_rel[0])

    exit_value_j, hold_slots, reason = simulate_series_first_hit(
        slots_rel=slots_rel,
        values_rel=values_rel,
        entry_value=entry_value,
        tp_pct=params.tp_pct,
        sl_pct=params.sl_pct,
        timeout_slots=params.timeout_slots,
        timeout_mode=str(timeout_mode),
        soft_stops=soft_stops,
        soft_a_slots=soft_a_slots,
        soft_b_window_slots=soft_b_window_slots,
        soft_b_ratio=soft_b_ratio,
    )

    exit_value_j = int(exit_value_j)
    hold_slots = int(hold_slots)
    exit_value = float(values_rel[exit_value_j])
    gross_pnl = exit_value - entry_value
    net_pnl = gross_pnl - float(creation_fee_sol)

    # Risk metric: max adverse excursion up to exit
    prefix_max, prefix_min, _prefix_max_neg = compute_prefix_arrays(values_rel)
    worst_value_to_exit = float(prefix_min[exit_value_j])
    mae_pct = (worst_value_to_exit / entry_value) - 1.0

    return {
        "ok": True,
        "pool_entry_full": pool_entry_full,
        "pool_entry_retrievable": pool_entry,
        # In --timeout-mode exact, exit_slot can fall between events; pool_exit_* is the last observed pool value
        # at or before exit_slot (piecewise-constant between events).
        "pool_exit_full": float(pool_full[entry_i + exit_value_j]),
        "pool_exit_retrievable": float(pool[entry_i + exit_value_j]),
        "scale": scale,
        "exit_reason": reason,
        "entry_slot": entry_slot,
        "exit_slot": int(entry_slot + hold_slots),
        "hold_slots": hold_slots,
        "entry_value": entry_value,
        "exit_value": exit_value,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "mae_pct": mae_pct,
        "max_value_to_exit": float(prefix_max[exit_value_j]),
        "max_return_pct_to_exit": (float(prefix_max[exit_value_j]) / entry_value) - 1.0,
    }


# ----------------------------
# Mint summarization (features)
# ----------------------------

def summarize_one_mint(
    mint: str,
    mdf: pd.DataFrame,
    pool_mode: str = "mayhem_only",
    bot_window_slots: int = 200,
) -> Dict[str, object]:
    entry_i = find_launch_index(mdf)
    slots = mdf["slot"].to_numpy(np.int64)
    pool_full = mdf["pool_sol_after"].to_numpy(np.float64)
    pool = compute_retrievable_pool_mayhem_only(mdf, entry_i) if pool_mode == "mayhem_only" else pool_full

    entry_slot = int(slots[entry_i]) if len(slots) else None
    entry_pool = float(pool[entry_i]) if len(pool) else float("nan")
    entry_block_time = ""
    if "block_time" in mdf.columns and len(mdf):
        bt_val = mdf.iloc[entry_i]["block_time"]
        if isinstance(bt_val, str):
            entry_block_time = bt_val
        elif pd.notna(bt_val):
            entry_block_time = str(bt_val)

    pnl_col = "tx_pnl_incl_network_fee" if "tx_pnl_incl_network_fee" in mdf.columns else "tx_pnl_excl_network_fee"
    dev_net_pnl = float(mdf.loc[mdf["actor"] == DEV_ACTOR, pnl_col].sum())
    mayhem_net_pnl = float(mdf.loc[mdf["actor"] == MAYHEM_ACTOR, pnl_col].sum())
    dev_profitable = dev_net_pnl > 0
    mayhem_profitable = mayhem_net_pnl > 0
    if dev_profitable and mayhem_profitable:
        net_winner = "BOTH"
    elif dev_profitable:
        net_winner = "DEV"
    elif mayhem_profitable:
        net_winner = "MAYHEM"
    else:
        net_winner = "NONE"

    out: Dict[str, object] = {
        "mint": mint,
        "n_events": int(len(mdf)),
        "entry_i": int(entry_i),
        "entry_slot": entry_slot,
        "entry_block_time": entry_block_time,
        "entry_pool": entry_pool,
        "dev_initial_buy_sol": find_first_dev_buy_sol(mdf),
        "dev_net_pnl_incl_network_fee": dev_net_pnl,
        "mayhem_net_pnl_incl_network_fee": mayhem_net_pnl,
        "net_winner": net_winner,
    }

    # Bot response within window after entry
    if entry_slot is not None:
        window_end = entry_slot + int(bot_window_slots)
        in_window = (mdf["slot"] >= entry_slot) & (mdf["slot"] <= window_end) & (mdf["actor"] == MAYHEM_ACTOR)
        bot_df = mdf.loc[in_window, ["action_type", "pool_sol_change"]]
        bot_buy = bot_df.loc[bot_df["action_type"] == "BUY", "pool_sol_change"].sum()
        bot_sell = -bot_df.loc[bot_df["action_type"] == "SELL", "pool_sol_change"].sum()  # make positive
        out["bot_window_buy_sol"] = float(bot_buy)
        out["bot_window_sell_sol"] = float(bot_sell)
        out["bot_window_net_sol"] = float(bot_buy - bot_sell)
        out["bot_window_trades"] = int(len(bot_df))
    else:
        out["bot_window_buy_sol"] = float("nan")
        out["bot_window_sell_sol"] = float("nan")
        out["bot_window_net_sol"] = float("nan")
        out["bot_window_trades"] = 0

    # Peak/trough stats relative to entry
    if len(pool) and math.isfinite(entry_pool) and entry_pool > 0 and entry_i < len(pool):
        rel_pool = pool[entry_i:]
        peak_j = int(np.argmax(rel_pool))
        trough_j = int(np.argmin(rel_pool))

        out["peak_pool"] = float(rel_pool[peak_j])
        out["peak_slot"] = int(slots[entry_i + peak_j])
        out["time_to_peak_slots"] = int(slots[entry_i + peak_j] - slots[entry_i])
        out["max_return_pct"] = float(rel_pool[peak_j] / entry_pool - 1.0)

        out["trough_pool"] = float(rel_pool[trough_j])
        out["trough_slot"] = int(slots[entry_i + trough_j])
        out["min_return_pct"] = float(rel_pool[trough_j] / entry_pool - 1.0)
    else:
        out["peak_pool"] = float("nan")
        out["peak_slot"] = None
        out["time_to_peak_slots"] = None
        out["max_return_pct"] = float("nan")
        out["trough_pool"] = float("nan")
        out["trough_slot"] = None
        out["min_return_pct"] = float("nan")

    return out


# ----------------------------
# CLI commands
# ----------------------------

def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def parse_int_list(s: str) -> List[int]:
    return _parse_int_list(s)


def parse_soft_stops(spec: str) -> Set[str]:
    out: Set[str] = set()
    for part in str(spec).replace(",", " ").split():
        key = part.strip().lower()
        if key in {"a", "b"}:
            out.add(key)
    return out


def run_summarize(args: argparse.Namespace) -> int:
    df = sort_events(load_events_csv(args.csv, extra_cols=("block_time", "tx_pnl_incl_network_fee")))
    rows: List[Dict[str, object]] = []
    for mint, slc in iter_mint_slices(df):
        mdf = df.iloc[slc]
        rows.append(summarize_one_mint(mint, mdf, pool_mode=str(args.pool_mode), bot_window_slots=int(args.bot_window_slots)))
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    return 0


def run_grid(args: argparse.Namespace) -> int:
    df = sort_events(load_events_csv(args.csv))

    buy_list = parse_float_list(args.buys)
    tp_list = parse_float_list(args.tps)
    sl_list = parse_float_list(args.sls)
    timeout_list = parse_int_list(args.timeouts)
    timeout_mode = str(getattr(args, "timeout_mode", "next_tick"))
    order_by = str(getattr(args, "order_by", "entry_slot"))
    soft_stops = parse_soft_stops(str(getattr(args, "soft_stops", "")))
    soft_stops_key = ",".join(sorted(soft_stops))

    soft_a_list = parse_int_list(args.soft_a_slots)
    soft_b_window_list = parse_int_list(args.soft_b_window_slots)
    soft_b_ratio_list = parse_float_list(args.soft_b_ratio)

    if not soft_a_list or not soft_b_window_list or not soft_b_ratio_list:
        print("Error: soft stop lists must not be empty.", file=sys.stderr)
        return 2

    if "a" in soft_stops:
        for a in soft_a_list:
            if a <= 0:
                print(f"Error: --soft-a-slots must be > 0 (got {a})", file=sys.stderr)
                return 2
        soft_a_list = sorted(set(soft_a_list))
    else:
        soft_a_list = [soft_a_list[0]]

    if "b" in soft_stops:
        for w in soft_b_window_list:
            if w <= 0:
                print(f"Error: --soft-b-window-slots must be > 0 (got {w})", file=sys.stderr)
                return 2
        for r in soft_b_ratio_list:
            if not (0.0 < float(r) < 1.0):
                print(f"Error: --soft-b-ratio must be in (0, 1) (got {r})", file=sys.stderr)
                return 2
        soft_b_window_list = sorted(set(soft_b_window_list))
        soft_b_ratio_list = sorted(set(float(r) for r in soft_b_ratio_list))
    else:
        soft_b_window_list = [soft_b_window_list[0]]
        soft_b_ratio_list = [float(soft_b_ratio_list[0])]

    soft_b_pairs = list(itertools.product(soft_b_window_list, soft_b_ratio_list))

    slippage_bps = int(args.slippage_bps)
    if slippage_bps < 0:
        print(f"Error: --slippage-bps must be >= 0 (got {slippage_bps})", file=sys.stderr)
        return 2
    slip_pct = float(slippage_bps) / 10000.0

    slot_ms = float(args.slot_ms)
    if not (math.isfinite(slot_ms) and slot_ms > 0):
        print(f"Error: --slot-ms must be > 0 (got {args.slot_ms})", file=sys.stderr)
        return 2

    start_sol_raw = str(args.start_sol).strip().lower()
    if start_sol_raw == "auto":
        start_sol = None
    else:
        try:
            start_sol = float(start_sol_raw)
        except ValueError:
            print(f"Error: invalid --start-sol value {args.start_sol!r}", file=sys.stderr)
            return 2
        if not (math.isfinite(start_sol) and start_sol > 0):
            print(f"Error: --start-sol must be > 0 (got {args.start_sol})", file=sys.stderr)
            return 2

    launches_per_day = _parse_int_list(args.launches_per_day)
    worst_windows_raw = _parse_int_list(args.worst_rolling_windows)
    worst_windows: List[int] = []
    seen_windows = set()
    for w in worst_windows_raw:
        if w <= 0 or w in seen_windows:
            continue
        seen_windows.add(w)
        worst_windows.append(w)

    # Prebuild param list
    params_list = [
        StrategyParams(
            buy_sol=b,
            tp_pct=tp,
            sl_pct=sl,
            timeout_slots=to,
            soft_stops=soft_stops_key,
            soft_a_slots=soft_a_slots,
            soft_b_window_slots=soft_b_window,
            soft_b_ratio=soft_b_ratio,
        )
        for b, tp, sl, to, soft_a_slots, (soft_b_window, soft_b_ratio) in itertools.product(
            buy_list,
            tp_list,
            sl_list,
            timeout_list,
            soft_a_list,
            soft_b_pairs,
        )
    ]

    # Aggregation buffers
    agg: Dict[StrategyParams, ExtendedAgg] = {}
    for p in params_list:
        start_val = float(p.buy_sol) if start_sol is None else float(start_sol)
        agg[p] = ExtendedAgg(
            start_sol=start_val,
            order_by=order_by,
            worst_windows=worst_windows,
            scale_mode=str(args.scale_mode),
            slot_ms=slot_ms,
        )

    mint_entries: List[Tuple[int, str, slice, int]] = []
    for mint, slc in iter_mint_slices(df):
        mdf = df.iloc[slc]
        entry_i = find_launch_index(mdf)
        slots = mdf["slot"].to_numpy(np.int64)
        if len(slots) == 0:
            continue
        entry_slot = int(slots[entry_i])
        mint_entries.append((entry_slot, str(mint), slc, entry_i))

    mint_entries.sort(key=lambda x: (x[0], x[1]))

    for entry_slot, mint, slc, entry_i in mint_entries:
        mdf = df.iloc[slc]

        slots = mdf["slot"].to_numpy(np.int64)
        pool_full = mdf["pool_sol_after"].to_numpy(np.float64)
        if len(pool_full) == 0:
            continue
        pool = compute_retrievable_pool_mayhem_only(mdf, entry_i) if str(args.pool_mode) == "mayhem_only" else pool_full

        pool_entry = float(pool[entry_i])
        if not (pool_entry > 0 and math.isfinite(pool_entry)):
            continue

        offsets_rel = slots[entry_i:] - entry_slot  # monotone

        # If requested, only evaluate the buy bucket that matches this mint's observed entry pool.
        if str(args.buy_match) == "entry_pool":
            assigned_buy = _assign_buy_bucket(pool_entry, buy_list, float(args.buy_tol))
            if assigned_buy is None:
                continue
            buys_iter = [assigned_buy]
        else:
            buys_iter = buy_list

        for buy in buys_iter:
            if buy <= 0:
                continue
            scale = float(buy) / pool_entry if args.scale_mode == "linear" else 1.0
            values_rel = pool[entry_i:] * scale
            entry_value = float(values_rel[0])

            prefix_max, prefix_min, prefix_max_neg = compute_prefix_arrays(values_rel)

            # Precompute trigger indices for every tp/sl/timeout for this mint+buy
            tp_j = {tp: first_idx_ge_prefix(prefix_max, entry_value * (1.0 + tp)) for tp in tp_list}
            sl_j = {sl: first_idx_le_via_prefix(prefix_max_neg, entry_value * (1.0 - sl)) for sl in sl_list}
            to_j = {to: timeout_index(offsets_rel, to) for to in timeout_list}
            # In --timeout-mode exact, TIMEOUT uses the last observed value at or before timeout_slots.
            to_k_exact = (
                {
                    to: (max(int(np.searchsorted(offsets_rel, int(to), side="right")) - 1, 0) if to_j[to] is not None else None)
                    for to in timeout_list
                }
                if timeout_mode == "exact"
                else {}
            )

            soft_a_j = {a: timeout_index(offsets_rel, a) for a in soft_a_list} if "a" in soft_stops else {}
            soft_a_k_exact = (
                {
                    a: (max(int(np.searchsorted(offsets_rel, int(a), side="right")) - 1, 0) if soft_a_j[a] is not None else None)
                    for a in soft_a_list
                }
                if timeout_mode == "exact" and "a" in soft_stops
                else {}
            )
            soft_b_j_by_ratio = (
                {r: first_idx_le_via_prefix(prefix_max_neg, entry_value * float(r)) for r in soft_b_ratio_list}
                if "b" in soft_stops
                else {}
            )

            for tp, sl, to, soft_a_slots, (soft_b_window, soft_b_ratio) in itertools.product(
                tp_list,
                sl_list,
                timeout_list,
                soft_a_list,
                soft_b_pairs,
            ):
                p = StrategyParams(
                    buy_sol=buy,
                    tp_pct=tp,
                    sl_pct=sl,
                    timeout_slots=to,
                    soft_stops=soft_stops_key,
                    soft_a_slots=soft_a_slots,
                    soft_b_window_slots=soft_b_window,
                    soft_b_ratio=soft_b_ratio,
                )

                if not soft_stops:
                    # Compare TP/SL/TIMEOUT by time reached (slot offset), not index; TP/SL win ties vs TIMEOUT.
                    cand: List[Tuple[int, int, int, str]] = []  # (time_reached, priority, exit_value_j, reason)
                    if tp_j[tp] is not None:
                        j = int(tp_j[tp])
                        cand.append((int(offsets_rel[j]), 0, j, "TP"))
                    if sl_j[sl] is not None:
                        j = int(sl_j[sl])
                        cand.append((int(offsets_rel[j]), 1, j, "SL"))
                    if to_j[to] is not None:
                        if timeout_mode == "next_tick":
                            j = int(to_j[to])
                            cand.append((int(offsets_rel[j]), 2, j, "TIMEOUT"))
                        elif timeout_mode == "exact":
                            k = int(to_k_exact[to])  # present because to_j[to] is not None
                            cand.append((int(to), 2, k, "TIMEOUT"))
                        else:
                            # Unknown timeout mode: fall back to current behavior.
                            j = int(to_j[to])
                            cand.append((int(offsets_rel[j]), 2, j, "TIMEOUT"))
                else:
                    cand = []
                    if tp_j[tp] is not None:
                        j = int(tp_j[tp])
                        cand.append((int(offsets_rel[j]), 0, j, "TP"))
                    if "b" in soft_stops:
                        b_j = soft_b_j_by_ratio.get(soft_b_ratio)
                        if b_j is not None and int(offsets_rel[int(b_j)]) <= int(soft_b_window):
                            cand.append((int(offsets_rel[int(b_j)]), 1, int(b_j), "SOFT_B"))
                    if sl_j[sl] is not None:
                        j = int(sl_j[sl])
                        cand.append((int(offsets_rel[j]), 2, j, "SL"))
                    if "a" in soft_stops:
                        a_j = soft_a_j.get(soft_a_slots)
                        if a_j is not None:
                            if timeout_mode == "next_tick":
                                cand.append((int(offsets_rel[int(a_j)]), 3, int(a_j), "SOFT_A"))
                            elif timeout_mode == "exact":
                                k = soft_a_k_exact.get(soft_a_slots)
                                if k is not None:
                                    cand.append((int(soft_a_slots), 3, int(k), "SOFT_A"))
                            else:
                                j = int(a_j)
                                cand.append((int(offsets_rel[j]), 3, j, "SOFT_A"))
                    if to_j[to] is not None:
                        if timeout_mode == "next_tick":
                            j = int(to_j[to])
                            cand.append((int(offsets_rel[j]), 4, j, "TIMEOUT"))
                        elif timeout_mode == "exact":
                            k = int(to_k_exact[to])  # present because to_j[to] is not None
                            cand.append((int(to), 4, k, "TIMEOUT"))
                        else:
                            j = int(to_j[to])
                            cand.append((int(offsets_rel[j]), 4, j, "TIMEOUT"))

                if cand:
                    hold_slots, _prio, exit_j, reason = min(cand)
                    exit_j = int(exit_j)
                    hold_slots = int(hold_slots)
                else:
                    exit_j, reason = len(values_rel) - 1, "END"
                    hold_slots = int(offsets_rel[int(exit_j)])

                exit_value = float(values_rel[exit_j])
                gross = exit_value - entry_value
                net_raw = gross - float(args.creation_fee_sol)
                slippage_cost = slip_pct * (abs(entry_value) + abs(exit_value)) if slip_pct > 0 else 0.0
                net = net_raw - slippage_cost

                worst = float(prefix_min[exit_j])
                mae_pct = (worst / entry_value) - 1.0
                best = float(prefix_max[exit_j])
                mfe_pct = (best / entry_value) - 1.0

                exit_slot = int(entry_slot + hold_slots)

                a = agg[p]
                a.update_trade(
                    net_pnl=net,
                    net_pnl_raw=net_raw,
                    slippage_cost=slippage_cost,
                    gross_pnl=gross,
                    hold_slots=hold_slots,
                    mae_pct=mae_pct,
                    mfe_pct=mfe_pct,
                    exit_reason=reason,
                    buy_sol=float(buy),
                    entry_slot=int(entry_slot),
                    exit_slot=exit_slot,
                    mint=mint,
                )

    # Emit aggregated results
    rows = []
    for p, a in agg.items():
        metrics = a.finalize(launches_per_day)
        if not metrics:
            continue
        row: Dict[str, object] = {
            "buy_sol": p.buy_sol,
            "tp_pct": p.tp_pct,
            "sl_pct": p.sl_pct,
            "timeout_slots": p.timeout_slots,
            "slippage_bps": slippage_bps,
            "soft_stops": p.soft_stops,
            "soft_a_slots": p.soft_a_slots,
            "soft_b_window_slots": p.soft_b_window_slots,
            "soft_b_ratio": p.soft_b_ratio,
        }

        nested_keys = {"sharpe_annualized", "sortino_annualized", "calmar_by_lpd", "recovery_time_by_lpd", "worst_rolling_pnl", "throughput", "projections"}
        for k, v in metrics.items():
            if k in nested_keys:
                continue
            row[k] = v

        for w in worst_windows:
            row[f"worst_rolling_pnl_{w}"] = (metrics.get("worst_rolling_pnl") or {}).get(int(w))

        for lpd in launches_per_day:
            lpd_key = int(lpd)
            row[f"sharpe_ratio_annualized@{lpd_key}_launches"] = (metrics.get("sharpe_annualized") or {}).get(lpd_key)
            row[f"sortino_ratio_annualized@{lpd_key}_launches"] = (metrics.get("sortino_annualized") or {}).get(lpd_key)
            row[f"calmar_ratio@{lpd_key}_launches"] = (metrics.get("calmar_by_lpd") or {}).get(lpd_key)
            rt = (metrics.get("recovery_time_by_lpd") or {}).get(lpd_key)
            row[f"time_to_recovery_days@{lpd_key}_launches"] = rt.get("days") if rt else None
            row[f"time_to_recovery_hours@{lpd_key}_launches"] = rt.get("hours") if rt else None
            proj = (metrics.get("projections") or {}).get(lpd_key)
            row[f"avg_net_pnl_per_day@{lpd_key}_launches"] = proj.get("per_day") if proj else None
            row[f"avg_net_pnl_per_hour@{lpd_key}_launches"] = proj.get("per_hour") if proj else None

        throughput = metrics.get("throughput") or {}
        row["pnl_per_sol_slot_total"] = throughput.get("pnl_per_sol_slot_total")
        row["pnl_per_sol_minute_total"] = throughput.get("pnl_per_sol_minute_total")
        row["denom_sol_slots"] = throughput.get("denom_sol_slots")

        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(["total_net_pnl", "win_rate"], ascending=[False, False])
    out_df.to_csv(args.out, index=False)
    return 0


def run_simulate(args: argparse.Namespace) -> int:
    """
    Output one row per mint for a single concrete strategy.
    Useful for:
      - PnL distribution plots
      - MAE quantiles for stop-loss selection
      - "frequency vs magnitude" comparisons
    """
    df = sort_events(load_events_csv(args.csv))
    params = StrategyParams(
        buy_sol=float(args.buy),
        tp_pct=float(args.tp),
        sl_pct=float(args.sl),
        timeout_slots=int(args.timeout),
    )
    soft_stops = parse_soft_stops(str(args.soft_stops))
    soft_a_slots = int(args.soft_a_slots)
    soft_b_window_slots = int(args.soft_b_window_slots)
    soft_b_ratio = float(args.soft_b_ratio)

    if "a" in soft_stops and soft_a_slots <= 0:
        print(f"Error: --soft-a-slots must be > 0 (got {args.soft_a_slots})", file=sys.stderr)
        return 2
    if "b" in soft_stops:
        if soft_b_window_slots <= 0:
            print(f"Error: --soft-b-window-slots must be > 0 (got {args.soft_b_window_slots})", file=sys.stderr)
            return 2
        if not (0.0 < soft_b_ratio < 1.0):
            print(f"Error: --soft-b-ratio must be in (0, 1) (got {args.soft_b_ratio})", file=sys.stderr)
            return 2

    if bool(getattr(args, "strict", False)) and str(args.buy_match) != "entry_pool":
        print("Error: --strict requires --buy-match entry_pool", file=sys.stderr)
        return 2

    buy_tol = float(args.buy_tol)
    if bool(getattr(args, "strict", False)):
        buy_tol = min(buy_tol, 0.02)

    rows: List[Dict[str, object]] = []
    for mint, slc in iter_mint_slices(df):
        mdf = df.iloc[slc]
        entry_i = find_launch_index(mdf)
        slots = mdf["slot"].to_numpy(np.int64)
        pool_full = mdf["pool_sol_after"].to_numpy(np.float64)
        if len(pool_full) == 0:
            continue
        pool = compute_retrievable_pool_mayhem_only(mdf, entry_i) if str(args.pool_mode) == "mayhem_only" else pool_full
        if str(args.buy_match) == "entry_pool":
            pool_entry = float(pool[entry_i])
            if not (pool_entry > 0 and math.isfinite(pool_entry)):
                continue
            if abs(pool_entry - float(params.buy_sol)) > buy_tol:
                continue

        res = simulate_one_mint(
            slots=slots,
            pool_full=pool_full,
            pool=pool,
            entry_i=entry_i,
            params=params,
            creation_fee_sol=float(args.creation_fee_sol),
            scale_mode=str(args.scale_mode),
            timeout_mode=str(args.timeout_mode),
            soft_stops=soft_stops,
            soft_a_slots=soft_a_slots,
            soft_b_window_slots=soft_b_window_slots,
            soft_b_ratio=soft_b_ratio,
        )
        if not res.get("ok", False):
            continue
        res_out = {
            "mint": mint,
            "entry_i": entry_i,
            "buy_sol": params.buy_sol,
            "tp_pct": params.tp_pct,
            "sl_pct": params.sl_pct,
            "timeout_slots": params.timeout_slots,
            **{k: v for k, v in res.items() if k != "ok"},
        }
        rows.append(res_out)

    out_df = pd.DataFrame(rows)
    # Ensure chronological order for downstream equity-curve stats, even if the consumer doesn't sort.
    if not out_df.empty and "entry_slot" in out_df.columns:
        out_df = out_df.sort_values(["entry_slot", "mint"], kind="mergesort").reset_index(drop=True)
    out_df.to_csv(args.out, index=False)
    return 0


def _assign_buy_bucket(value: float, buckets: Sequence[float], tol: float) -> Optional[float]:
    """
    Map a dev_initial_buy_sol to the nearest configured bucket if within tolerance.
    Returns the bucket value or None.
    """
    if not (math.isfinite(value) and value > 0) or not buckets:
        return None
    arr = np.asarray(list(buckets), dtype=float)
    j = int(np.argmin(np.abs(arr - value)))
    if abs(float(arr[j]) - float(value)) <= float(tol):
        return float(arr[j])
    return None


def run_reach(args: argparse.Namespace) -> int:
    """
    For each mint, determine whether the pool ever reaches each TP threshold (percent gain from entry),
    within an optional max-hold window, and aggregate by observed dev initial buy bucket.

    This is intended to help build the "Dev Buy Amounts vs Potential Take Profit targets" table
    described in the guide.
    """
    df = sort_events(load_events_csv(args.csv))

    tp_list = parse_float_list(args.tps)
    buy_buckets = parse_float_list(args.buy_buckets) if args.buy_buckets else []
    buy_tol = float(args.buy_tol)

    max_slots = int(args.max_slots) if args.max_slots is not None else None

    # stats[(bucket, tp)] = dict
    stats: Dict[Tuple[Optional[float], float], Dict[str, float]] = {}

    for mint, slc in iter_mint_slices(df):
        mdf = df.iloc[slc]
        entry_i = find_launch_index(mdf)

        slots = mdf["slot"].to_numpy(np.int64)
        pool_full = mdf["pool_sol_after"].to_numpy(np.float64)
        if len(pool_full) == 0:
            continue
        pool = compute_retrievable_pool_mayhem_only(mdf, entry_i) if str(args.pool_mode) == "mayhem_only" else pool_full

        entry_slot = int(slots[entry_i])
        entry_pool = float(pool[entry_i])
        if not (entry_pool > 0 and math.isfinite(entry_pool)):
            continue

        # optional time window truncation (to avoid right-censoring or to focus on early opportunity window)
        if max_slots is not None and max_slots >= 0:
            offsets = slots[entry_i:] - entry_slot
            end = int(np.searchsorted(offsets, max_slots, side="right"))
            if end <= 0:
                continue
            values = pool[entry_i : entry_i + end]
        else:
            values = pool[entry_i:]

        prefix_max = np.maximum.accumulate(values)

        dev_buy = find_first_dev_buy_sol(mdf)
        bucket = _assign_buy_bucket(dev_buy, buy_buckets, buy_tol) if buy_buckets else None

        for tp in tp_list:
            key = (bucket, tp)
            st = stats.get(key)
            if st is None:
                st = {"n": 0, "hits": 0, "sum_time_to_hit_slots": 0.0}
                stats[key] = st

            st["n"] += 1
            target = entry_pool * (1.0 + float(tp))
            j = int(np.searchsorted(prefix_max, target, side="left"))
            if j < len(prefix_max):
                st["hits"] += 1
                # approximate time-to-hit: slot delta at index j
                st["sum_time_to_hit_slots"] += float(slots[entry_i + j] - entry_slot)

    rows = []
    for (bucket, tp), st in stats.items():
        n = int(st["n"])
        hits = int(st["hits"])
        avg_time = (st["sum_time_to_hit_slots"] / hits) if hits else float("nan")
        rows.append(
            {
                "buy_bucket": bucket if bucket is not None else "ALL/UNBUCKETED",
                "tp_pct": tp,
                "n_mints": n,
                "hit_rate": (hits / n) if n else float("nan"),
                "avg_time_to_hit_slots": avg_time,
                "avg_time_to_hit_seconds": avg_time * (float(args.slot_ms) / 1000.0) if math.isfinite(avg_time) else float("nan"),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["buy_bucket", "tp_pct"], ascending=[True, True])
    out_df.to_csv(args.out, index=False)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_sum = sub.add_parser("summarize", help="Build per-mint summary features (one row per mint).")
    ap_sum.add_argument("--csv", required=True, help="Path to the exported CSV.")
    ap_sum.add_argument("--out", required=True, help="Output CSV path for per-mint summary.")
    ap_sum.add_argument("--bot-window-slots", type=int, default=200, help="Window after launch to measure bot response.")
    ap_sum.add_argument(
        "--pool-mode",
        choices=["full", "mayhem_only"],
        default="mayhem_only",
        help="Pool series to use. full=raw pool_sol_after. mayhem_only=entry baseline + MAYHEM net only.",
    )
    ap_sum.set_defaults(func=run_summarize)

    ap_grid = sub.add_parser("grid", help="Run a grid search over TP/SL/timeout for observed price action.")
    ap_grid.add_argument("--csv", required=True, help="Path to the exported CSV.")
    ap_grid.add_argument("--out", required=True, help="Output CSV path for aggregated grid results.")
    ap_grid.add_argument("--buys", default="0.5,1.0,1.5,2.0", help="Comma list of initial buys (SOL).")
    ap_grid.add_argument("--buy-match", choices=["entry_pool", "none"], default="entry_pool",
                         help="How to apply --buys. entry_pool=assign each mint to nearest buy bucket by entry pool (recommended). none=apply every buy to every mint.")
    ap_grid.add_argument("--buy-tol", type=float, default=0.15,
                         help="Absolute tolerance for assigning a mint to a buy bucket when --buy-match entry_pool.")
    ap_grid.add_argument("--tps", default="0.05,0.10,0.20,0.50", help="Comma list of take-profit percentages.")
    ap_grid.add_argument("--sls", default="0.05,0.10,0.20", help="Comma list of stop-loss percentages.")
    ap_grid.add_argument("--timeouts", default="50,100,200,400", help="Comma list of deadline timeouts (slots).")
    ap_grid.add_argument(
        "--soft-stops",
        default="",
        help="Comma/space list of soft stops to enable: a,b (empty disables).",
    )
    ap_grid.add_argument("--soft-a-slots", default="180", help="Comma list of soft stop A timeouts (slots).")
    ap_grid.add_argument("--soft-b-window-slots", default="20", help="Comma list of soft stop B windows (slots).")
    ap_grid.add_argument("--soft-b-ratio", default="0.45", help="Comma list of soft stop B pool ratio thresholds (0-1).")
    ap_grid.add_argument(
        "--pool-mode",
        choices=["full", "mayhem_only"],
        default="mayhem_only",
        help="Pool series to use. full=raw pool_sol_after. mayhem_only=entry baseline + MAYHEM net only.",
    )
    ap_grid.add_argument(
        "--timeout-mode",
        choices=["next_tick", "exact"],
        default="next_tick",
        help=(
            "How TIMEOUT exits are computed. next_tick=exit at the first observed event with offset >= timeout "
            "(current behavior). exact=exit exactly at entry_slot+timeout using the last observed value at or before "
            "that slot (piecewise-constant between events)."
        ),
    )
    ap_grid.add_argument("--creation-fee-sol", type=float, default=DEFAULT_CREATION_FEE_SOL, help="Per-launch fee to subtract.")
    ap_grid.add_argument("--slippage-bps", type=int, default=0, help="Per-swap slippage in bps (applied on entry and exit).")
    ap_grid.add_argument("--start-sol", default="auto", help="Starting bankroll for risk metrics. auto=buy_sol.")
    ap_grid.add_argument("--slot-ms", type=float, default=DEFAULT_SLOT_MS, help="Slot duration in ms for time conversion.")
    ap_grid.add_argument("--launches-per-day", default="750,480,288", help="Comma list of launches per day for annualization.")
    ap_grid.add_argument(
        "--order-by",
        choices=["entry_slot", "exit_slot"],
        default="entry_slot",
        help="Trade ordering for sequence metrics.",
    )
    ap_grid.add_argument(
        "--worst-rolling-windows",
        default="20,50,100",
        help="Comma list of rolling window sizes for worst PnL.",
    )
    ap_grid.add_argument(
        "--scale-mode",
        choices=["none", "linear"],
        default="linear",
        help=(
            "Pool mapping / sizing. linear=scales pool series so entry_value == buy_sol, and PnL is proportional to "
            "pool return. none=use raw pool values; buy_sol does NOT size the trade and is only a label."
        ),
    )
    ap_grid.set_defaults(func=run_grid)

    ap_sim = sub.add_parser("simulate", help="Per-mint results for a single strategy (distributions).")
    ap_sim.add_argument("--csv", required=True, help="Path to the exported CSV.")
    ap_sim.add_argument("--out", required=True, help="Output CSV path for per-mint strategy results.")
    ap_sim.add_argument("--buy", type=float, required=True, help="Initial dev buy size (SOL).")
    ap_sim.add_argument("--buy-match", choices=["entry_pool", "none"], default="entry_pool",
                        help="Only simulate mints whose entry pool is within --buy-tol of --buy (recommended).")
    ap_sim.add_argument("--buy-tol", type=float, default=0.15,
                        help="Absolute tolerance for --buy-match entry_pool.")
    ap_sim.add_argument(
        "--strict",
        action="store_true",
        help="Use strict buy tolerance (±0.02 SOL) when --buy-match entry_pool.",
    )
    ap_sim.add_argument("--tp", type=float, required=True, help="Take profit percent (e.g. 0.10 for +10%%).")
    ap_sim.add_argument("--sl", type=float, required=True, help="Stop loss percent (e.g. 0.10 for -10%%).")
    ap_sim.add_argument("--timeout", type=int, required=True, help="Timeout in slots.")
    ap_sim.add_argument(
        "--timeout-mode",
        choices=["next_tick", "exact"],
        default="next_tick",
        help=(
            "How TIMEOUT exits are computed. next_tick=exit at the first observed event with offset >= timeout "
            "(current behavior). exact=exit exactly at entry_slot+timeout using the last observed value at or before "
            "that slot (piecewise-constant between events)."
        ),
    )
    ap_sim.add_argument(
        "--soft-stops",
        default="",
        help="Comma/space list of soft stops to enable: a,b (empty disables).",
    )
    ap_sim.add_argument("--soft-a-slots", type=int, default=180, help="Soft stop A timeout in slots.")
    ap_sim.add_argument("--soft-b-window-slots", type=int, default=20, help="Soft stop B lookback window in slots.")
    ap_sim.add_argument("--soft-b-ratio", type=float, default=0.45, help="Soft stop B pool ratio threshold (0-1).")
    ap_sim.add_argument("--creation-fee-sol", type=float, default=DEFAULT_CREATION_FEE_SOL, help="Per-launch fee to subtract.")
    ap_sim.add_argument(
        "--pool-mode",
        choices=["full", "mayhem_only"],
        default="mayhem_only",
        help="Pool series to use. full=raw pool_sol_after. mayhem_only=entry baseline + MAYHEM net only.",
    )
    ap_sim.add_argument(
        "--scale-mode",
        choices=["none", "linear"],
        default="linear",
        help=(
            "Pool mapping / sizing. linear=scales pool series so entry_value == buy_sol, and PnL is proportional to "
            "pool return. none=use raw pool values; buy_sol does NOT size the trade and is only a label."
        ),
    )
    ap_sim.set_defaults(func=run_simulate)

    ap_reach = sub.add_parser("reach", help="TP reachability table grouped by dev buy bucket.")
    ap_reach.add_argument("--csv", required=True, help="Path to the exported CSV.")
    ap_reach.add_argument("--out", required=True, help="Output CSV path for TP reachability.")
    ap_reach.add_argument("--tps", default="0.05,0.10,0.20,0.50", help="Comma list of TP percentages to test.")
    ap_reach.add_argument("--buy-buckets", default="0.5,1.0,1.5,2.0", help="Comma list of dev buy buckets (SOL). Empty => no bucketing.")
    ap_reach.add_argument("--buy-tol", type=float, default=0.15, help="Absolute tolerance for assigning to a buy bucket.")
    ap_reach.add_argument("--max-slots", type=int, default=None, help="Only consider first N slots after entry (optional).")
    ap_reach.add_argument(
        "--pool-mode",
        choices=["full", "mayhem_only"],
        default="mayhem_only",
        help="Pool series to use. full=raw pool_sol_after. mayhem_only=entry baseline + MAYHEM net only.",
    )
    ap_reach.add_argument("--slot-ms", type=float, default=DEFAULT_SLOT_MS, help="Slot duration in ms for time conversion.")
    ap_reach.set_defaults(func=run_reach)

    return ap


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
