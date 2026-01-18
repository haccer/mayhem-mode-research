#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any


@dataclass(frozen=True)
class Trade:
    line_no: int  # 1-based line in CSV (header is line 1)
    mint: str | None
    entry_slot: int | None
    exit_slot: int | None
    net_pnl_raw: Decimal
    slippage_cost: Decimal
    net_pnl: Decimal  # net after slippage
    buy_sol: Decimal | None
    exit_reason: str | None
    hold_slots: int | None
    mae_pct: float | None
    mfe_pct: float | None


def _fmt_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _fmt_percent(fraction: Decimal, *, places: int = 2) -> str:
    quant = Decimal("1").scaleb(-places)  # e.g. 1E-2
    pct = (fraction * Decimal("100")).quantize(quant, rounding=ROUND_HALF_UP)
    return format(pct, "f")


def _fmt_float(value: float) -> str:
    if not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.12g}"


def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if fraction <= 0:
        return float(sorted_values[0])
    if fraction >= 1:
        return float(sorted_values[-1])

    n = len(sorted_values)
    k = (n - 1) * float(fraction)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(sorted_values[f])
    d0 = float(sorted_values[f])
    d1 = float(sorted_values[c])
    return d0 + (d1 - d0) * (k - f)


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_strategy_cell(field: str, raw: str) -> str:
    raw = (raw or "").strip()
    if raw == "":
        return ""

    if field in {"timeout_slots", "entry_i"}:
        try:
            return str(int(Decimal(raw)))
        except (InvalidOperation, ValueError):
            return raw

    try:
        return _fmt_decimal(Decimal(raw))
    except InvalidOperation:
        return raw


def _format_strategy(fields: list[str], values: tuple[str, ...]) -> str:
    return " ".join(f"{k}={v}" for k, v in zip(fields, values, strict=False) if v != "")


def _classify_exit_reason(reason: str | None) -> str:
    r = (reason or "").strip().upper()
    if r == "TP":
        return "TP"
    if r == "SL":
        return "SL"
    if r in {"TIMEOUT", "TO"}:
        return "TIMEOUT"
    if r == "END":
        return "END"
    return r if r else "UNKNOWN"


def calculate(
    path: str,
    *,
    start_sol: Decimal | None,
    net_pnl_col: str,
    mint_col: str | None,
    launches_per_day: list[int],
    mc_runs: int,
    mc_seed: int | None,
    slippage_bps: int,
    slot_ms: float,
    order_by: str | None = None,
) -> dict[str, Any]:
    if int(slippage_bps) < 0:
        raise ValueError(f"Invalid slippage (bps must be >= 0): {slippage_bps}")
    if not (math.isfinite(float(slot_ms)) and float(slot_ms) > 0):
        raise ValueError(f"Invalid --slot-ms (must be > 0): {slot_ms}")

    slip_pct = Decimal(int(slippage_bps)) / Decimal("10000")

    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("CSV is empty")

        header = [h.strip() for h in header]
        col_index = {name: i for i, name in enumerate(header)}

        if net_pnl_col not in col_index:
            raise ValueError(f"Missing column {net_pnl_col!r}. Found: {header}")

        net_idx = col_index[net_pnl_col]
        mint_idx = col_index.get(mint_col) if mint_col else None

        buy_idx = col_index.get("buy_sol")
        entry_idx = col_index.get("entry_value")
        exit_idx = col_index.get("exit_value")
        entry_slot_idx = col_index.get("entry_slot")
        exit_slot_idx = col_index.get("exit_slot")
        exit_reason_idx = col_index.get("exit_reason")
        hold_slots_idx = col_index.get("hold_slots")
        mae_pct_idx = col_index.get("mae_pct")
        mfe_pct_idx = col_index.get("max_return_pct_to_exit")

        # Trade ordering for sequence-based metrics (equity curve, drawdowns, streaks, rolling windows).
        requested_order_by = (str(order_by).strip().lower() if order_by is not None else "")
        if requested_order_by == "":
            if entry_slot_idx is not None:
                effective_order_by = "entry_slot"
            elif exit_slot_idx is not None:
                effective_order_by = "exit_slot"
            else:
                effective_order_by = "file"
        else:
            if requested_order_by not in {"file", "entry_slot", "exit_slot"}:
                raise ValueError(f"Invalid --order-by value: {order_by!r}")
            if requested_order_by == "entry_slot" and entry_slot_idx is None:
                raise ValueError("Missing column 'entry_slot' required for --order-by entry_slot.")
            if requested_order_by == "exit_slot" and exit_slot_idx is None:
                raise ValueError("Missing column 'exit_slot' required for --order-by exit_slot.")
            effective_order_by = requested_order_by

        sizing_checked = 0
        sizing_mismatched = 0
        first_sizing_mismatch: bool | None = None

        if slip_pct == 0:
            slippage_method = "none"
        elif entry_idx is not None and exit_idx is not None:
            slippage_method = "entry_exit_values"
        elif buy_idx is not None:
            slippage_method = "buy_sol_notional"
        else:
            raise ValueError("To apply slippage, need columns entry_value & exit_value, or buy_sol.")

        strategy_fields = [c for c in ["buy_sol", "tp_pct", "sl_pct", "timeout_slots", "entry_i"] if c in col_index]
        strategy_idxs = [col_index[c] for c in strategy_fields]
        strategy_variants: set[tuple[str, ...]] = set()
        first_strategy: tuple[str, ...] | None = None

        trades: list[Trade] = []
        buy_sol_values: set[Decimal] = set()

        bad: list[tuple[int, str]] = []
        for line_no, row in enumerate(reader, start=2):
            if not row or all(not (cell or "").strip() for cell in row):
                continue

            mint = None
            if mint_idx is not None and mint_idx < len(row):
                mint = (row[mint_idx] or "").strip() or None

            try:
                raw_net = (row[net_idx] if net_idx < len(row) else "").strip()
                net_pnl_raw = Decimal(raw_net) if raw_net else Decimal("0")
            except (InvalidOperation, IndexError) as e:
                bad.append((line_no, f"net_pnl parse error: {e}"))
                continue

            buy_sol: Decimal | None = None
            if buy_idx is not None:
                try:
                    raw_buy = (row[buy_idx] if buy_idx < len(row) else "").strip()
                    if raw_buy:
                        buy_sol = Decimal(raw_buy)
                        buy_sol_values.add(buy_sol)
                except InvalidOperation:
                    bad.append((line_no, "invalid buy_sol"))
                    continue

            exit_reason = None
            if exit_reason_idx is not None and exit_reason_idx < len(row):
                exit_reason = (row[exit_reason_idx] or "").strip() or None

            entry_slot: int | None = None
            if entry_slot_idx is not None and entry_slot_idx < len(row):
                try:
                    raw_entry_slot = (row[entry_slot_idx] or "").strip()
                    entry_slot = int(Decimal(raw_entry_slot)) if raw_entry_slot else None
                except (InvalidOperation, ValueError):
                    bad.append((line_no, "invalid entry_slot"))
                    continue

            exit_slot: int | None = None
            if exit_slot_idx is not None and exit_slot_idx < len(row):
                try:
                    raw_exit_slot = (row[exit_slot_idx] or "").strip()
                    exit_slot = int(Decimal(raw_exit_slot)) if raw_exit_slot else None
                except (InvalidOperation, ValueError):
                    bad.append((line_no, "invalid exit_slot"))
                    continue

            hold_slots: int | None = None
            if hold_slots_idx is not None:
                try:
                    raw_hold = (row[hold_slots_idx] if hold_slots_idx < len(row) else "").strip()
                    hold_slots = int(Decimal(raw_hold)) if raw_hold else 0
                except (InvalidOperation, ValueError):
                    bad.append((line_no, "invalid hold_slots"))
                    continue

            mae_pct: float | None = None
            if mae_pct_idx is not None and mae_pct_idx < len(row):
                raw_mae = (row[mae_pct_idx] or "").strip()
                if raw_mae:
                    try:
                        mae_pct = float(raw_mae)
                    except ValueError:
                        bad.append((line_no, "invalid mae_pct"))
                        continue

            mfe_pct: float | None = None
            if mfe_pct_idx is not None and mfe_pct_idx < len(row):
                raw_mfe = (row[mfe_pct_idx] or "").strip()
                if raw_mfe:
                    try:
                        mfe_pct = float(raw_mfe)
                    except ValueError:
                        bad.append((line_no, "invalid max_return_pct_to_exit"))
                        continue

            entry_value_dec: Decimal | None = None
            if entry_idx is not None and entry_idx < len(row):
                raw_entry_value = (row[entry_idx] or "").strip()
                if raw_entry_value:
                    try:
                        entry_value_dec = Decimal(raw_entry_value)
                    except InvalidOperation:
                        if slip_pct != 0 and slippage_method == "entry_exit_values":
                            bad.append((line_no, "invalid entry_value/exit_value"))
                            continue

            exit_value_dec: Decimal | None = None
            if exit_idx is not None and exit_idx < len(row):
                raw_exit_value = (row[exit_idx] or "").strip()
                if raw_exit_value:
                    try:
                        exit_value_dec = Decimal(raw_exit_value)
                    except InvalidOperation:
                        if slip_pct != 0 and slippage_method == "entry_exit_values":
                            bad.append((line_no, "invalid entry_value/exit_value"))
                            continue

            if buy_sol is not None and entry_value_dec is not None and buy_sol > 0:
                sizing_checked += 1
                mismatch = abs(entry_value_dec - buy_sol) / buy_sol
                mismatch_flag = mismatch > Decimal("0.05")
                if first_sizing_mismatch is None:
                    first_sizing_mismatch = mismatch_flag
                if mismatch_flag:
                    sizing_mismatched += 1

            slippage_cost = Decimal("0")
            if slip_pct != 0:
                if slippage_method == "entry_exit_values":
                    if entry_value_dec is None or exit_value_dec is None:
                        bad.append((line_no, "invalid entry_value/exit_value"))
                        continue
                    slippage_cost = slip_pct * (abs(entry_value_dec) + abs(exit_value_dec))
                elif slippage_method == "buy_sol_notional":
                    if buy_sol is None:
                        bad.append((line_no, "missing buy_sol for slippage"))
                        continue
                    slippage_cost = slip_pct * (abs(buy_sol) * 2)

            net_pnl = net_pnl_raw - slippage_cost

            if strategy_idxs:
                vals = tuple(
                    _parse_strategy_cell(field, row[idx] if idx < len(row) else "")
                    for field, idx in zip(strategy_fields, strategy_idxs, strict=False)
                )
                strategy_variants.add(vals)
                if first_strategy is None:
                    first_strategy = vals

            trades.append(
                Trade(
                    line_no=line_no,
                    mint=mint,
                    entry_slot=entry_slot,
                    exit_slot=exit_slot,
                    net_pnl_raw=net_pnl_raw,
                    slippage_cost=slippage_cost,
                    net_pnl=net_pnl,
                    buy_sol=buy_sol,
                    exit_reason=exit_reason,
                    hold_slots=hold_slots,
                    mae_pct=mae_pct,
                    mfe_pct=mfe_pct,
                )
            )

        if bad:
            raise ValueError(f"Failed parsing {len(bad)} row(s); first error at line {bad[0][0]}: {bad[0][1]}")

    if not trades:
        raise ValueError("No trades found (CSV has only header / empty rows).")

    if sizing_checked > 0:
        warn_many = sizing_mismatched >= max(3, int(math.ceil(0.1 * float(sizing_checked))))
        warn_first = bool(first_sizing_mismatch)
        if warn_first or warn_many:
            print(
                "entry_value differs from buy_sol; this usually means the backtest was run with --scale-mode none and buy_sol is not sizing the trade.",
                file=sys.stderr,
            )

    # Infer start balance
    start_sol_source: str
    if start_sol is not None:
        start_sol_source = "--start-sol"
    else:
        if len(buy_sol_values) == 1:
            start_sol = next(iter(buy_sol_values))
            start_sol_source = "buy_sol"
        elif len(buy_sol_values) == 0:
            start_sol = Decimal("2.0")
            start_sol_source = "default_2.0"
        else:
            sorted_vals = ", ".join(sorted((_fmt_decimal(v) for v in buy_sol_values), key=lambda x: Decimal(x)))
            raise ValueError(
                f"Cannot infer start SOL (buy_sol has multiple values: {sorted_vals}). Pass --start-sol explicitly."
            )

    if start_sol <= 0:
        raise ValueError(f"Invalid start SOL (must be > 0): {start_sol}")

    n = len(trades)

    # Basic aggregates
    net_sum_raw = sum((t.net_pnl_raw for t in trades), Decimal("0"))
    slippage_cost_total = sum((t.slippage_cost for t in trades), Decimal("0"))
    net_sum = sum((t.net_pnl for t in trades), Decimal("0"))
    avg_net_pnl_per_launch = (net_sum / Decimal(n)) if n else None

    projections = []
    if avg_net_pnl_per_launch is not None:
        for lpd in launches_per_day:
            per_day = avg_net_pnl_per_launch * Decimal(int(lpd))
            projections.append(
                {
                    "launches_per_day": int(lpd),
                    "avg_net_pnl_per_day": per_day,
                    "avg_net_pnl_per_hour": per_day / Decimal("24"),
                }
            )

    wins = sum(1 for t in trades if t.net_pnl > 0)
    losses = sum(1 for t in trades if t.net_pnl < 0)
    breakeven = n - wins - losses
    win_rate = (Decimal(wins) / Decimal(n)) if n else None

    gross_profit = sum((t.net_pnl for t in trades if t.net_pnl > 0), Decimal("0"))
    gross_loss = sum((t.net_pnl for t in trades if t.net_pnl < 0), Decimal("0"))  # negative

    profit_factor: Decimal | None = None
    if gross_loss < 0:
        profit_factor = gross_profit / (-gross_loss)
    elif gross_profit > 0:
        profit_factor = Decimal("Infinity")

    omega_ratio_0 = profit_factor  # threshold=0 => equals profit factor for discrete pnl samples

    # Realized payoff structure
    wins_sol = sorted(float(t.net_pnl) for t in trades if t.net_pnl > 0)
    losses_sol = sorted(float(t.net_pnl) for t in trades if t.net_pnl < 0)  # negative
    losses_sol_abs = sorted(-x for x in losses_sol)

    avg_win_sol = (gross_profit / Decimal(wins)) if wins else None
    avg_loss_sol = (gross_loss / Decimal(losses)) if losses else None  # negative
    avg_loss_sol_abs = (-avg_loss_sol) if avg_loss_sol is not None else None
    median_win_sol = _percentile(wins_sol, 0.50)
    median_loss_sol_abs = _percentile(losses_sol_abs, 0.50)
    median_loss_sol = -median_loss_sol_abs if median_loss_sol_abs is not None else None
    p90_win_sol = _percentile(wins_sol, 0.90)
    p95_win_sol = _percentile(wins_sol, 0.95)
    p95_loss_sol_abs = _percentile(losses_sol_abs, 0.95)
    p99_loss_sol_abs = _percentile(losses_sol_abs, 0.99)
    p95_loss_sol = -p95_loss_sol_abs if p95_loss_sol_abs is not None else None
    p99_loss_sol = -p99_loss_sol_abs if p99_loss_sol_abs is not None else None

    win_loss_ratio = None
    if avg_win_sol is not None and avg_loss_sol_abs is not None and avg_loss_sol_abs > 0:
        win_loss_ratio = avg_win_sol / avg_loss_sol_abs

    tail_ratio_p95 = None
    if p95_win_sol is not None and p95_loss_sol_abs is not None and p95_loss_sol_abs > 0:
        tail_ratio_p95 = float(p95_win_sol) / float(p95_loss_sol_abs)

    # Exit reason decomposition
    exit_counts: dict[str, int] = {}
    exit_sums: dict[str, Decimal] = {}
    for t in trades:
        key = _classify_exit_reason(t.exit_reason)
        exit_counts[key] = exit_counts.get(key, 0) + 1
        exit_sums[key] = exit_sums.get(key, Decimal("0")) + t.net_pnl
    exit_rates = {k: (v / n) for k, v in exit_counts.items()} if n else {}
    exit_avg_pnl = {k: (exit_sums[k] / Decimal(exit_counts[k])) for k in exit_counts if exit_counts[k] > 0}

    # Sequence + equity-curve metrics (order-sensitive; file/entry_slot/exit_slot)
    seq_trades = trades
    if effective_order_by != "file":
        # Python's sort is stable, so ties preserve the input CSV order.
        seq_trades = list(trades)
        if effective_order_by == "entry_slot":
            seq_trades.sort(key=lambda t: (t.entry_slot is None, t.entry_slot or 0))
        elif effective_order_by == "exit_slot":
            seq_trades.sort(key=lambda t: (t.exit_slot is None, t.exit_slot or 0))

    equity = start_sol
    min_equity = start_sol
    min_line = None
    min_mint = None

    peak_equity = start_sol
    peak_line = None
    peak_mint = None
    peak_trade_index = 0

    max_drawdown_sol = Decimal("0")
    max_dd_peak_equity = start_sol
    max_dd_peak_line = None
    max_dd_peak_mint = None
    max_dd_peak_trade_index = 0
    max_dd_trough_equity = start_sol
    max_dd_trough_line = None
    max_dd_trough_mint = None
    max_dd_trough_trade_index = 0
    max_dd_recovery_trade_index: int | None = None
    max_dd_recovery_line = None
    max_dd_recovery_mint = None

    # Drawdown episodes for avg drawdown + durations
    in_dd = False
    ep_peak_equity = start_sol
    ep_peak_trade_index = 0
    ep_trough_equity = start_sol
    ep_trough_trade_index = 0
    ep_depths_sol: list[Decimal] = []
    ep_depths_pct: list[Decimal] = []

    # Underwater + Ulcer Index
    underwater_trades = 0
    underwater_periods: list[int] = []
    current_underwater_len = 0
    dd_sq_sum = 0.0

    # Streaks and rolling metrics
    longest_losing_streak = 0
    current_losing_streak = 0
    win_flags = []

    pnls_f = [float(t.net_pnl) for t in seq_trades]

    for i, t in enumerate(seq_trades, start=1):
        pnl = t.net_pnl
        equity += pnl

        # min balance
        if equity < min_equity:
            min_equity = equity
            min_line = t.line_no
            min_mint = t.mint

        # streaks
        if pnl < 0:
            current_losing_streak += 1
            longest_losing_streak = max(longest_losing_streak, current_losing_streak)
        else:
            current_losing_streak = 0

        win_flags.append(1 if pnl > 0 else 0)

        # peaks & underwater
        if equity > peak_equity:
            peak_equity = equity
            peak_line = t.line_no
            peak_mint = t.mint
            peak_trade_index = i
            if current_underwater_len:
                underwater_periods.append(current_underwater_len)
                current_underwater_len = 0
            dd_pct = 0.0
        else:
            underwater_trades += 1
            current_underwater_len += 1
            dd_pct = float((peak_equity - equity) / peak_equity) if peak_equity > 0 else 0.0

        dd_sq_sum += dd_pct * dd_pct

        # drawdown episodes
        if not in_dd and equity < peak_equity:
            in_dd = True
            ep_peak_equity = peak_equity
            ep_peak_trade_index = peak_trade_index
            ep_trough_equity = equity
            ep_trough_trade_index = i
        elif in_dd:
            if equity < ep_trough_equity:
                ep_trough_equity = equity
                ep_trough_trade_index = i
            if equity >= ep_peak_equity:
                depth = ep_peak_equity - ep_trough_equity
                ep_depths_sol.append(depth)
                ep_depths_pct.append((depth / ep_peak_equity) if ep_peak_equity > 0 else Decimal("0"))
                in_dd = False

        # max drawdown
        drawdown_sol = peak_equity - equity
        if drawdown_sol > max_drawdown_sol:
            max_drawdown_sol = drawdown_sol
            max_dd_peak_equity = peak_equity
            max_dd_peak_line = peak_line
            max_dd_peak_mint = peak_mint
            max_dd_peak_trade_index = peak_trade_index
            max_dd_trough_equity = equity
            max_dd_trough_line = t.line_no
            max_dd_trough_mint = t.mint
            max_dd_trough_trade_index = i
            max_dd_recovery_trade_index = None
            max_dd_recovery_line = None
            max_dd_recovery_mint = None

        if (
            max_drawdown_sol > 0
            and max_dd_recovery_trade_index is None
            and i > max_dd_trough_trade_index
            and equity >= max_dd_peak_equity
        ):
            max_dd_recovery_trade_index = i
            max_dd_recovery_line = t.line_no
            max_dd_recovery_mint = t.mint

    if current_underwater_len:
        underwater_periods.append(current_underwater_len)

    if in_dd and ep_peak_equity > ep_trough_equity:
        depth = ep_peak_equity - ep_trough_equity
        ep_depths_sol.append(depth)
        ep_depths_pct.append((depth / ep_peak_equity) if ep_peak_equity > 0 else Decimal("0"))

    percent_time_underwater = (underwater_trades / n) if n else None
    ulcer_index = math.sqrt(dd_sq_sum / n) if n else None
    ulcer_index_pct_points = (ulcer_index * 100.0) if ulcer_index is not None else None
    avg_underwater_length = statistics.fmean(underwater_periods) if underwater_periods else None
    max_underwater_length = max(underwater_periods) if underwater_periods else None

    max_drawdown_pct = (max_drawdown_sol / max_dd_peak_equity) if max_dd_peak_equity > 0 else Decimal("0")
    max_drawdown_recovery_trades = (
        (max_dd_recovery_trade_index - max_dd_trough_trade_index) if max_dd_recovery_trade_index is not None else None
    )

    avg_drawdown_sol = (sum(ep_depths_sol, Decimal("0")) / Decimal(len(ep_depths_sol))) if ep_depths_sol else None
    avg_drawdown_pct = (sum(ep_depths_pct, Decimal("0")) / Decimal(len(ep_depths_pct))) if ep_depths_pct else None

    recovery_factor: Decimal | None = None
    if max_drawdown_sol > 0:
        recovery_factor = net_sum / max_drawdown_sol
    elif net_sum > 0:
        recovery_factor = Decimal("Infinity")

    # Rolling win rate (50 trade window)
    rolling_win_rate_50_stats = None
    if n >= 50:
        w = 50
        s = sum(win_flags[:w])
        rwr = [s / w]
        for i in range(w, n):
            s += win_flags[i] - win_flags[i - w]
            rwr.append(s / w)
        rwr_sorted = sorted(rwr)
        rolling_win_rate_50_stats = {
            "min": min(rwr),
            "max": max(rwr),
            "mean": statistics.fmean(rwr),
            "p05": _percentile(rwr_sorted, 0.05),
            "p50": _percentile(rwr_sorted, 0.50),
            "p95": _percentile(rwr_sorted, 0.95),
        }

    # Worst rolling window PnL
    worst_rolling_pnl: dict[int, float] = {}
    for w in (20, 50, 100):
        if n < w:
            continue
        s = sum(pnls_f[:w])
        m = s
        for i in range(w, n):
            s += pnls_f[i] - pnls_f[i - w]
            if s < m:
                m = s
        worst_rolling_pnl[w] = m

    # Throughput-adjusted return
    throughput = None
    if any(t.hold_slots is not None for t in trades) and any(t.buy_sol is not None for t in trades):
        denom_sol_slots = 0.0
        ratios: list[float] = []
        holds: list[int] = []
        for t in trades:
            if t.buy_sol is None or t.hold_slots is None:
                continue
            if t.hold_slots <= 0:
                continue
            buy_f = float(t.buy_sol)
            if not (math.isfinite(buy_f) and buy_f > 0):
                continue
            denom = buy_f * float(t.hold_slots)
            denom_sol_slots += denom
            ratios.append(float(t.net_pnl) / denom)
            holds.append(int(t.hold_slots))
        if denom_sol_slots > 0 and holds:
            ratios_sorted = sorted(ratios)
            holds_sorted = sorted(holds)
            total_pnl_f = float(net_sum)
            slot_minutes = float(slot_ms) / 60000.0
            denom_sol_minutes = denom_sol_slots * slot_minutes
            throughput = {
                "pnl_per_sol_slot_total": total_pnl_f / denom_sol_slots,
                "pnl_per_sol_minute_total": (total_pnl_f / denom_sol_minutes) if denom_sol_minutes > 0 else None,
                "pnl_per_sol_slot_avg": statistics.fmean(ratios),
                "pnl_per_sol_slot_median": _percentile(ratios_sorted, 0.50),
                "pnl_per_sol_slot_p95": _percentile(ratios_sorted, 0.95),
                "hold_slots_median": _percentile([float(h) for h in holds_sorted], 0.50),
                "hold_slots_p95": _percentile([float(h) for h in holds_sorted], 0.95),
                "hold_minutes_median": (_percentile([float(h) for h in holds_sorted], 0.50) or 0.0) * slot_minutes,
                "hold_minutes_p95": (_percentile([float(h) for h in holds_sorted], 0.95) or 0.0) * slot_minutes,
            }

    # MAE/MFE per trade
    mae_vals = [t.mae_pct for t in trades if t.mae_pct is not None]
    mfe_vals = [t.mfe_pct for t in trades if t.mfe_pct is not None]
    mae_mfe = None
    if mae_vals or mfe_vals:
        mae_sorted = sorted(mae_vals)
        mfe_sorted = sorted(mfe_vals)
        mae_mfe = {
            "avg_mae_pct": statistics.fmean(mae_vals) if mae_vals else None,
            "median_mae_pct": _percentile(mae_sorted, 0.50) if mae_sorted else None,
            "p05_mae_pct": _percentile(mae_sorted, 0.05) if mae_sorted else None,
            "avg_mfe_pct": statistics.fmean(mfe_vals) if mfe_vals else None,
            "median_mfe_pct": _percentile(mfe_sorted, 0.50) if mfe_sorted else None,
            "p95_mfe_pct": _percentile(mfe_sorted, 0.95) if mfe_sorted else None,
        }

    # Sharpe/Sortino (per-trade returns vs starting bankroll)
    sharpe_ratio = None
    sortino_ratio = None
    sharpe_annualized: dict[int, float] = {}
    sortino_annualized: dict[int, float] = {}
    rets = [p / float(start_sol) for p in pnls_f]
    if rets and float(start_sol) > 0:
        mean_r = statistics.fmean(rets)
        std_r = statistics.pstdev(rets) if len(rets) > 1 else 0.0
        if std_r > 0:
            sharpe_ratio = mean_r / std_r
        downside = math.sqrt(sum((min(r, 0.0) ** 2) for r in rets) / float(len(rets)))
        if downside > 0:
            sortino_ratio = mean_r / downside
        for lpd in launches_per_day:
            tpy = float(lpd) * 365.0
            scale = math.sqrt(tpy) if tpy > 0 else 0.0
            if sharpe_ratio is not None and math.isfinite(sharpe_ratio):
                sharpe_annualized[int(lpd)] = sharpe_ratio * scale
            if sortino_ratio is not None and math.isfinite(sortino_ratio):
                sortino_annualized[int(lpd)] = sortino_ratio * scale

    # Calmar ratio per scenario (annualized return / max drawdown pct)
    calmar_by_lpd: dict[int, Decimal] = {}
    if avg_net_pnl_per_launch is not None and max_drawdown_pct > 0:
        for lpd in launches_per_day:
            annual_net = avg_net_pnl_per_launch * Decimal(int(lpd)) * Decimal("365")
            annual_return = annual_net / start_sol
            calmar_by_lpd[int(lpd)] = annual_return / max_drawdown_pct
    elif avg_net_pnl_per_launch is not None and max_drawdown_pct == 0 and net_sum > 0:
        for lpd in launches_per_day:
            calmar_by_lpd[int(lpd)] = Decimal("Infinity")

    # Time to recovery estimates for the max drawdown episode
    recovery_time_by_lpd: dict[int, dict[str, Decimal]] = {}
    if max_drawdown_recovery_trades is not None:
        rt = Decimal(int(max_drawdown_recovery_trades))
        for lpd in launches_per_day:
            lpd_d = Decimal(int(lpd))
            days = (rt / lpd_d) if lpd_d > 0 else Decimal("0")
            recovery_time_by_lpd[int(lpd)] = {"days": days, "hours": days * Decimal("24")}

    # Risk of ruin (Brownian approximation on per-trade pnl)
    ruin_prob_brownian = None
    bankroll_for_ruin_prob_1pct_brownian = None
    bankroll_for_ruin_prob_0p1pct_brownian = None
    if pnls_f:
        mu = statistics.fmean(pnls_f)
        var = statistics.pvariance(pnls_f) if len(pnls_f) > 1 else 0.0
        if var == 0.0:
            ruin_prob_brownian = 1.0 if mu <= 0 else 0.0
        elif mu <= 0:
            ruin_prob_brownian = 1.0
        else:
            b = float(start_sol)
            ruin_prob_brownian = math.exp(-2.0 * mu * b / var)
            bankroll_for_ruin_prob_1pct_brownian = -(var / (2.0 * mu)) * math.log(0.01)
            bankroll_for_ruin_prob_0p1pct_brownian = -(var / (2.0 * mu)) * math.log(0.001)

    # Kelly criterion (heuristics)
    kelly: dict[str, float] = {}
    if avg_win_sol is not None and avg_loss_sol_abs is not None and avg_loss_sol_abs > 0 and n:
        p = wins / n
        q = 1.0 - p
        b = float(avg_win_sol / avg_loss_sol_abs)
        kelly["binary"] = max(0.0, min(1.0, p - (q / b))) if b > 0 else 0.0
    if pnls_f:
        mu = statistics.fmean(pnls_f)
        var = statistics.pvariance(pnls_f) if len(pnls_f) > 1 else 0.0
        if var > 0:
            f = (mu * float(start_sol)) / var
            kelly["mean_var"] = max(0.0, min(1.0, f))

    # Monte Carlo: shuffle trade order to estimate path risk
    mc = None
    if int(mc_runs) > 0 and pnls_f:
        rng = random.Random(mc_seed)
        seq = pnls_f.copy()
        max_dds: list[float] = []
        min_bals: list[float] = []
        required_starts: list[float] = []
        neg_count = 0
        start_f = float(start_sol)

        for _ in range(int(mc_runs)):
            rng.shuffle(seq)
            cum = 0.0
            peak = 0.0
            min_cum = 0.0
            max_dd = 0.0
            for pnl in seq:
                cum += pnl
                if cum > peak:
                    peak = cum
                dd = peak - cum
                if dd > max_dd:
                    max_dd = dd
                if cum < min_cum:
                    min_cum = cum

            max_dds.append(max_dd)
            min_bal = start_f + min_cum
            min_bals.append(min_bal)
            required_starts.append(max(0.0, -min_cum))
            if min_bal < 0:
                neg_count += 1

        max_dds.sort()
        min_bals.sort()
        required_starts.sort()
        mc = {
            "runs": int(mc_runs),
            "seed": mc_seed,
            "prob_negative_balance": neg_count / float(mc_runs),
            "max_drawdown_sol_mean": statistics.fmean(max_dds) if max_dds else None,
            "max_drawdown_sol_p50": _percentile(max_dds, 0.50),
            "max_drawdown_sol_p95": _percentile(max_dds, 0.95),
            "min_running_balance_p01": _percentile(min_bals, 0.01),
            "min_running_balance_p05": _percentile(min_bals, 0.05),
            "min_running_balance_p10": _percentile(min_bals, 0.10),
            "required_bankroll_p99_for_ruin_lt_1pct": _percentile(required_starts, 0.99),
            "required_bankroll_p999_for_ruin_lt_0p1pct": _percentile(required_starts, 0.999),
        }

    return {
        "path": path,
        "order_by": effective_order_by,
        "rows": n,
        "start_sol": start_sol,
        "start_sol_source": start_sol_source,
        "slippage_bps": int(slippage_bps),
        "slippage_method": slippage_method,
        "slot_ms": float(slot_ms),
        "strategy_fields": strategy_fields,
        "strategy_variants": len(strategy_variants),
        "strategy": dict(zip(strategy_fields, next(iter(strategy_variants)), strict=False)) if len(strategy_variants) == 1 else None,
        "strategy_sample": dict(zip(strategy_fields, first_strategy, strict=False)) if first_strategy is not None else None,
        "net_pnl_sum_raw": net_sum_raw,
        "slippage_cost_total": slippage_cost_total,
        "net_pnl_sum": net_sum,
        "avg_net_pnl_per_launch": avg_net_pnl_per_launch,
        "projections": projections,
        "end_sol": start_sol + net_sum,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "omega_ratio_0": omega_ratio_0,
        "avg_win_sol": avg_win_sol,
        "avg_loss_sol": avg_loss_sol,
        "avg_loss_sol_abs": avg_loss_sol_abs,
        "median_win_sol": median_win_sol,
        "median_loss_sol_abs": median_loss_sol_abs,
        "median_loss_sol": median_loss_sol,
        "p90_win_sol": p90_win_sol,
        "p95_win_sol": p95_win_sol,
        "p95_loss_sol_abs": p95_loss_sol_abs,
        "p99_loss_sol_abs": p99_loss_sol_abs,
        "p95_loss_sol": p95_loss_sol,
        "p99_loss_sol": p99_loss_sol,
        "win_loss_ratio": win_loss_ratio,
        "tail_ratio_p95": tail_ratio_p95,
        "exit_counts": exit_counts,
        "exit_rates": exit_rates,
        "exit_avg_pnl": exit_avg_pnl,
        "longest_losing_streak": longest_losing_streak,
        "rolling_win_rate_50_stats": rolling_win_rate_50_stats,
        "worst_rolling_pnl": worst_rolling_pnl,
        "percent_time_underwater": percent_time_underwater,
        "underwater_periods": len(underwater_periods),
        "avg_underwater_length": avg_underwater_length,
        "max_underwater_length": max_underwater_length,
        "ulcer_index": ulcer_index,
        "ulcer_index_pct_points": ulcer_index_pct_points,
        "min_running_balance": min_equity,
        "min_running_balance_line": min_line,
        "min_running_balance_mint": min_mint,
        "max_drawdown_sol": max_drawdown_sol,
        "max_drawdown_pct": max_drawdown_pct,
        "max_drawdown_peak_balance": max_dd_peak_equity,
        "max_drawdown_peak_line": max_dd_peak_line,
        "max_drawdown_peak_mint": max_dd_peak_mint,
        "max_drawdown_trough_balance": max_dd_trough_equity,
        "max_drawdown_trough_line": max_dd_trough_line,
        "max_drawdown_trough_mint": max_dd_trough_mint,
        "max_drawdown_recovery_trades": max_drawdown_recovery_trades,
        "max_drawdown_recovery_line": max_dd_recovery_line,
        "max_drawdown_recovery_mint": max_dd_recovery_mint,
        "avg_drawdown_sol": avg_drawdown_sol,
        "avg_drawdown_pct": avg_drawdown_pct,
        "recovery_factor": recovery_factor,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "sharpe_annualized": sharpe_annualized,
        "sortino_annualized": sortino_annualized,
        "calmar_by_lpd": calmar_by_lpd,
        "recovery_time_by_lpd": recovery_time_by_lpd,
        "throughput": throughput,
        "mae_mfe": mae_mfe,
        "ruin_prob_brownian": ruin_prob_brownian,
        "bankroll_for_ruin_prob_1pct_brownian": bankroll_for_ruin_prob_1pct_brownian,
        "bankroll_for_ruin_prob_0p1pct_brownian": bankroll_for_ruin_prob_0p1pct_brownian,
        "kelly": kelly,
        "monte_carlo": mc,
    }


def _print_kv(key: str, value: Any) -> None:
    if value is None:
        print(f"{key}: NA")
        return
    if isinstance(value, Decimal):
        if value.is_infinite():
            print(f"{key}: inf")
        else:
            print(f"{key}: {_fmt_decimal(value)}")
        return
    if isinstance(value, float):
        print(f"{key}: {_fmt_float(value)}")
        return
    print(f"{key}: {value}")


class _Tee:
    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Compute strategy metrics from a strat.csv file.")
    ap.add_argument(
        "csv_path",
        nargs="?",
        default="section2_results/strat.csv",
        help="Path to strat.csv (default: section2_results/strat.csv)",
    )
    ap.add_argument(
        "--start-sol",
        default="auto",
        help="Starting SOL balance. Use a number, or 'auto' to infer from unique buy_sol in the CSV (default: auto).",
    )
    ap.add_argument("--net-pnl-col", default="net_pnl", help="Column name for net pnl (default: net_pnl)")
    ap.add_argument("--mint-col", default="mint", help="Optional mint column name (default: mint)")
    ap.add_argument(
        "--launches-per-day",
        default="750,480,288",
        help="Comma list of assumed launches per day for SOL/hour and SOL/day projections (default: 750,480,288)",
    )
    ap.add_argument(
        "--mc-runs",
        type=int,
        default=10000,
        help="Monte Carlo runs (shuffle trade order); 0 disables (default: 10000).",
    )
    ap.add_argument(
        "--mc-seed",
        type=int,
        default=1,
        help="Monte Carlo RNG seed (default: 1).",
    )
    ap.add_argument(
        "--slippage-bps",
        type=int,
        default=100,
        help="Per-swap slippage in bps. Applied on entry and exit (default: 100). Use 0 to disable.",
    )
    ap.add_argument(
        "--slot-ms",
        type=float,
        default=400.0,
        help="Slot duration in ms for SOL-minute throughput conversions (default: 400).",
    )
    ap.add_argument(
        "--order-by",
        choices=["file", "entry_slot", "exit_slot"],
        default=None,
        help="Trade ordering for sequence-based metrics. Default: entry_slot if present, else exit_slot, else file.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional output file path for the report. When set, output is also written to this file.",
    )
    args = ap.parse_args(argv)

    start_sol_raw = str(args.start_sol).strip()
    if start_sol_raw.lower() == "auto":
        start_sol: Decimal | None = None
    else:
        try:
            start_sol = Decimal(start_sol_raw)
        except InvalidOperation:
            print(f"Invalid --start-sol value: {args.start_sol!r}", file=sys.stderr)
            return 2

    try:
        launches_per_day = _parse_int_list(args.launches_per_day)
    except ValueError:
        print(f"Invalid --launches-per-day value: {args.launches_per_day!r}", file=sys.stderr)
        return 2

    mint_col = (args.mint_col or "").strip() or None

    try:
        out = calculate(
            args.csv_path,
            start_sol=start_sol,
            net_pnl_col=args.net_pnl_col.strip(),
            mint_col=mint_col,
            launches_per_day=launches_per_day,
            mc_runs=int(args.mc_runs),
            mc_seed=int(args.mc_seed) if int(args.mc_runs) > 0 else None,
            slippage_bps=int(args.slippage_bps),
            slot_ms=float(args.slot_ms),
            order_by=args.order_by,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    def emit_output() -> None:
        _print_kv("path", out["path"])
        print(f"start_balance: {_fmt_decimal(out['start_sol'])} ({out['start_sol_source']})")
        _print_kv("slippage_bps", out["slippage_bps"])
        _print_kv("slippage_method", out["slippage_method"])
        _print_kv("slot_ms", out["slot_ms"])
        _print_kv("order_by", out.get("order_by"))
    
        if out.get("strategy_fields"):
            if out.get("strategy") is not None:
                fields = list(out["strategy"].keys())
                vals = tuple(str(out["strategy"][k]) for k in fields)
                _print_kv("strategy", _format_strategy(fields, vals))
            elif out.get("strategy_sample") is not None:
                fields = list(out["strategy_sample"].keys())
                vals = tuple(str(out["strategy_sample"][k]) for k in fields)
                _print_kv("strategy", _format_strategy(fields, vals))
                _print_kv("strategy_variants", out.get("strategy_variants"))
    
        _print_kv("rows", out["rows"])
        _print_kv("net_pnl_sum_raw", out["net_pnl_sum_raw"])
        _print_kv("slippage_cost_total", out["slippage_cost_total"])
        _print_kv("net_pnl_sum", out["net_pnl_sum"])
        _print_kv("avg_net_pnl_per_launch", out["avg_net_pnl_per_launch"])
    
        for proj in out.get("projections", []):
            lpd = proj["launches_per_day"]
            _print_kv(f"avg_net_pnl_per_day@{lpd}_launches", proj["avg_net_pnl_per_day"])
            _print_kv(f"avg_net_pnl_per_hour@{lpd}_launches", proj["avg_net_pnl_per_hour"])
    
        _print_kv("wins", out["wins"])
        _print_kv("losses", out["losses"])
        _print_kv("breakeven", out["breakeven"])
        _print_kv("win_rate", out["win_rate"])
        if out.get("win_rate") is not None:
            print(f"win_rate_pct: {_fmt_percent(out['win_rate'])}%")
    
        _print_kv("avg_win_sol", out["avg_win_sol"])
        _print_kv("avg_loss_sol", out["avg_loss_sol"])
        _print_kv("avg_loss_sol_abs", out.get("avg_loss_sol_abs"))
        _print_kv("median_win_sol", out["median_win_sol"])
        _print_kv("median_loss_sol_abs", out["median_loss_sol_abs"])
        _print_kv("median_loss_sol", out.get("median_loss_sol"))
        _print_kv("p90_win_sol", out["p90_win_sol"])
        _print_kv("p95_win_sol", out["p95_win_sol"])
        _print_kv("p95_loss_sol_abs", out["p95_loss_sol_abs"])
        _print_kv("p95_loss_sol", out.get("p95_loss_sol"))
        _print_kv("p99_loss_sol_abs", out["p99_loss_sol_abs"])
        _print_kv("p99_loss_sol", out.get("p99_loss_sol"))
        _print_kv("win_loss_ratio", out["win_loss_ratio"])
        _print_kv("tail_ratio_p95", out["tail_ratio_p95"])
        _print_kv("tail_ratio", out["tail_ratio_p95"])
    
        _print_kv("profit_factor", out["profit_factor"])
        _print_kv("omega_ratio_0", out["omega_ratio_0"])
        _print_kv("omega_ratio", out["omega_ratio_0"])
        _print_kv("recovery_factor", out["recovery_factor"])
        _print_kv("longest_losing_streak", out["longest_losing_streak"])
    
        # Stop/timeout decomposition (print key rates if present)
        for k in ("TP", "SL", "TIMEOUT", "END"):
            if k in out.get("exit_rates", {}):
                _print_kv(f"{k.lower()}_rate", out["exit_rates"][k])
        for k in ("TP", "SL", "TIMEOUT", "END"):
            if k in out.get("exit_avg_pnl", {}):
                _print_kv(f"avg_net_pnl_{k}", out["exit_avg_pnl"][k])
    
        _print_kv("percent_time_underwater", out["percent_time_underwater"])
        if out.get("percent_time_underwater") is not None:
            print(f"percent_time_underwater_pct: {_fmt_float(out['percent_time_underwater'] * 100.0)}%")
        _print_kv("underwater_periods", out["underwater_periods"])
        _print_kv("avg_underwater_length", out["avg_underwater_length"])
        _print_kv("max_underwater_length", out["max_underwater_length"])
        _print_kv("drawdown_duration_avg_trades", out["avg_underwater_length"])
        _print_kv("drawdown_duration_max_trades", out["max_underwater_length"])
        _print_kv("ulcer_index", out["ulcer_index"])
        _print_kv("ulcer_index_pct_points", out.get("ulcer_index_pct_points"))
    
        rwr = out.get("rolling_win_rate_50_stats") or {}
        for k in ("min", "max", "mean", "p05", "p50", "p95"):
            if k in rwr and rwr[k] is not None:
                _print_kv(f"rolling_win_rate_50_{k}", float(rwr[k]))
    
        for win_len, v in (out.get("worst_rolling_pnl") or {}).items():
            _print_kv(f"worst_rolling_pnl_{win_len}", float(v))
    
        _print_kv("min_running_balance", out["min_running_balance"])
        _print_kv("min_running_balance_line", out["min_running_balance_line"])
        _print_kv("min_running_balance_mint", out["min_running_balance_mint"])
    
        _print_kv("max_drawdown_sol", out["max_drawdown_sol"])
        _print_kv("max_drawdown_pct", out["max_drawdown_pct"])
        if isinstance(out.get("max_drawdown_pct"), Decimal):
            print(f"max_drawdown_pct_pct: {_fmt_percent(out['max_drawdown_pct'])}%")
        _print_kv("max_drawdown_peak_line", out["max_drawdown_peak_line"])
        _print_kv("max_drawdown_peak_mint", out["max_drawdown_peak_mint"])
        _print_kv("max_drawdown_trough_line", out["max_drawdown_trough_line"])
        _print_kv("max_drawdown_trough_mint", out["max_drawdown_trough_mint"])
        _print_kv("max_drawdown_recovery_trades", out["max_drawdown_recovery_trades"])
        _print_kv("max_drawdown_recovery_line", out["max_drawdown_recovery_line"])
        _print_kv("max_drawdown_recovery_mint", out["max_drawdown_recovery_mint"])
    
        _print_kv("avg_drawdown_sol", out["avg_drawdown_sol"])
        _print_kv("avg_drawdown_pct", out["avg_drawdown_pct"])
        if isinstance(out.get("avg_drawdown_pct"), Decimal):
            print(f"avg_drawdown_pct_pct: {_fmt_percent(out['avg_drawdown_pct'])}%")
    
        _print_kv("sharpe_ratio", out["sharpe_ratio"])
        _print_kv("sortino_ratio", out["sortino_ratio"])
        for lpd, v in (out.get("sharpe_annualized") or {}).items():
            _print_kv(f"sharpe_ratio_annualized@{lpd}_launches", float(v))
        for lpd, v in (out.get("sortino_annualized") or {}).items():
            _print_kv(f"sortino_ratio_annualized@{lpd}_launches", float(v))
    
        for lpd, calmar in (out.get("calmar_by_lpd") or {}).items():
            _print_kv(f"calmar_ratio@{lpd}_launches", calmar)
        for lpd, t in (out.get("recovery_time_by_lpd") or {}).items():
            _print_kv(f"time_to_recovery_days@{lpd}_launches", t["days"])
            _print_kv(f"time_to_recovery_hours@{lpd}_launches", t["hours"])
    
        thr = out.get("throughput") or {}
        if thr:
            _print_kv("pnl_per_sol_slot_total", thr.get("pnl_per_sol_slot_total"))
            _print_kv("pnl_per_sol_minute_total", thr.get("pnl_per_sol_minute_total"))
            _print_kv("pnl_per_sol_slot_avg", thr.get("pnl_per_sol_slot_avg"))
            _print_kv("pnl_per_sol_slot_median", thr.get("pnl_per_sol_slot_median"))
            _print_kv("pnl_per_sol_slot_p95", thr.get("pnl_per_sol_slot_p95"))
            _print_kv("hold_slots_median", thr.get("hold_slots_median"))
            _print_kv("hold_slots_p95", thr.get("hold_slots_p95"))
            _print_kv("hold_minutes_median", thr.get("hold_minutes_median"))
            _print_kv("hold_minutes_p95", thr.get("hold_minutes_p95"))
    
        mm = out.get("mae_mfe") or {}
        if mm:
            _print_kv("avg_mae_pct", mm.get("avg_mae_pct"))
            _print_kv("median_mae_pct", mm.get("median_mae_pct"))
            _print_kv("p05_mae_pct", mm.get("p05_mae_pct"))
            _print_kv("avg_mfe_pct", mm.get("avg_mfe_pct"))
            _print_kv("median_mfe_pct", mm.get("median_mfe_pct"))
            _print_kv("p95_mfe_pct", mm.get("p95_mfe_pct"))
    
        _print_kv("ruin_prob_brownian", out.get("ruin_prob_brownian"))
        _print_kv("bankroll_for_ruin_prob_1pct_brownian", out.get("bankroll_for_ruin_prob_1pct_brownian"))
        _print_kv("bankroll_for_ruin_prob_0p1pct_brownian", out.get("bankroll_for_ruin_prob_0p1pct_brownian"))
    
        kelly = out.get("kelly") or {}
        if kelly:
            _print_kv("kelly_binary", kelly.get("binary"))
            _print_kv("kelly_mean_var", kelly.get("mean_var"))
    
        mc = out.get("monte_carlo")
        if mc is None:
            _print_kv("monte_carlo_runs", 0)
        else:
            _print_kv("monte_carlo_runs", mc.get("runs"))
            _print_kv("monte_carlo_seed", mc.get("seed"))
            _print_kv("monte_carlo_prob_negative_balance", mc.get("prob_negative_balance"))
            _print_kv("monte_carlo_max_drawdown_sol_mean", mc.get("max_drawdown_sol_mean"))
            _print_kv("monte_carlo_max_drawdown_sol_p50", mc.get("max_drawdown_sol_p50"))
            _print_kv("monte_carlo_max_drawdown_sol_p95", mc.get("max_drawdown_sol_p95"))
            _print_kv("monte_carlo_min_running_balance_p01", mc.get("min_running_balance_p01"))
            _print_kv("monte_carlo_min_running_balance_p05", mc.get("min_running_balance_p05"))
            _print_kv("monte_carlo_min_running_balance_p10", mc.get("min_running_balance_p10"))
            _print_kv("required_bankroll_p99_for_ruin_lt_1pct", mc.get("required_bankroll_p99_for_ruin_lt_1pct"))
            _print_kv("required_bankroll_p999_for_ruin_lt_0p1pct", mc.get("required_bankroll_p999_for_ruin_lt_0p1pct"))
    
        _print_kv("end_balance", out["end_sol"])

    out_file = None
    saved_stdout = None
    if args.out:
        try:
            out_file = open(args.out, "w", encoding="utf-8")
        except OSError as exc:
            print(f"Error writing --out file {args.out!r}: {exc}", file=sys.stderr)
            return 1
        saved_stdout = sys.stdout
        sys.stdout = _Tee(sys.stdout, out_file)
    try:
        emit_output()
    finally:
        if out_file is not None:
            sys.stdout = saved_stdout
            out_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
