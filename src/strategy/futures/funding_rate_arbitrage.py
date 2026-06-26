"""
Institutional Funding Capture Model — Binance USDT-M 永续合约

五层架构:
  Layer 1  Regime Engine        — 市场状态识别 (4 regime)
  Layer 2  Funding Alpha Engine — 资金费率预测 + 持续性 + 衰减半衰期
  Layer 3  Survival & Risk      — 生存概率 + 尾部风险 + 爆仓缓冲
  Layer 4  Execution Engine     — 自适应杠杆 + EV 驱动仓位 + 信号生成
  Layer 5  Multi-State EV       — 5 状态概率加权期望收益

核心理念: 将 funding 视为 carry 资产单独建模，而非方向策略的附加奖励。
最大化 Funding Carry Sharpe，最小化方向暴露与尾部风险。
"""

from typing import List, Dict, Any, Optional
from pydantic import Field
import math
import numpy as np
import pandas as pd

from src.strategy.types import Strategy
from src.factor import factor_manager
from src.strategy.futures._utils import (
    PRICE_COL, DEFAULT_TAKER_FEE, safe_float,
    calc_position_size, build_order, compute_trailing_stop, build_factors_json,
)

_SETTLEMENT_HOURS_UTC = (0, 8, 16)

# ── Regime constants ──
REGIME_MEAN_REVERTING = "mean_reverting"
REGIME_TREND_EXPANSION = "trend_expansion"
REGIME_FUNDING_EXTREME = "funding_extreme"
REGIME_LIQUIDITY_SHOCK = "liquidity_shock"


class FundingRateArbitrageStrategy(Strategy):
    """Institutional Funding Capture — 5-layer carry-driven model."""

    name: str = Field(default="funding_rate_arbitrage")
    description: str = Field(default="机构级资金费率捕获 — 5 层 carry 驱动模型")
    factor_names: List[str] = Field(
        default=["ema", "rsi", "atr", "adx",
                 "funding_percentile", "atr_percentile",
                 "funding_volatility", "price_acceleration",
                 "zscore_price", "er"],
    )

    # ── Trading params ──
    symbol: str = Field(default="BTCUSDT")
    quantity_precision: int = Field(default=3)
    min_quantity: float = Field(default=0.02)

    # ── Position management ──
    risk_per_trade: float = Field(default=0.015)
    atr_sl_multiplier: float = Field(default=2.0)
    atr_tp_multiplier: float = Field(default=3.5)
    min_leverage: int = Field(default=1)
    max_leverage: int = Field(default=5)
    min_notional_pct: float = Field(default=0.05)
    max_notional_pct: float = Field(default=0.40)
    fee_rate: float = Field(default=DEFAULT_TAKER_FEE)
    margin_buffer_ratio: float = Field(default=0.97)

    # ── Settlement windows ──
    entry_window_minutes: int = Field(default=90)
    exit_after_minutes: int = Field(default=30)
    allow_multi_settlement: bool = Field(default=False)

    # ── L1: Regime thresholds ──
    regime_adx_trend: float = Field(default=35.0)
    regime_accel_mult: float = Field(default=2.0)
    regime_funding_zscore: float = Field(default=2.0)
    regime_atr_spike_ratio: float = Field(default=2.0)

    # ── L2: Funding alpha params ──
    funding_ewma_span: int = Field(default=8)
    funding_momentum_window: int = Field(default=3)
    entry_threshold: float = Field(default=0.00003)
    strong_threshold: float = Field(default=0.00008)
    near_settlement_relax: float = Field(default=0.5)
    use_estimated_funding: bool = Field(default=False)
    funding_pct_period: int = Field(default=500)
    funding_pct_threshold: float = Field(default=0.75)
    funding_vol_period: int = Field(default=500)
    funding_vol_unstable_mult: float = Field(default=3.0, description="funding_vol > mult * entry_threshold → unstable")
    max_rate_history: int = Field(default=50)

    # ── L3: Survival & risk params ──
    survive_base: float = Field(default=0.70, description="Base survival probability")
    survive_sl_weight: float = Field(default=0.15)
    survive_time_weight: float = Field(default=0.10)
    survive_vol_weight: float = Field(default=0.10)
    survive_adx_weight: float = Field(default=0.05)
    survive_min_threshold: float = Field(default=0.40, description="Min p_survive to enter")
    tail_risk_base: float = Field(default=0.02)
    tail_risk_max: float = Field(default=0.15)
    tail_risk_threshold: float = Field(default=0.10, description="p_tail > this → no entry")
    liq_buffer_min: float = Field(default=3.0, description="Min liquidation buffer in ATR units")

    # ── L4: Execution params ──
    adaptive_lev_base: float = Field(default=2.0)
    ev_scale_max: float = Field(default=1.0, description="Max EV-based sizing multiplier")
    ev_threshold: float = Field(default=0.0, description="Min EV to enter")

    # ── L5: Multi-state EV params ──
    p_decay_base: float = Field(default=0.10, description="Base probability of funding decay exit")

    # ── Mean-reversion confirmation ──
    ema_fast_period: int = Field(default=20)
    zscore_period: int = Field(default=20)
    zscore_entry: float = Field(default=1.0)
    er_period: int = Field(default=60)
    er_max_for_fade: float = Field(default=0.65)

    # ── Trend protection ──
    ema_trend_period: int = Field(default=50)
    rsi_period: int = Field(default=14)
    rsi_extreme_long: float = Field(default=25.0)
    rsi_extreme_short: float = Field(default=75.0)
    atr_period: int = Field(default=14)
    atr_pct_window: int = Field(default=200)
    atr_pct_min: float = Field(default=0.05)
    atr_pct_max: float = Field(default=0.95)
    adx_period: int = Field(default=14)
    adx_extreme_threshold: float = Field(default=50.0)
    price_accel_period: int = Field(default=14)

    # ── Risk controls ──
    max_hold_bars: int = Field(default=510)
    min_hold_bars: int = Field(default=3)
    cooldown_bars: int = Field(default=60)
    max_consecutive_losses: int = Field(default=3)
    loss_pause_timeout: int = Field(default=240)
    trailing_trigger_atr: float = Field(default=1.5)
    trailing_pullback_atr: float = Field(default=0.6)

    # ── Flip handling ──
    flip_mode: str = Field(default="close_then_wait")
    reverse_cooldown_bars: int = Field(default=60)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._hold_bars: int = 0
        self._last_side: int = 0
        self._cooldown_remaining: int = 0
        self._consecutive_losses: int = 0
        self._loss_pause_bars: int = 0
        self._last_position: float = 0.0
        self._entry_price: Optional[float] = None
        self._best_price: Optional[float] = None
        self._settlements_collected: int = 0
        self._last_minutes_to_settlement: int = 999
        self._funding_rate_history: list = []
        self._reverse_cooldown_remaining: int = 0
        self._prev_atr: float = 0.0
        self._current_regime: str = REGIME_MEAN_REVERTING

    # ══════════════════════════════════════════════════════════════
    # Time utilities
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _get_current_time(df: pd.DataFrame) -> Optional[pd.Timestamp]:
        if len(df) == 0:
            return None
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index[-1]
        if "timestamp" in df.columns:
            try:
                return pd.Timestamp(df["timestamp"].iloc[-1])
            except Exception:
                return None
        return None

    @staticmethod
    def _minutes_to_next_settlement(ts: pd.Timestamp) -> int:
        current_minutes = ts.hour * 60 + ts.minute
        for h in _SETTLEMENT_HOURS_UTC:
            sm = h * 60
            if sm > current_minutes:
                return sm - current_minutes
        return 24 * 60 - current_minutes

    @staticmethod
    def _minutes_since_last_settlement(ts: pd.Timestamp) -> int:
        current_minutes = ts.hour * 60 + ts.minute
        for h in reversed(_SETTLEMENT_HOURS_UTC):
            sm = h * 60
            if sm <= current_minutes:
                return current_minutes - sm
        return current_minutes + (24 * 60 - _SETTLEMENT_HOURS_UTC[-1] * 60)

    def _effective_entry_threshold(self, minutes_to_next: int) -> float:
        if self.entry_window_minutes <= 0:
            return self.entry_threshold
        clamped = max(0, min(self.entry_window_minutes, minutes_to_next))
        proximity = 1.0 - (clamped / self.entry_window_minutes)
        relax = 1.0 - (1.0 - self.near_settlement_relax) * proximity
        relax = max(self.near_settlement_relax, min(1.0, relax))
        return self.entry_threshold * relax

    # ══════════════════════════════════════════════════════════════
    # Layer 1: Regime Engine
    # ══════════════════════════════════════════════════════════════

    def _detect_regime(
        self, adx: float, atr: float, price_accel: float,
        funding_rate: float, funding_vol: float,
    ) -> str:
        atr_spike = (
            self._prev_atr > 0
            and atr / self._prev_atr > self.regime_atr_spike_ratio
        )
        if atr_spike:
            return REGIME_LIQUIDITY_SHOCK

        if adx > self.regime_adx_trend and abs(price_accel) > self.regime_accel_mult * atr:
            return REGIME_TREND_EXPANSION

        if funding_vol > 0:
            funding_z = abs(funding_rate) / funding_vol
            if funding_z > self.regime_funding_zscore:
                return REGIME_FUNDING_EXTREME

        return REGIME_MEAN_REVERTING

    # ══════════════════════════════════════════════════════════════
    # Layer 2: Funding Alpha Engine
    # ══════════════════════════════════════════════════════════════

    def _update_rate_history(self, rate: float):
        if (not self._funding_rate_history
                or abs(rate - self._funding_rate_history[-1]) > 1e-10):
            self._funding_rate_history.append(rate)
            if len(self._funding_rate_history) > self.max_rate_history:
                self._funding_rate_history = self._funding_rate_history[-self.max_rate_history:]

    def _predict_funding(self) -> float:
        """EWMA + momentum extrapolation of next funding rate."""
        h = self._funding_rate_history
        if len(h) < 2:
            return h[-1] if h else 0.0
        span = min(self.funding_ewma_span, len(h))
        alpha = 2.0 / (span + 1)
        ewma = h[0]
        for v in h[1:]:
            ewma = alpha * v + (1 - alpha) * ewma
        mom_w = min(self.funding_momentum_window, len(h) - 1)
        if mom_w >= 1:
            momentum = (h[-1] - h[-1 - mom_w]) / mom_w
        else:
            momentum = 0.0
        return ewma + momentum

    def _funding_persistence_prob(self) -> float:
        """P(|funding_next| > entry_threshold), estimated from history."""
        h = self._funding_rate_history
        if len(h) < 3:
            return 0.5
        count = sum(1 for r in h if abs(r) > self.entry_threshold)
        return count / len(h)

    def _funding_half_life(self) -> float:
        """Estimate AR(1) half-life of funding rate series.
        Returns bars until extreme funding decays by half. Clamped to [1, 100]."""
        h = self._funding_rate_history
        if len(h) < 5:
            return 10.0
        y = np.array(h[1:], dtype=float)
        x = np.array(h[:-1], dtype=float)
        var_x = np.var(x)
        if var_x < 1e-20:
            return 10.0
        phi = float(np.cov(y, x)[0, 1] / var_x)
        phi = max(-0.99, min(0.99, phi))
        if abs(phi) < 0.01:
            return 100.0
        hl = -math.log(2) / math.log(abs(phi))
        return max(1.0, min(100.0, hl))

    def _funding_regime_stable(self, funding_vol: float) -> bool:
        """Returns False if funding volatility is too high for reliable carry."""
        return funding_vol <= self.entry_threshold * self.funding_vol_unstable_mult

    def _rate_avg(self) -> float:
        if not self._funding_rate_history:
            return 0.0
        return sum(self._funding_rate_history) / len(self._funding_rate_history)

    # ══════════════════════════════════════════════════════════════
    # Layer 3: Survival & Risk Engine
    # ══════════════════════════════════════════════════════════════

    def _survival_probability(
        self, sl_distance: float, atr: float, minutes_to_settlement: int,
        adx: float, atr_pct: float,
    ) -> float:
        """Heuristic logistic model for P(hold to settlement without SL hit)."""
        p = self.survive_base

        if atr > 0 and sl_distance > 0:
            sl_atr_ratio = sl_distance / atr
            sl_bonus = min(sl_atr_ratio / 4.0, 1.0) * self.survive_sl_weight
            p += sl_bonus

        if minutes_to_settlement <= self.entry_window_minutes:
            time_bonus = (1.0 - minutes_to_settlement / max(1, self.entry_window_minutes))
            p += time_bonus * self.survive_time_weight

        vol_bonus = (1.0 - min(atr_pct, 1.0)) * self.survive_vol_weight
        p += vol_bonus

        adx_bonus = max(0.0, (50.0 - adx) / 50.0) * self.survive_adx_weight
        p += adx_bonus

        return max(0.05, min(0.95, p))

    def _tail_risk_prob(
        self, atr_pct: float, price_accel: float, atr: float, regime: str,
    ) -> float:
        """Estimate P(extreme 3-sigma move) from volatility regime and acceleration."""
        p = self.tail_risk_base

        p += min(atr_pct, 1.0) * 0.05

        if atr > 0:
            accel_ratio = abs(price_accel) / atr
            p += min(accel_ratio / 5.0, 0.05)

        if regime == REGIME_LIQUIDITY_SHOCK:
            p += 0.05
        elif regime == REGIME_TREND_EXPANSION:
            p += 0.03

        return max(0.01, min(self.tail_risk_max, p))

    def _liquidation_buffer(
        self, equity: float, atr: float, leverage: int, notional: float,
    ) -> float:
        """Returns liquidation buffer in ATR units. Higher = safer."""
        if atr <= 0 or leverage <= 0 or notional <= 0:
            return 999.0
        margin = equity
        maintenance_margin = notional * 0.004
        available = margin - maintenance_margin
        if available <= 0:
            return 0.0
        return available / (atr * abs(notional / (equity if equity > 0 else 1.0)))

    # ══════════════════════════════════════════════════════════════
    # Layer 5: Multi-State EV Model
    # ══════════════════════════════════════════════════════════════

    def _compute_full_ev(
        self,
        funding_pred: float,
        p_survive: float,
        p_tail: float,
        funding_half_life: float,
        notional: float,
        quantity: float,
        atr: float,
    ) -> Dict[str, Any]:
        """5-state probability-weighted expected value model.

        States: TP, SL, Survive+Funding, Funding-Decay-Exit, Tail-Event
        """
        tp_dist = atr * self.atr_tp_multiplier if atr > 0 else 0.0
        sl_dist = atr * self.atr_sl_multiplier if atr > 0 else 0.0
        total_dist = tp_dist + sl_dist

        p_tp_rw = sl_dist / total_dist if total_dist > 0 else 0.5
        p_sl_rw = 1.0 - p_tp_rw

        p_decay = min(self.p_decay_base * (10.0 / max(funding_half_life, 1.0)), 0.30)

        budget = 1.0 - p_tail - p_decay
        budget = max(budget, 0.1)

        p_survive_net = p_survive * budget * 0.3
        p_tp = p_tp_rw * budget * 0.35
        p_sl = p_sl_rw * budget * 0.35

        tp_gain = quantity * tp_dist
        sl_loss = quantity * sl_dist
        funding_income = abs(funding_pred) * notional
        round_trip_cost = 2.0 * self.fee_rate * notional
        spread_cost = round_trip_cost * 0.5
        tail_loss = quantity * sl_dist * 2.0

        ev = (
            p_tp * tp_gain
            + p_survive_net * funding_income
            - p_sl * sl_loss
            - p_decay * spread_cost
            - p_tail * tail_loss
            - round_trip_cost
        )

        return {
            "ev": round(ev, 4),
            "p_tp": round(p_tp, 4),
            "p_sl": round(p_sl, 4),
            "p_survive_net": round(p_survive_net, 4),
            "p_decay": round(p_decay, 4),
            "p_tail": round(p_tail, 4),
            "tp_gain": round(tp_gain, 4),
            "sl_loss": round(sl_loss, 4),
            "funding_income": round(funding_income, 4),
            "round_trip_cost": round(round_trip_cost, 4),
            "tail_loss": round(tail_loss, 4),
            "profitable": ev > self.ev_threshold,
        }

    # ══════════════════════════════════════════════════════════════
    # Layer 4: Execution Engine
    # ══════════════════════════════════════════════════════════════

    def _adaptive_leverage(
        self, funding_pred: float, funding_vol: float, regime: str,
    ) -> int:
        """Carry/Vol ratio-driven leverage. Conservative in unstable regimes."""
        if funding_vol <= 0 or regime == REGIME_LIQUIDITY_SHOCK:
            return self.min_leverage

        carry_vol_ratio = abs(funding_pred) / funding_vol
        raw_lev = self.adaptive_lev_base * carry_vol_ratio

        if regime == REGIME_TREND_EXPANSION:
            raw_lev *= 0.5
        elif regime == REGIME_FUNDING_EXTREME:
            raw_lev *= 0.8

        return max(self.min_leverage, min(self.max_leverage, int(round(raw_lev))))

    def _ev_position_scale(self, ev: float, notional: float) -> float:
        """Scale position size proportional to EV strength. Returns [0.3, 1.0]."""
        if notional <= 0 or ev <= 0:
            return 0.3
        ev_pct = ev / notional
        scale = min(ev_pct / 0.001, self.ev_scale_max)
        return max(0.3, min(1.0, scale))

    def _cap_sizing_for_margin(
        self, sizing: Dict[str, float], equity: float, price: float,
    ) -> Dict[str, float]:
        qty = safe_float(sizing.get("quantity", 0.0))
        lev = max(1, int(sizing.get("leverage", 1) or 1))
        if qty <= 0 or price <= 0 or equity <= 0:
            return sizing
        denom = price * (1.0 / lev + self.fee_rate)
        if denom <= 0:
            return sizing
        max_qty = (equity * self.margin_buffer_ratio) / denom
        if max_qty <= 0:
            return sizing
        if qty > max_qty:
            qty = round(max_qty, self.quantity_precision)
            qty = max(qty, self.min_quantity)
            sizing["quantity"] = qty
            sizing["notional"] = round(qty * price, 4)
            sizing["notional_pct"] = round((qty * price) / equity, 4) if equity > 0 else 0.0
            sizing["actual_risk_usdt"] = round(qty * safe_float(sizing.get("sl_distance", 0.0)), 4)
            sizing["actual_risk_pct"] = round(sizing["actual_risk_usdt"] / equity, 6) if equity > 0 else 0.0
        return sizing

    # ── Settlement detection ──

    def _detect_settlement_passed(self, minutes_to_next: int):
        if (self._last_minutes_to_settlement <= self.entry_window_minutes
                and minutes_to_next > 400):
            self._settlements_collected += 1
        self._last_minutes_to_settlement = minutes_to_next

    # ── Risk state tracking ──

    def _update_risk_state(self, current_pos: float, price: float):
        old_exited = (
            (self._last_position != 0.0 and current_pos == 0.0)
            or (self._last_position > 0 and current_pos < 0)
            or (self._last_position < 0 and current_pos > 0)
        )

        if old_exited and self._entry_price is not None:
            if self._last_position > 0:
                pnl = (price - self._entry_price) * abs(self._last_position)
            else:
                pnl = (self._entry_price - price) * abs(self._last_position)
            if pnl < 0:
                self._consecutive_losses += 1
                self._cooldown_remaining = self.cooldown_bars
            else:
                self._consecutive_losses = 0

        if current_pos != 0 and self._last_position == 0:
            self._best_price = None
            self._settlements_collected = 0
        elif old_exited:
            self._best_price = None
            self._entry_price = None
            self._settlements_collected = 0

        self._last_position = current_pos

    # ── Trailing stop ──

    def _check_trailing_exit(self, curr_side: int, price: float, atr: float) -> bool:
        if self._entry_price is None or atr <= 0:
            return False
        if curr_side == 1:
            unrealized = price - self._entry_price
        else:
            unrealized = self._entry_price - price

        trigger = atr * self.trailing_trigger_atr
        if self._best_price is not None:
            if curr_side == 1:
                best_unrealized = self._best_price - self._entry_price
            else:
                best_unrealized = self._entry_price - self._best_price
            if best_unrealized >= trigger:
                pullback = best_unrealized - unrealized
                if pullback >= atr * self.trailing_pullback_atr:
                    return True
        return False

    # ══════════════════════════════════════════════════════════════
    # Signal generation (integrates all layers)
    # ══════════════════════════════════════════════════════════════

    def _generate_signal(
        self,
        df: pd.DataFrame,
        current_pos: float,
        funding_rate: float,
        current_time: Optional[pd.Timestamp],
        regime: str,
        funding_pred: float,
        p_survive: float,
        p_tail: float,
        ev_info: Dict[str, Any],
    ) -> float:
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)

        # ── Cooldown / pause ──
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if curr_side == 0:
                return 0.0

        if self._consecutive_losses >= self.max_consecutive_losses:
            self._loss_pause_bars += 1
            if self._loss_pause_bars >= self.loss_pause_timeout:
                self._consecutive_losses = 0
                self._loss_pause_bars = 0
            elif curr_side == 0:
                return 0.0

        if curr_side == 0 and self._reverse_cooldown_remaining > 0:
            self._reverse_cooldown_remaining -= 1
            return 0.0

        price = safe_float(df[PRICE_COL].iloc[-1])
        rsi = safe_float(df[f"rsi_{self.rsi_period}"].iloc[-1], 50.0)
        ema_trend = safe_float(df[f"ema_{self.ema_trend_period}"].iloc[-1])
        adx = safe_float(df[f"adx_{self.adx_period}"].iloc[-1], 0.0)
        atr = safe_float(df[f"atr_{self.atr_period}"].iloc[-1])
        funding_pct = safe_float(df[f"funding_pct_{self.funding_pct_period}"].iloc[-1], 0.5)
        ema_fast = safe_float(df[f"ema_{self.ema_fast_period}"].iloc[-1])
        zscore = safe_float(df[f"zscore_{self.zscore_period}"].iloc[-1], 0.0)
        er = safe_float(df[f"er_{self.er_period}"].iloc[-1], 0.0)
        atr_pct = safe_float(df[f"atr_pct_{self.atr_period}_{self.atr_pct_window}"].iloc[-1], 0.5)

        minutes_to_next = (
            self._minutes_to_next_settlement(current_time)
            if current_time is not None else 999
        )
        minutes_since_last = (
            self._minutes_since_last_settlement(current_time)
            if current_time is not None else 999
        )

        if curr_side != 0 and current_time is not None:
            self._detect_settlement_passed(minutes_to_next)

        # ══════════════════════════════════════════════════════
        # Position management (exit logic)
        # ══════════════════════════════════════════════════════
        if curr_side != 0:
            if curr_side == self._last_side:
                self._hold_bars += 1
            else:
                self._hold_bars = 1
                self._last_side = curr_side

            if self._hold_bars >= self.max_hold_bars:
                return float(-curr_side)

            # Regime kill-switch: LIQUIDITY_SHOCK → immediate exit
            if regime == REGIME_LIQUIDITY_SHOCK and self._hold_bars >= self.min_hold_bars:
                return float(-curr_side)

            # Tail risk kill-switch
            if p_tail > self.tail_risk_threshold and self._hold_bars >= self.min_hold_bars:
                return float(-curr_side)

            # Trailing stop
            if self._hold_bars >= self.min_hold_bars:
                if self._check_trailing_exit(curr_side, price, atr):
                    return float(-curr_side)

            # Post-settlement exit
            if self._settlements_collected >= 1:
                if minutes_since_last >= self.exit_after_minutes:
                    return float(-curr_side)
                if abs(funding_rate) < self.entry_threshold * 0.5:
                    return float(-curr_side)
                if not self.allow_multi_settlement:
                    return 0.0

            # Funding decay exit: predicted funding collapsed
            if self._hold_bars >= self.min_hold_bars:
                if abs(funding_pred) < self.entry_threshold * 0.3:
                    return float(-curr_side)

            return 0.0

        # ══════════════════════════════════════════════════════
        # Entry logic (carry-driven)
        # ══════════════════════════════════════════════════════

        # Gate 1: Regime
        if regime == REGIME_LIQUIDITY_SHOCK:
            return 0.0
        if regime == REGIME_TREND_EXPANSION:
            return 0.0

        # Gate 2: Time window
        if current_time is not None and minutes_to_next > self.entry_window_minutes:
            return 0.0

        # Gate 3: Funding direction (use predicted funding, not just current)
        effective_thr = self._effective_entry_threshold(minutes_to_next)
        funding_signal = funding_pred if abs(funding_pred) > abs(funding_rate) else funding_rate

        if abs(funding_signal) < effective_thr:
            return 0.0

        direction = 0
        if funding_signal > effective_thr:
            if funding_pct > self.funding_pct_threshold or funding_pct == 0.5:
                direction = -1
        elif funding_signal < -effective_thr:
            if funding_pct < (1 - self.funding_pct_threshold) or funding_pct == 0.5:
                direction = 1

        if direction == 0:
            return 0.0

        is_strong = abs(funding_signal) >= self.strong_threshold

        # Gate 4: Survival probability
        if p_survive < self.survive_min_threshold and not is_strong:
            return 0.0

        # Gate 5: Tail risk
        if p_tail > self.tail_risk_threshold:
            return 0.0

        # Gate 6: EV gate (multi-state)
        if not ev_info.get("profitable", False):
            return 0.0

        # Gate 7: Mean-reversion confirmation (unless strong or FUNDING_EXTREME regime)
        if not is_strong and regime != REGIME_FUNDING_EXTREME:
            if direction == -1:
                zscore_ok = zscore > self.zscore_entry
                ema_reversal = ema_fast > 0 and price < ema_fast
                if not (zscore_ok or ema_reversal):
                    return 0.0
            elif direction == 1:
                zscore_ok = zscore < -self.zscore_entry
                ema_reversal = ema_fast > 0 and price > ema_fast
                if not (zscore_ok or ema_reversal):
                    return 0.0

        # Gate 8: ER trend filter
        if not is_strong and er > self.er_max_for_fade and minutes_to_next > 10:
            return 0.0

        # Gate 9: ATR volatility filter
        if atr_pct < self.atr_pct_min or atr_pct > self.atr_pct_max:
            return 0.0

        # Gate 10: RSI extreme protection
        if direction == -1 and rsi < self.rsi_extreme_long:
            return 0.0
        if direction == 1 and rsi > self.rsi_extreme_short:
            return 0.0

        # Gate 11: ADX trend protection
        if not is_strong and adx > self.adx_extreme_threshold and ema_trend > 0:
            trend_bullish = price > ema_trend
            if direction == -1 and trend_bullish:
                return 0.0
            if direction == 1 and not trend_bullish:
                return 0.0

        self._hold_bars = 0
        self._last_side = direction
        self._settlements_collected = 0
        self._last_minutes_to_settlement = minutes_to_next
        return float(direction)

    # ══════════════════════════════════════════════════════════════
    # __call__ — main entry point
    # ══════════════════════════════════════════════════════════════

    async def __call__(
        self,
        df: pd.DataFrame,
        current_pos: float = 0.0,
        equity: float = 0.0,
        funding_rate: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # ── Compute factors ──
        await factor_manager("ema", df)
        await factor_manager("rsi", df)
        await factor_manager("atr", df)
        await factor_manager("adx", df)
        await factor_manager("funding_percentile", df, periods=[self.funding_pct_period])
        await factor_manager("atr_percentile", df, periods=[self.atr_pct_window])
        await factor_manager("funding_volatility", df, periods=[self.funding_vol_period])
        await factor_manager("price_acceleration", df, periods=[self.price_accel_period])
        await factor_manager("zscore_price", df, periods=[self.zscore_period])
        await factor_manager("er", df, periods=[self.er_period])

        current_time = self._get_current_time(df)
        price = safe_float(df[PRICE_COL].iloc[-1])
        atr = safe_float(df[f"atr_{self.atr_period}"].iloc[-1])
        adx = safe_float(df[f"adx_{self.adx_period}"].iloc[-1], 0.0)
        atr_pct = safe_float(df[f"atr_pct_{self.atr_period}_{self.atr_pct_window}"].iloc[-1], 0.5)
        funding_vol = safe_float(df[f"funding_vol_{self.funding_vol_period}"].iloc[-1], 0.0)
        price_accel = safe_float(df[f"price_accel_{self.price_accel_period}"].iloc[-1], 0.0)

        # Funding estimation fallback
        if self.use_estimated_funding and abs(funding_rate) <= 0.000101:
            if len(df) >= self.ema_trend_period:
                ema_val = safe_float(df[f"ema_{self.ema_trend_period}"].iloc[-1])
                if ema_val > 0:
                    estimated = (price - ema_val) / ema_val * 0.1
                    if abs(estimated) > abs(funding_rate):
                        funding_rate = estimated

        self._update_rate_history(funding_rate)
        self._update_risk_state(current_pos, price)

        # ── Layer 1: Regime ──
        regime = self._detect_regime(adx, atr, price_accel, funding_rate, funding_vol)
        self._current_regime = regime

        # ── Layer 2: Funding Alpha ──
        funding_pred = self._predict_funding()
        funding_persist = self._funding_persistence_prob()
        half_life = self._funding_half_life()
        funding_stable = self._funding_regime_stable(funding_vol)

        # ── Layer 3: Survival & Risk ──
        sl_distance = atr * self.atr_sl_multiplier
        minutes_to_next = (
            self._minutes_to_next_settlement(current_time)
            if current_time is not None else 999
        )
        p_survive = self._survival_probability(
            sl_distance, atr, minutes_to_next, adx, atr_pct,
        )
        p_tail = self._tail_risk_prob(atr_pct, price_accel, atr, regime)
        liq_buf = self._liquidation_buffer(equity, atr, self.min_leverage, equity * 0.2)

        # ── Layer 4: Adaptive leverage ──
        adaptive_lev = self._adaptive_leverage(funding_pred, funding_vol, regime)

        # ── Layer 5: Multi-state EV (preliminary sizing for EV calc) ──
        prelim_sizing = calc_position_size(
            equity=equity, price=price, atr=atr,
            risk_pct=self.risk_per_trade, atr_sl_mult=self.atr_sl_multiplier,
            min_leverage=self.min_leverage, max_leverage=adaptive_lev,
            min_notional_pct=self.min_notional_pct, max_notional_pct=self.max_notional_pct,
            min_quantity=self.min_quantity, quantity_precision=self.quantity_precision,
        )
        prelim_sizing = self._cap_sizing_for_margin(prelim_sizing, equity, price)

        ev_info = self._compute_full_ev(
            funding_pred=funding_pred,
            p_survive=p_survive,
            p_tail=p_tail,
            funding_half_life=half_life,
            notional=prelim_sizing.get("notional", 0.0),
            quantity=prelim_sizing.get("quantity", 0.0),
            atr=atr,
        )

        # ── Signal generation (L4) ──
        signal = self._generate_signal(
            df, current_pos, funding_rate, current_time,
            regime, funding_pred, p_survive, p_tail, ev_info,
        )
        sig_int = int(np.sign(signal)) if signal != 0 else 0

        # ── Final sizing: EV-scaled ──
        ev_scale = self._ev_position_scale(
            ev_info.get("ev", 0.0), prelim_sizing.get("notional", 0.0),
        )
        final_sizing = dict(prelim_sizing)
        final_sizing["leverage"] = adaptive_lev

        if sig_int != 0 and current_pos == 0:
            scaled_qty = round(
                final_sizing["quantity"] * ev_scale, self.quantity_precision,
            )
            scaled_qty = max(scaled_qty, self.min_quantity)
            final_sizing["quantity"] = scaled_qty
            final_sizing["notional"] = round(scaled_qty * price, 4)

        # ── Liquidation buffer gate ──
        if sig_int != 0 and current_pos == 0 and liq_buf < self.liq_buffer_min:
            sig_int = 0

        # ── Funding stability gate ──
        if sig_int != 0 and current_pos == 0 and not funding_stable:
            if regime != REGIME_FUNDING_EXTREME:
                sig_int = 0

        # ── Build order ──
        result = build_order(
            symbol=self.symbol, signal=sig_int, current_pos=current_pos,
            sizing=final_sizing, price=price, atr=atr,
            atr_sl_multiplier=self.atr_sl_multiplier,
            atr_tp_multiplier=self.atr_tp_multiplier,
            use_trailing_stop=False,
            flip_mode=self.flip_mode,
            fee_rate=self.fee_rate,
            quantity_precision=self.quantity_precision,
        )

        if result["signal"] == "CLOSE" and self.flip_mode == "close_then_wait":
            self._reverse_cooldown_remaining = max(0, int(self.reverse_cooldown_bars))

        if result["signal"] in ("LONG", "SHORT"):
            self._entry_price = price
            self._best_price = price

        # ── Trailing stop info ──
        effective_pos = current_pos
        if result["signal"] == "LONG":
            effective_pos = final_sizing["quantity"]
        elif result["signal"] == "SHORT":
            effective_pos = -final_sizing["quantity"]

        trailing_info, self._best_price = compute_trailing_stop(
            effective_pos, price, atr, self.atr_sl_multiplier,
            self._entry_price, self._best_price,
        )

        # ── Update prev ATR for regime spike detection ──
        self._prev_atr = atr

        # ── Settlement info ──
        settlement_info: Dict[str, Any] = {}
        if current_time is not None:
            mins_to = self._minutes_to_next_settlement(current_time)
            mins_since = self._minutes_since_last_settlement(current_time)
            settlement_info = {
                "current_time_utc": str(current_time),
                "minutes_to_next_settlement": mins_to,
                "minutes_since_last_settlement": mins_since,
                "in_entry_window": mins_to <= self.entry_window_minutes,
                "settlements_collected": self._settlements_collected,
            }

        factors_json = build_factors_json(df, [
            f"ema_{self.ema_fast_period}", f"ema_{self.ema_trend_period}",
            f"rsi_{self.rsi_period}",
            f"atr_{self.atr_period}",
            f"adx_{self.adx_period}",
            f"funding_pct_{self.funding_pct_period}",
            f"atr_pct_{self.atr_period}_{self.atr_pct_window}",
            f"funding_vol_{self.funding_vol_period}",
            f"price_accel_{self.price_accel_period}",
            f"zscore_{self.zscore_period}",
            f"er_{self.er_period}",
        ], extra={
            "regime": regime,
            "funding_rate": round(funding_rate, 8),
            "funding_pred": round(funding_pred, 8),
            "funding_stable": funding_stable,
            "p_survive": round(p_survive, 3),
            "p_tail": round(p_tail, 4),
            "adaptive_leverage": adaptive_lev,
            "hold_bars": self._hold_bars,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_remaining": self._cooldown_remaining,
        })

        meta = {
            **result["meta"],
            "regime": regime,
            "funding_rate": funding_rate,
            "funding_pred": round(funding_pred, 8),
            "funding_persist_prob": round(funding_persist, 3),
            "funding_half_life": round(half_life, 1),
            "funding_vol": round(funding_vol, 8),
            "funding_stable": funding_stable,
            "funding_rate_avg": round(self._rate_avg(), 6),
            "p_survive": round(p_survive, 3),
            "p_tail": round(p_tail, 4),
            "liq_buffer_atr": round(liq_buf, 2),
            "ev": ev_info,
            "ev_scale": round(ev_scale, 3),
            "adaptive_leverage": adaptive_lev,
            "settlement": settlement_info,
            "factors_json": factors_json,
        }

        return {
            "signal": result["signal"],
            "order": result["order"],
            "leverage": final_sizing["leverage"],
            "sizing": final_sizing,
            "atr": atr,
            "meta": meta,
            "factors_json": factors_json,
            "risk": {
                "cooldown_remaining": self._cooldown_remaining,
                "consecutive_losses": self._consecutive_losses,
                "loss_pause_bars": self._loss_pause_bars,
                "hold_bars": self._hold_bars,
                "reverse_cooldown_remaining": self._reverse_cooldown_remaining,
            },
            "trailing_stop": trailing_info,
        }
