"""
Adaptive Trend Fusion Strategy — Industrial-Grade Architecture v2
Binance USDT-M Perpetual Futures (BTCUSDT)

Pipeline:
  FeatureEngine → RegimeClassifier → AlphaEngine → RiskFilter
  → ExitController → PositionSizer → OrderRouter

Module Responsibilities
─────────────────────────────────────────────────────────────
FeatureEngine    : Feature computation   (pure data transformation, no trading logic)
RegimeClassifier : Market regime detection (trend_up / trend_down / ranging)
                   Regime is now a first-class gate across the whole pipeline.
AlphaEngine      : Alpha signal generation (squeeze-breakout + multi-factor trend scoring)
                   ranging → trend entries suppressed (use_regime_gate=True)
                   trend   → pyramiding & scoring bonus
RiskFilter       : Pre-trade risk gates + directional validity checks
                   Includes estimated slippage cost in fee-edge filter.
ExitController   : In-position exit management (stops / trailing / timeout)
                   min_position_bars split into min_hold_before_trailing /
                   min_hold_before_timeout for finer control.
                   trend regime → wider stops / longer hold.
PositionSizer    : Capital allocation and position sizing
                   Applies a regime_scale (ranging < 1) to raw sizing.
OrderRouter      : Order construction and routing to exchange
                   Maker/taker adaptive: strong signals use MARKET,
                   weak signals emit LIMIT (post-only intent flag).
                   entry_price NOT written here — left for fill callback.

info dict lifecycle
─────────────────────────────────────────────────────────────
FeatureEngine    writes  info["features"]
RegimeClassifier writes  info["regime"], info["macro_dir"]
AlphaEngine      writes  info["alpha"] = {direction, score, signal_type}
RiskFilter       writes  info["trade_signal"] (±1 or 0), info["risk_blocked"]
                         info["decision_trace"] entries
ExitController   writes  info["exit_signal"] (±1 or None)
PositionSizer    writes  info["sizing"]
OrderRouter      writes  info["order_result"], info["trailing_stop"]

Bar-close execution model
─────────────────────────────────────────────────────────────
This strategy is a BAR-CLOSE system:
  - All features (including this bar's high/low) are computed from the
    CLOSED bar before any order is placed.
  - The caller must ensure orders are submitted AFTER the bar closes.
  - TradeState.entry_price is intentionally NOT set by OrderRouter;
    it must be updated by the caller from the actual fill price.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import Field

from src.strategy.types import Strategy
from src.factor import factor_manager
from src.strategy.futures._utils import (
    PRICE_COL, VOL_COL, DEFAULT_TAKER_FEE,
    compute_fee_info, build_factors_json,
)


# ===========================================================================
# _SizingAlgorithms — 仓位计算算法集（内部工具，全为静态方法）
# ===========================================================================

class _SizingAlgorithms:
    """仓位规模计算纯函数库，供 PositionSizer 调用。"""

    @staticmethod
    def fixed(quantity: float, leverage: int, price: float) -> Dict[str, Any]:
        return {"quantity": quantity, "leverage": leverage,
                "notional": quantity * price, "risk_usdt": 0.0, "method": "fixed"}

    @staticmethod
    def notional_percent(equity, notional_pct, price, leverage=1, max_leverage=20,
                         min_quantity=0.0, quantity_precision=3) -> Dict[str, Any]:
        if equity <= 0 or price <= 0:
            return {"quantity": 0.0, "leverage": max(1, int(leverage or 1)),
                    "notional": 0.0, "risk_usdt": 0.0, "method": "notional_pct"}
        pct      = max(0.0, float(notional_pct or 0.0))
        notional = equity * pct
        quantity = notional / price
        if min_quantity and min_quantity > 0:
            quantity = max(quantity, float(min_quantity))
        quantity = round(quantity, quantity_precision)
        lev      = max(1, min(int(leverage or 1), int(max_leverage)))
        return {"quantity": float(quantity), "leverage": lev,
                "notional": float(quantity * price), "risk_usdt": 0.0,
                "notional_pct": float(pct), "method": "notional_pct"}

    @staticmethod
    def margin_percent(equity, margin_pct, price, leverage=1, max_leverage=20,
                       min_quantity=0.0, quantity_precision=3) -> Dict[str, Any]:
        if equity <= 0 or price <= 0:
            return {"quantity": 0.0, "leverage": max(1, int(leverage or 1)),
                    "notional": 0.0, "risk_usdt": 0.0, "method": "margin_pct"}
        pct        = max(0.0, float(margin_pct or 0.0))
        lev        = max(1, min(int(leverage or 1), int(max_leverage)))
        notional   = equity * pct * float(lev)
        quantity   = notional / price
        if min_quantity and min_quantity > 0:
            quantity = max(quantity, float(min_quantity))
        quantity = round(quantity, quantity_precision)
        notional = quantity * price
        return {"quantity": float(quantity), "leverage": lev,
                "notional": float(notional),
                "margin_usdt": float(notional / lev if lev > 0 else 0.0),
                "margin_pct": float(pct), "risk_usdt": 0.0, "method": "margin_pct"}

    @staticmethod
    def risk_percent(equity, risk_pct, price, atr, atr_sl_multiplier=2.0,
                     max_leverage=20, min_quantity=0.02, quantity_precision=3) -> Dict[str, Any]:
        if atr <= 0 or price <= 0 or equity <= 0:
            return {"quantity": min_quantity, "leverage": 1,
                    "notional": min_quantity * price, "risk_usdt": 0.0, "method": "risk_percent"}
        risk_usdt   = equity * risk_pct
        sl_distance = atr * atr_sl_multiplier
        quantity    = max(risk_usdt / sl_distance, min_quantity)
        quantity    = round(quantity, quantity_precision)
        notional    = quantity * price
        leverage    = max(1, min(int(np.ceil(notional / equity)), max_leverage))
        return {"quantity": quantity, "leverage": leverage, "notional": notional,
                "risk_usdt": risk_usdt, "sl_distance": sl_distance, "method": "risk_percent"}

    @staticmethod
    def volatility_target(equity, target_vol, price, atr, max_leverage=20,
                          min_quantity=0.02, quantity_precision=3) -> Dict[str, Any]:
        if atr <= 0 or price <= 0 or equity <= 0:
            return {"quantity": min_quantity, "leverage": 1,
                    "notional": min_quantity * price, "risk_usdt": 0.0, "method": "vol_target"}
        daily_vol_pct   = max(atr / price, 0.01)
        target_notional = equity * target_vol / daily_vol_pct
        quantity        = max(target_notional / price, min_quantity)
        quantity        = round(quantity, quantity_precision)
        notional        = quantity * price
        leverage        = max(1, min(int(np.ceil(notional / equity)), max_leverage))
        return {"quantity": quantity, "leverage": leverage, "notional": notional,
                "daily_vol_pct": daily_vol_pct, "method": "vol_target"}

    @staticmethod
    def kelly(equity, win_rate, avg_win, avg_loss, price, fraction=0.5,
              max_leverage=20, min_quantity=0.02, quantity_precision=3) -> Dict[str, Any]:
        if avg_loss <= 0 or equity <= 0 or price <= 0:
            return {"quantity": min_quantity, "leverage": 1,
                    "notional": min_quantity * price, "risk_usdt": 0.0,
                    "kelly_f": 0.0, "method": "kelly"}
        payoff_ratio = avg_win / avg_loss
        kelly_f  = max(0.0, win_rate - (1.0 - win_rate) / payoff_ratio) * fraction
        risk_usdt = equity * kelly_f
        quantity  = max(risk_usdt / price, min_quantity)
        quantity  = round(quantity, quantity_precision)
        notional  = quantity * price
        leverage  = max(1, min(int(np.ceil(notional / equity)), max_leverage))
        return {"quantity": quantity, "leverage": leverage, "notional": notional,
                "risk_usdt": risk_usdt, "kelly_f": kelly_f, "method": "kelly"}


# ===========================================================================
# TradeState — 跨模块共享的持久化交易状态
# ===========================================================================

@dataclass
class TradeState:
    """在整个 pipeline 流程中通过 info['trade_state'] 共享。

    Attributes:
        entry_price    : 最近一次开仓实际成交价（由外部 fill 回报写入，NOT by OrderRouter）
        entry_leverage : 开仓时使用的杠杆倍数（由外部 fill 回报写入）
        last_position  : 上一 bar 的持仓量（由 RiskFilter 更新，用于平仓检测）
        hold_bars      : 本次持仓已持续的 bar 数（由 AlphaEngine 更新）
        last_side      : 上一 bar 的持仓方向（由 AlphaEngine 更新）
        pyramid_count  : 本轮持仓已加仓次数（由 AlphaEngine 更新）
        last_add_bar   : 上次加仓时的 hold_bars 值（由 AlphaEngine 更新）
        pending_fill   : True = 已下单未收到 fill，阻止重复下单（由 caller 管理）

    NOTE (bar-close system):
        entry_price MUST be set by the caller's fill callback, not by
        OrderRouter.  OrderRouter only constructs the *intent* order.
        Until a fill is received, entry_price remains None / previous value.
    """
    entry_price:    Optional[float] = None
    entry_leverage: int             = 1
    last_position:  float           = 0.0
    hold_bars:      int             = 0
    last_side:      int             = 0
    pyramid_count:  int             = 0
    last_add_bar:   int             = 0
    pending_fill:   bool            = False

    def update_from_fill(self, fill_price: float, leverage: int) -> None:
        """成交回报到达后由 caller 调用，更新 entry_price / leverage。"""
        self.entry_price    = float(fill_price)
        self.entry_leverage = max(1, int(leverage))
        self.pending_fill   = False


# ===========================================================================
# FeatureEngine — 特征工程 / 指标计算层
# ===========================================================================

class FeatureEngine:
    """将原始 OHLCV 数据转换为策略所需的全部技术特征。

    职责：纯数据转换，无任何交易逻辑。
    输出：info["features"] — 所有指标标量值的字典，key 为英文缩写。

    额外计算：
    - 过度延伸自适应分位数阈值（ext_vwap_thresh / ext_ema_thresh）
    - 宏观趋势方向 macro_dir（+1 / -1 / 0）
    """

    def __init__(self, cfg: Any) -> None:
        self._c = cfg
        self._last_swing_high: Optional[float] = None
        self._last_swing_low:  Optional[float] = None

    def __call__(self, df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, Dict]:
        c      = self._c
        closes  = df[PRICE_COL].astype(float)
        highs   = df["high"].astype(float) if "high" in df.columns else closes
        lows    = df["low"].astype(float)  if "low"  in df.columns else closes
        volumes = df[VOL_COL].astype(float)

        def _last(col: str) -> float:
            if col not in df.columns:
                return float("nan")
            val = df[col].iloc[-1]
            if val is None:
                return float("nan")
            try:
                return float(val)
            except (TypeError, ValueError):
                return float("nan")

        rsi_col  = f"rsi_{c.rsi_period}"
        rsi      = float(df[rsi_col].iloc[-1])
        rsi_prev = float(df[rsi_col].iloc[-2]) if len(df) > 1 else rsi

        bb_width_col  = f"bb_width_{c.bb_width_period}"
        bb_width_prev = float("nan")
        if bb_width_col in df.columns and len(df) >= 2 and not np.isnan(df[bb_width_col].iloc[-2]):
            bb_width_prev = float(df[bb_width_col].iloc[-2])

        swing_high, swing_low = self._find_swings(highs, lows, c.swing_lookback)
        self._last_swing_high = swing_high
        self._last_swing_low  = swing_low

        vol_surge = self._detect_vol_surge(
            volumes, float(volumes.iloc[-1]), c.vol_surge_lookback, c.vol_surge_multiplier)
        vol_mean  = (float(volumes.iloc[-c.vol_surge_lookback:].mean())
                     if len(volumes) >= c.vol_surge_lookback else float(volumes.mean()))
        vol_ratio = float(volumes.iloc[-1] / vol_mean) if vol_mean > 0 else float("nan")

        # 自适应过度延伸阈值（历史分位数）
        ext_vwap_thresh = float("nan")
        ext_ema_thresh  = float("nan")
        if c.use_extension_filter and c.extension_adaptive:
            atr_col   = f"atr_{c.atr_period}"
            vwap_col  = f"vwap_{c.vwap_period}"
            ema_s_col = f"ema_{c.ema_slow_period}"
            lb = min(c.extension_lookback, len(closes))
            if lb >= 30 and atr_col in df.columns:
                atr_s     = df[atr_col].iloc[-lb:].astype(float)
                valid_atr = atr_s.replace(0, np.nan)
                if vwap_col in df.columns:
                    vwap_s      = df[vwap_col].iloc[-lb:].astype(float)
                    ext_vwap_s  = ((closes.iloc[-lb:] - vwap_s) / valid_atr).abs().dropna()
                    if len(ext_vwap_s) >= 20:
                        ext_vwap_thresh = float(
                            np.nanpercentile(ext_vwap_s, c.extension_percentile * 100))
                if ema_s_col in df.columns:
                    ema_s_s    = df[ema_s_col].iloc[-lb:].astype(float)
                    ext_ema_s  = ((closes.iloc[-lb:] - ema_s_s) / valid_atr).abs().dropna()
                    if len(ext_ema_s) >= 20:
                        ext_ema_thresh = float(
                            np.nanpercentile(ext_ema_s, c.extension_percentile * 100))

        # 宏观方向（ROC）
        macro_dir = 0
        roc_macro = _last(f"roc_{c.macro_slope_period}")
        if c.use_macro_slope and not np.isnan(roc_macro):
            sb = float(c.macro_slope_buf)
            if   roc_macro > 1.0 + sb: macro_dir =  1
            elif roc_macro < 1.0 - sb: macro_dir = -1

        # funding_rate（仅用于日志记录，不做门控）
        funding_rate = _last("funding_rate")

        info["features"] = {
            "closes":  closes, "highs": highs, "lows": lows, "volumes": volumes,
            "price":   float(closes.iloc[-1]),
            "volume":  float(volumes.iloc[-1]),
            "ema_fast":  _last(f"ema_{c.ema_fast_period}"),
            "ema_slow":  _last(f"ema_{c.ema_slow_period}"),
            "ema_macro": _last(f"ema_{c.ema_macro_period}"),
            "rsi": rsi, "rsi_prev": rsi_prev,
            "adx":      float(df[f"adx_{c.adx_period}"].iloc[-1]),
            "plus_di":  _last(f"plus_di_{c.adx_period}"),
            "minus_di": _last(f"minus_di_{c.adx_period}"),
            "bb_width":      _last(bb_width_col),
            "bb_width_prev": bb_width_prev,
            "vwap":  _last(f"vwap_{c.vwap_period}"),
            "er":    _last(f"er_{c.er_period}"),
            "atr":   _last(f"atr_{c.atr_period}"),
            "roc":   roc_macro,
            "macro_dir": int(macro_dir),
            "swing_high": swing_high,
            "swing_low":  swing_low,
            "vol_surge":  vol_surge,
            "vol_ratio":  vol_ratio,
            "ext_vwap_thresh": ext_vwap_thresh,
            "ext_ema_thresh":  ext_ema_thresh,
            "funding_rate":   funding_rate,   # 仅日志，不做门控
        }
        return df, info

    @staticmethod
    def _find_swings(high: pd.Series, low: pd.Series,
                     lookback: int) -> Tuple[Optional[float], Optional[float]]:
        if len(high) < lookback * 2:
            return None, None
        return high.iloc[-(lookback + 1):-1].max(), low.iloc[-(lookback + 1):-1].min()

    @staticmethod
    def _detect_vol_surge(volumes: pd.Series, current_vol: float,
                           lookback: int, multiplier: float) -> bool:
        if len(volumes) < lookback:
            return False
        avg = volumes.iloc[-lookback:].mean()
        return avg > 0 and current_vol >= avg * multiplier


# ===========================================================================
# RegimeClassifier — 市场状态分类器
# ===========================================================================

class RegimeClassifier:
    """根据 EMA 结构 + 宏观斜率将当前市场状态分类。

    分类结果：
      "trend_up"   : 快线在慢线上方 且 宏观斜率向上（或中性）
      "trend_down" : 对称
      "ranging"    : 其余情况（震荡/无明确趋势）

    输出：
      info["regime"]    : "trend_up" | "trend_down" | "ranging"
      info["macro_dir"] : +1 / -1 / 0
    """

    def __init__(self, cfg: Any) -> None:
        self._c = cfg

    def __call__(self, df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, Dict]:
        c   = self._c
        ft  = info["features"]
        buf = float(c.ema_trend_buffer)

        price    = ft["price"]
        ema_fast = ft["ema_fast"]
        ema_slow = ft["ema_slow"]
        macro_dir = int(ft.get("macro_dir", 0) or 0)

        trend_up = (
            not np.isnan(ema_fast) and not np.isnan(ema_slow)
            and price    > ema_slow * (1 + buf)
            and ema_fast > ema_slow * 1.0005
            and macro_dir >= 0
        )
        trend_down = (
            not np.isnan(ema_fast) and not np.isnan(ema_slow)
            and price    < ema_slow * (1 - buf)
            and ema_fast < ema_slow * 0.9995
            and macro_dir <= 0
        )

        if   macro_dir ==  1 and trend_up:   regime = "trend_up"
        elif macro_dir == -1 and trend_down: regime = "trend_down"
        elif trend_up:                       regime = "trend_up"
        elif trend_down:                     regime = "trend_down"
        else:                                regime = "ranging"

        info["regime"]    = regime
        info["macro_dir"] = macro_dir
        return df, info


# ===========================================================================
# AlphaEngine — Alpha 信号生成引擎
# ===========================================================================

class AlphaEngine:
    """从市场特征中提取方向性 Alpha 信号。

    整合两类信号策略：
      1. Volatility Squeeze Breakout — 布林带收缩后的扩张突破
      2. Multi-Factor Trend Scoring  — EMA 排列、RSI 回踩、成交量、ADX 综合评分

    内部状态：
      _squeeze_* : 布林带 squeeze 计数与状态
      _pending_* : 突破回踩确认状态机（仅当价格确认回踩后才正式入场）
      _ema_fast_prev / _ema_slow_prev : 用于金叉/死叉检测

    输出 info["alpha"]:
      direction   : +1（做多）/ -1（做空）/ 0（无信号）
      score       : 多因子综合评分
      signal_type : "breakout" | "trend" | "pending" | "none"
    """

    def __init__(self, cfg: Any, risk_filter: "RiskFilter") -> None:
        self._c  = cfg
        self._rf = risk_filter   # 用于方向有效性检验（is_direction_blocked）

        # Squeeze 状态
        self._squeeze_count:       int  = 0
        self._squeeze_armed:       bool = False
        self._squeeze_window_left: int  = 0

        # Pending 突破状态机
        self._pending_dir:       int             = 0
        self._pending_level:     Optional[float] = None
        self._pending_ttl:       int             = 0
        self._pending_retested:  bool            = False
        self._pending_is_squeeze: bool           = False

        # EMA 金叉/死叉记忆
        self._ema_fast_prev: Optional[float] = None
        self._ema_slow_prev: Optional[float] = None

    # ------------------------------------------------------------------
    # Pipeline 入口
    # ------------------------------------------------------------------

    def __call__(self, df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, Dict]:
        ft        = info["features"]
        curr_side = int(info.get("curr_side", 0))
        ts: TradeState = info["trade_state"]

        # 初始化决策链追踪列表
        if "decision_trace" not in info:
            info["decision_trace"] = []

        alpha = self._generate(ft, curr_side, ts)
        info["alpha"] = alpha

        # 记录 alpha 决策
        info["decision_trace"].append({
            "module":      "AlphaEngine",
            "direction":   alpha.get("direction", 0),
            "score":       alpha.get("score", 0.0),
            "signal_type": alpha.get("signal_type", "none"),
            "blocked_by":  alpha.get("blocked_by"),
            "regime":      ft.get("regime", "unknown"),
            "pending_dir": self._pending_dir,
            "pending_ttl": self._pending_ttl,
        })

        # 更新 TradeState 中的持仓 bar 计数
        if curr_side == 0:
            ts.hold_bars = ts.last_side = ts.pyramid_count = ts.last_add_bar = 0
        elif curr_side == ts.last_side:
            ts.hold_bars += 1
        else:
            ts.hold_bars = 1
            ts.last_side = curr_side

        return df, info

    # ------------------------------------------------------------------
    # 核心信号生成
    # ------------------------------------------------------------------

    def _generate(self, ft: Dict, curr_side: int,
                  ts: TradeState) -> Dict[str, Any]:
        c      = self._c
        regime = str(ft.get("regime", "ranging"))

        # ── Regime gate（只在无持仓时门控新开仓）─────────────────────────
        # ranging 时趋势单完全禁止，breakout 信号可保留（由 breakout_ok_in_ranging 控制）
        if curr_side == 0 and bool(getattr(c, "use_regime_gate", True)):
            regime_blocks_trend    = (regime == "ranging"
                                      and not bool(getattr(c, "allow_trend_in_ranging", False)))
            regime_blocks_breakout = (regime == "ranging"
                                      and not bool(getattr(c, "breakout_ok_in_ranging", True)))
        else:
            regime_blocks_trend    = False
            regime_blocks_breakout = False

        # 1. 优先处理 pending 突破
        if not regime_blocks_breakout:
            pending_result = self._process_pending(ft, curr_side)
            if pending_result is not None:
                return pending_result

        # 2. 检测新的 squeeze 突破
        if not regime_blocks_breakout:
            squeeze_result = self._detect_squeeze(ft, curr_side)
            if squeeze_result is not None:
                return squeeze_result

        # 3. 多因子趋势评分入场（ranging 时禁止）
        if regime_blocks_trend and curr_side == 0:
            return {"direction": 0, "score": 0.0, "signal_type": "none",
                    "blocked_by": "regime_ranging"}

        return self._score_trend_entry(ft, curr_side, ts)

    # ------------------------------------------------------------------
    # 1. Pending 突破回踩确认（对应原 _process_pending_breakout）
    # ------------------------------------------------------------------

    def _process_pending(self, ft: Dict, curr_side: int) -> Optional[Dict[str, Any]]:
        c = self._c
        if not (curr_side == 0 and self._pending_dir != 0
                and self._pending_ttl > 0 and self._pending_level is not None):
            return None

        lvl   = float(self._pending_level)
        d     = int(self._pending_dir)
        price = ft["price"]
        self._pending_ttl -= 1

        # 价格突破取消线 → 取消
        if ((d ==  1 and price < lvl * (1 - c.breakout_cancel_buf_pct))
                or (d == -1 and price > lvl * (1 + c.breakout_cancel_buf_pct))):
            self._clear_pending()
            return {"direction": 0, "score": 0.0, "signal_type": "none"}

        # 回踩检测
        if d == 1  and float(ft["lows"].iloc[-1])  <= lvl * (1 + c.breakout_retest_buf_pct):
            self._pending_retested = True
        if d == -1 and float(ft["highs"].iloc[-1]) >= lvl * (1 - c.breakout_retest_buf_pct):
            self._pending_retested = True

        # 回踩后再度确认 → 发出信号
        if self._pending_retested:
            confirmed = (
                (d ==  1 and price > lvl * (1 + c.breakout_confirm_buf_pct))
                or (d == -1 and price < lvl * (1 - c.breakout_confirm_buf_pct))
            )
            if confirmed:
                if c.long_only and d == -1:
                    self._clear_pending()
                    return {"direction": 0, "score": 0.0, "signal_type": "none"}
                if self._rf.is_direction_blocked(d, ft, is_squeeze=self._pending_is_squeeze):
                    self._clear_pending()
                    return {"direction": 0, "score": 0.0, "signal_type": "none"}
                macro_dir = int(ft.get("macro_dir", 0) or 0)
                er_val    = ft.get("er", float("nan"))

                # Breakout 硬过滤：当宏观方向明确且与 breakout 方向相反时，直接 veto
                # 仅在 ER 达到阈值时启用（避免在低效率/噪声区间过度依赖 macro_dir）。
                if (
                    bool(getattr(c, "use_macro_breakout_veto", True))
                    and macro_dir in (1, -1)
                    and macro_dir == -d
                    and (not np.isnan(er_val))
                    and float(er_val) >= float(getattr(c, "macro_breakout_veto_er_min", c.macro_bias_er_min))
                    and (bool(getattr(c, "macro_breakout_veto_apply_to_squeeze", True)) or (not self._pending_is_squeeze))
                ):
                    self._clear_pending()
                    return {"direction": 0, "score": 0.0, "signal_type": "none"}

                self._clear_pending()
                base = 2.0
                if macro_dir == d:
                    base += float(c.macro_align_bonus)
                elif macro_dir == -d:
                    base -= float(c.macro_contra_penalty)
                return {"direction": d, "score": float(base), "signal_type": "breakout"}

        if self._pending_ttl <= 0:
            self._clear_pending()
        return {"direction": 0, "score": 0.0, "signal_type": "pending"}

    # ------------------------------------------------------------------
    # 2. Squeeze 突破检测（对应原 _detect_squeeze_breakout）
    # ------------------------------------------------------------------

    def _detect_squeeze(self, ft: Dict, curr_side: int) -> Optional[Dict[str, Any]]:
        c    = self._c
        bb_w = ft["bb_width"]
        if not (c.use_squeeze_breakout and curr_side == 0 and not np.isnan(bb_w)):
            return None

        # squeeze 计数
        if bb_w < c.bb_width_min_entry:
            self._squeeze_count += 1
            if self._squeeze_count >= c.squeeze_min_bars:
                self._squeeze_armed        = True
                self._squeeze_window_left  = c.squeeze_breakout_window
        else:
            if self._squeeze_armed and self._squeeze_window_left > 0:
                self._squeeze_window_left -= 1
            else:
                self._squeeze_armed       = False
                self._squeeze_count       = 0
                self._squeeze_window_left = 0

        bb_prev   = ft["bb_width_prev"]
        expanding = (
            not np.isnan(bb_prev) and bb_prev > 0
            and (bb_w - bb_prev) / bb_prev >= c.bb_width_expansion_pct
        )

        sh, sl = ft["swing_high"], ft["swing_low"]
        if not (self._squeeze_armed and expanding and ft["vol_surge"]
                and sh is not None and sl is not None):
            return None

        price      = ft["price"]
        prev_close = float(ft["closes"].iloc[-2])

        if   price > sh and prev_close <= sh: d =  1
        elif price < sl and prev_close >= sl: d = -1
        else:                                 return None

        if self._rf.is_direction_blocked(d, ft, is_squeeze=True):
            return None

        # 进入 pending 确认
        self._squeeze_armed       = False
        self._squeeze_count       = 0
        self._squeeze_window_left = 0
        self._set_pending(d, float(sh if d == 1 else sl),
                          c.breakout_confirm_window, is_squeeze=True)
        return {"direction": 0, "score": 0.0, "signal_type": "pending"}

    # ------------------------------------------------------------------
    # 3. 多因子趋势评分（对应原 _evaluate_trend_entry）
    # ------------------------------------------------------------------

    def _score_trend_entry(self, ft: Dict, curr_side: int,
                           ts: TradeState) -> Dict[str, Any]:
        c          = self._c
        price      = ft["price"]
        buf        = float(c.ema_trend_buffer)
        ema_fast   = ft["ema_fast"]
        ema_slow   = ft["ema_slow"]
        _no_signal = {"direction": 0, "score": 0.0, "signal_type": "none"}

        if np.isnan(ema_fast) or np.isnan(ema_slow):
            self._ema_fast_prev = ema_fast
            self._ema_slow_prev = ema_slow
            return _no_signal

        trend_long  = price > ema_slow * (1 + buf) and ema_fast > ema_slow * 1.0005
        trend_short = price < ema_slow * (1 - buf) and ema_fast < ema_slow * 0.9995

        # 宏观方向锚定：允许在 EMA 未完全排列前捕捉大趋势反转
        macro_dir = int(ft.get("macro_dir", 0) or 0)
        er_val    = ft.get("er", float("nan"))
        use_macro = (
            curr_side == 0 and bool(c.use_macro_bias_entry)
            and macro_dir in (1, -1)
            and not np.isnan(er_val) and float(er_val) >= float(c.macro_bias_er_min)
        )

        if not use_macro:
            if not trend_long and not trend_short:
                self._ema_fast_prev = ema_fast
                self._ema_slow_prev = ema_slow
                return _no_signal
            direction = 1 if trend_long else -1
        else:
            direction = macro_dir

        if c.long_only and direction == -1:
            self._ema_fast_prev = ema_fast
            self._ema_slow_prev = ema_slow
            return _no_signal

        # 反追价（避免 RSI 低位追空 / 高位追多）
        if curr_side == 0:
            rsi_now = float(ft.get("rsi", float("nan")))
            if direction == 1 and not np.isnan(rsi_now) and rsi_now >= float(c.rsi_anti_chase_long_max):
                return {"direction": 0, "score": 0.0, "signal_type": "none"}
            if direction == -1 and not np.isnan(rsi_now) and rsi_now <= float(c.rsi_anti_chase_short_min):
                return {"direction": 0, "score": 0.0, "signal_type": "none"}

        # 工业级 timing gate：趋势方向成立后，仅在“回抽/反弹 + RSI 反转确认”时新开仓
        # - 解决 EMA/ADX 天然滞后导致的“低位追空/高位追多”
        # - 仅影响 curr_side==0 的新开仓，不影响加仓/平仓
        if curr_side == 0 and bool(c.use_entry_timing_gate):
            if not self._timing_gate_ok(direction, ft):
                return {"direction": 0, "score": 0.0, "signal_type": "none"}

        # 趋势延续型摆动突破 → 进入 pending
        if (curr_side == 0 and not self._has_pending()
                and ft["swing_high"] is not None and ft["swing_low"] is not None):
            sh, sl = float(ft["swing_high"]), float(ft["swing_low"])
            prev_c = float(ft["closes"].iloc[-2])
            if ft.get("vol_surge") or (not np.isnan(ft["adx"]) and ft["adx"] >= c.adx_threshold):
                atr = ft["atr"]
                if price > sh and prev_c <= sh:
                    self._set_pending(1, sh, c.breakout_confirm_window)
                    self._ema_fast_prev = ema_fast
                    self._ema_slow_prev = ema_slow
                    return {"direction": 0, "score": 0.0, "signal_type": "pending"}
                if price < sl and prev_c >= sl:
                    self._set_pending(-1, sl, c.breakout_confirm_window)
                    self._ema_fast_prev = ema_fast
                    self._ema_slow_prev = ema_slow
                    return {"direction": 0, "score": 0.0, "signal_type": "pending"}

        # ── 多因子评分 ──────────────────────────────────────────
        bull_score = bear_score = 0.0

        # 基础 EMA 排列分
        if direction == 1:
            if price > ema_slow * (1 + buf):     bull_score += c.trend_align_score
            if ema_fast > ema_slow * 1.0005:     bull_score += c.trend_align_score
        else:
            if price < ema_slow * (1 - buf):     bear_score += c.trend_align_score
            if ema_fast < ema_slow * 0.9995:     bear_score += c.trend_align_score

        # 趋势环境加分
        if self._is_trending(ft):
            if direction == 1: bull_score += c.env_align_score
            else:              bear_score += c.env_align_score

        # ROC 宏观方向：软偏置（对齐加分、逆向扣分；不再硬门控）
        if macro_dir in (1, -1):
            if macro_dir == direction:
                if direction == 1: bull_score += float(c.macro_align_bonus)
                else:              bear_score += float(c.macro_align_bonus)
            elif macro_dir == -direction:
                if direction == 1: bull_score -= float(c.macro_contra_penalty)
                else:              bear_score -= float(c.macro_contra_penalty)

        # Trigger A: EMA 金叉/死叉
        event_pts = 0.0
        if self._ema_fast_prev is not None and self._ema_slow_prev is not None:
            prev_above = float(self._ema_fast_prev) > float(self._ema_slow_prev)
            curr_above = float(ema_fast) > float(ema_slow)
            if not prev_above and curr_above:  bull_score += 1.0; event_pts += 1.0
            elif prev_above and not curr_above: bear_score += 1.0; event_pts += 1.0

        self._ema_fast_prev = ema_fast
        self._ema_slow_prev = ema_slow

        # Trigger B: 价格突破摆动高低点 + 成交量
        atr = ft["atr"]
        sh, sl = ft["swing_high"], ft["swing_low"]
        if sh is not None and sl is not None:
            sh, sl = float(sh), float(sl)
            if not np.isnan(atr) and atr > 0:
                if price > sh and (price - sh) <= c.breakout_chase_atr * atr:
                    bull_score += 1.0; event_pts += 1.0
                if price < sl and (sl - price) <= c.breakout_chase_atr * atr:
                    bear_score += 1.0; event_pts += 1.0
            else:
                if price > sh: bull_score += 1.0; event_pts += 1.0
                if price < sl: bear_score += 1.0; event_pts += 1.0
            if ft.get("vol_surge"):
                if direction == 1: bull_score += 0.5
                else:              bear_score += 0.5

        # Trigger C: RSI 从极值回升
        rsi, rsi_prev = ft["rsi"], ft["rsi_prev"]
        if direction ==  1 and rsi_prev <= c.rsi_pullback_long_max  and rsi > rsi_prev:
            bull_score += 1.0; event_pts += 1.0
            if rsi_prev < c.rsi_oversold: bull_score += c.rsi_extreme_bonus
        if direction == -1 and rsi_prev >= c.rsi_pullback_short_min and rsi < rsi_prev:
            bear_score += 1.0; event_pts += 1.0
            if rsi_prev > c.rsi_overbought: bear_score += c.rsi_extreme_bonus

        # Trigger D: 价格回踩 EMA 快线
        if ema_fast > 0:
            p2e = (price - ema_fast) / ema_fast
            if (direction ==  1 and c.pullback_ema_lo <= p2e <= c.pullback_ema_hi
                    and rsi < c.pullback_rsi_long_max):
                bull_score += 1.0; event_pts += 1.0
            if (direction == -1 and -c.pullback_ema_hi <= p2e <= -c.pullback_ema_lo
                    and rsi > c.pullback_rsi_short_min):
                bear_score += 1.0; event_pts += 1.0

        # Boost A: ADX 强趋势
        if ft["adx"] > c.adx_threshold:
            if direction == 1: bull_score += 0.5
            else:              bear_score += 0.5

        # Boost B: 成交量高于均值
        vr = ft.get("vol_ratio", float("nan"))
        if not np.isnan(vr):
            if vr >= 1.0:
                if direction == 1: bull_score += 0.3
                else:              bear_score += 0.3
            if vr >= 1.3:
                if direction == 1: bull_score += 0.2
                else:              bear_score += 0.2

        # Boost C: regime 顺势奖励（trend_up 做多 / trend_down 做空 额外 +0.5）
        regime = str(ft.get("regime", "ranging"))
        if (regime == "trend_up"   and direction ==  1) or \
           (regime == "trend_down" and direction == -1):
            regime_bonus = float(getattr(c, "regime_trend_score_bonus", 0.5))
            if direction == 1: bull_score += regime_bonus
            else:              bear_score += regime_bonus

        score = bull_score if direction == 1 else bear_score

        if score < c.min_entry_conditions:
            return {"direction": 0, "score": score, "signal_type": "none"}

        if curr_side == 0 and event_pts < c.min_event_points:
            if not self._strong_trend_override(ft, direction):
                return {"direction": 0, "score": score, "signal_type": "none"}

        # 金字塔加仓（trend regime 时允许额外加仓次数）
        if curr_side != 0 and int(np.sign(direction)) == curr_side:
            regime = str(ft.get("regime", "ranging"))
            in_trend_regime = (
                (regime == "trend_up"   and curr_side ==  1)
                or (regime == "trend_down" and curr_side == -1)
            )
            max_adds = c.max_pyramid_adds
            if in_trend_regime:
                max_adds += int(getattr(c, "regime_trend_extra_pyramid", 1))
            can_add = (
                ts.pyramid_count < max_adds
                and score >= c.min_entry_conditions + c.pyramid_score_boost
                and (ts.hold_bars - ts.last_add_bar) >= c.pyramid_min_interval
            )
            if can_add and ts.entry_price is not None and not np.isnan(atr) and atr > 0:
                profit = ((price - ts.entry_price) if curr_side == 1
                          else (ts.entry_price - price))
                if profit < c.pyramid_profit_atr * atr:
                    can_add = False
            if not can_add:
                return {"direction": 0, "score": score, "signal_type": "none"}
            ts.pyramid_count += 1
            ts.last_add_bar   = ts.hold_bars
            return {"direction": direction, "score": score, "signal_type": "trend_pyramid"}

        # 反向信号（持仓期 + ER 检查）
        if curr_side != 0 and int(np.sign(direction)) != curr_side:
            if ts.hold_bars < self._dynamic_hold_bars(ft["adx"]):
                return {"direction": 0, "score": score, "signal_type": "none"}
            if not np.isnan(er_val) and er_val < c.er_min_reverse_close:
                return {"direction": 0, "score": score, "signal_type": "none"}

        return {"direction": direction, "score": score, "signal_type": "trend"}

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _timing_gate_ok(self, direction: int, ft: Dict) -> bool:
        """新开仓 timing gate（两种都要：EMA_fast 回抽/反弹 + RSI 反转确认）。"""
        c = self._c
        price = float(ft.get("price", float("nan")))
        ema_fast = float(ft.get("ema_fast", float("nan")))
        rsi = float(ft.get("rsi", float("nan")))
        rsi_prev = float(ft.get("rsi_prev", float("nan")))

        if np.isnan(price) or price <= 0:
            return False

        # A) EMA_fast 回抽/反弹（使用现有 pullback_ema_* + pullback_rsi_* 阈值）
        if bool(c.timing_require_ema_pullback):
            if np.isnan(ema_fast) or ema_fast <= 0:
                return False
            p2e = (price - ema_fast) / ema_fast
            if direction == 1:
                if not (float(c.pullback_ema_lo) <= p2e <= float(c.pullback_ema_hi)):
                    return False
                # 多头回踩时要求 RSI 不要太高（避免追多）
                if not np.isnan(rsi) and rsi >= float(c.pullback_rsi_long_max):
                    return False
            else:
                if not (-float(c.pullback_ema_hi) <= p2e <= -float(c.pullback_ema_lo)):
                    return False
                # 空头反弹时要求 RSI 不要太低（避免追空）
                if not np.isnan(rsi) and rsi <= float(c.pullback_rsi_short_min):
                    return False

        # B) RSI 反转确认（用 rsi_pullback_* 做“反转触发区间”）
        if bool(c.timing_require_rsi_reversal):
            if np.isnan(rsi) or np.isnan(rsi_prev):
                return False
            if direction == 1:
                if not (rsi_prev <= float(c.rsi_pullback_long_max) and rsi > rsi_prev):
                    return False
            else:
                if not (rsi_prev >= float(c.rsi_pullback_short_min) and rsi < rsi_prev):
                    return False

        return True

    def _is_trending(self, ft: Dict) -> bool:
        c = self._c
        if not np.isnan(ft["bb_width"]) and ft["bb_width"] < c.bb_width_trend_min:
            return False
        if not np.isnan(ft["er"]) and ft["er"] < c.er_min_entry:
            return False
        return True

    def _strong_trend_override(self, ft: Dict, direction: int) -> bool:
        c = self._c
        if not self._is_trending(ft):
            return False
        adx = float(ft.get("adx", float("nan")))
        if np.isnan(adx) or adx < float(c.low_event_adx_override):
            return False
        er = ft.get("er", float("nan"))
        if not np.isnan(er) and er < float(c.er_min_entry) + float(c.strong_trend_er_boost):
            return False
        return True

    def _dynamic_hold_bars(self, adx: float) -> int:
        c    = self._c
        mult = min(adx / c.adx_threshold, 2.5) if adx > c.adx_threshold else 1.0
        return int(c.min_hold_bars * mult)

    def _set_pending(self, direction: int, level: float, window: int,
                     is_squeeze: bool = False) -> None:
        self._pending_dir        = direction
        self._pending_level      = level
        self._pending_ttl        = max(1, int(window))
        self._pending_retested   = False
        self._pending_is_squeeze = is_squeeze

    def _clear_pending(self) -> None:
        self._pending_dir = 0; self._pending_level = None
        self._pending_ttl = 0; self._pending_retested = False
        self._pending_is_squeeze = False

    def _has_pending(self) -> bool:
        return self._pending_dir != 0


# ===========================================================================
# RiskFilter — 风险过滤器 / 交易门控
# ===========================================================================

class RiskFilter:
    """分两层对交易信号进行门控过滤。

    Layer 1 — 系统风险门控（与方向无关，影响 info["risk_blocked"]）:
      - 冷却期 (cooldown)
      - 连败熔断 (consecutive loss circuit breaker)
      - 时间段过滤 (time-of-day filter)
      - 手续费/波动率边际 (fee-edge filter)

    Layer 2 — 方向有效性验证（输出 info["trade_signal"]）:
      - RSI 极值禁入 (RSI extreme block)
      - 趋势效率门槛 (ER threshold)
      - 过度延伸过滤 (price extension filter)

    公开接口：
      is_direction_blocked(d, features)  供 AlphaEngine 在信号生成期间调用
      set_reverse_cooldown(n)            供 OrderRouter 在反向开仓等待时调用
    """

    def __init__(self, cfg: Any) -> None:
        self._c = cfg
        self._cooldown_remaining:          int = 0
        self._consecutive_losses:          int = 0
        self._loss_pause_bars:             int = 0
        self._reverse_cooldown_remaining:  int = 0

    # ------------------------------------------------------------------
    # Pipeline 入口
    # ------------------------------------------------------------------

    def __call__(self, df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, Dict]:
        c         = self._c
        ft        = info["features"]
        ts: TradeState = info["trade_state"]
        curr_side = int(info.get("curr_side", 0))
        current_pos = float(info.get("current_pos", 0.0))

        if "decision_trace" not in info:
            info["decision_trace"] = []

        # 更新连败 / 平仓检测
        self._update_position_state(current_pos, ft["price"], ts,
                                    info.get("last_exit_price"))

        # Layer 1: 系统风险门控
        gate_cooldown  = self._cooldown_remaining > 0
        gate_consec    = self._consecutive_losses >= c.max_consecutive_losses
        gate_rev_cd    = curr_side == 0 and self._reverse_cooldown_remaining > 0
        gate_env       = self._env_gates(ft, curr_side)
        gate_time      = self._time_gate(df, curr_side)
        sys_blocked    = self._system_gates(curr_side) or gate_env or gate_time

        blocked_reason = None
        if gate_cooldown:           blocked_reason = "cooldown"
        elif gate_consec:           blocked_reason = "consecutive_losses"
        elif gate_rev_cd:           blocked_reason = "reverse_cooldown"
        elif gate_env:              blocked_reason = "env_gate"
        elif gate_time:             blocked_reason = "time_gate"

        info["risk_blocked"] = sys_blocked
        alpha = info.get("alpha", {})
        raw_dir = int(alpha.get("direction", 0))

        dir_blocked = False
        dir_blocked_reason = None
        if not sys_blocked and raw_dir != 0:
            if self.is_direction_blocked(raw_dir, ft,
                    is_squeeze=(alpha.get("signal_type") == "breakout")):
                dir_blocked        = True
                dir_blocked_reason = "direction_blocked"

        info["trade_signal"] = 0 if (sys_blocked or dir_blocked) else raw_dir

        info["decision_trace"].append({
            "module":          "RiskFilter",
            "sys_blocked":     sys_blocked,
            "dir_blocked":     dir_blocked,
            "blocked_reason":  blocked_reason or dir_blocked_reason,
            "cooldown":        self._cooldown_remaining,
            "consec_losses":   self._consecutive_losses,
            "trade_signal":    info["trade_signal"],
        })

        return df, info

    # ------------------------------------------------------------------
    # 公开：方向有效性检查（纯函数，可在 AlphaEngine 中提前调用）
    # ------------------------------------------------------------------

    def is_direction_blocked(
        self, d: int, ft: Dict,
        *, is_squeeze: bool = False,
    ) -> bool:
        """返回 True 表示该方向入场被阻止（不修改任何状态）。"""
        c     = self._c
        price = float(ft["price"])

        if c.long_only and d == -1:
            return True

        # 过度延伸过滤
        if c.use_extension_filter:
            atr = ft.get("atr", float("nan"))
            if not np.isnan(atr) and atr > 0:
                vwap_th = ft.get("ext_vwap_thresh", float("nan"))
                vwap_limit = float(vwap_th) if not np.isnan(vwap_th) else float(c.max_vwap_extension_atr)
                ema_th = ft.get("ext_ema_thresh", float("nan"))
                ema_limit = float(ema_th) if not np.isnan(ema_th) else float(c.max_ema_extension_atr)
                vwap = ft.get("vwap", float("nan"))
                if not np.isnan(vwap):
                    ext = (price - vwap) / atr
                    if (d ==  1 and ext >  vwap_limit) or (d == -1 and ext < -vwap_limit):
                        return True
                ema_s = ft.get("ema_slow", float("nan"))
                if not np.isnan(ema_s):
                    ext = (price - ema_s) / atr
                    if (d ==  1 and ext >  ema_limit) or (d == -1 and ext < -ema_limit):
                        return True

        # RSI 极值禁入
        rsi         = float(ft["rsi"])
        long_limit  = float(c.rsi_block_long_squeeze  if is_squeeze else c.rsi_block_long_over)
        short_limit = float(c.rsi_block_short_squeeze if is_squeeze else c.rsi_block_short_under)
        if (d ==  1 and rsi >= long_limit)  or (d == -1 and rsi <= short_limit):
            return True

        # ER 趋势效率门槛
        er = ft.get("er", float("nan"))
        if not np.isnan(er):
            er_thresh = float(c.er_min_entry_breakout if is_squeeze else c.er_min_entry)
            if er < er_thresh:
                return True

        return False

    def set_reverse_cooldown(self, n: int) -> None:
        """由 OrderRouter 在 close_then_wait flip 时调用。"""
        self._reverse_cooldown_remaining = max(0, int(n))

    # ------------------------------------------------------------------
    # Layer 1 门控（内部）
    # ------------------------------------------------------------------

    def _update_position_state(self, current_pos: float, price: float,
                                ts: TradeState, last_exit_price: Any) -> None:
        c        = self._c
        last_pos = ts.last_position

        closed   = (last_pos != 0.0 and current_pos == 0.0)
        reversed_ = (last_pos > 0 and current_pos < 0) or (last_pos < 0 and current_pos > 0)

        if (closed or reversed_) and ts.entry_price is not None:
            px = float(last_exit_price) if last_exit_price is not None else price
            ep = ts.entry_price
            pnl = ((px - ep) * abs(last_pos) if last_pos > 0
                   else (ep - px) * abs(last_pos))
            if pnl < 0:
                self._consecutive_losses += 1
                self._cooldown_remaining  = c.cooldown_bars
                self._loss_pause_bars     = 0
            else:
                self._consecutive_losses = 0
                self._loss_pause_bars    = 0

        ts.last_position = current_pos

    def _system_gates(self, curr_side: int) -> bool:
        c = self._c
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return True
        if self._consecutive_losses >= c.max_consecutive_losses:
            self._loss_pause_bars += 1
            if self._loss_pause_bars >= c.loss_pause_timeout:
                self._consecutive_losses = 0
                self._loss_pause_bars    = 0
            else:
                return True
        if curr_side == 0 and self._reverse_cooldown_remaining > 0:
            self._reverse_cooldown_remaining -= 1
            return True
        return False

    def _env_gates(self, ft: Dict, curr_side: int) -> bool:
        c     = self._c
        price = ft["price"]
        atr   = ft["atr"]

        if (c.use_fee_edge_filter and curr_side == 0
                and not np.isnan(atr) and price > 0):
            atr_pct = atr / price
            # 工业级 round-trip cost：fee + 预期单边滑点（k*ATR_pct 估算）
            slip_est       = float(getattr(c, "slippage_atr_pct", 0.1)) * atr_pct
            round_trip_cost = (c.fee_rate * 2.0) + (float(c.extra_cost_bps) / 10000.0) + slip_est * 2.0
            if atr_pct < round_trip_cost * c.edge_over_fee_mult:
                return True

        if any(np.isnan(v) for v in [ft["ema_fast"], ft["ema_slow"], ft["rsi"], ft["adx"]]):
            return True

        return False

    def _time_gate(self, df: pd.DataFrame, curr_side: int) -> bool:
        c = self._c
        if not c.use_time_filter or curr_side != 0:
            return False
        ts = None
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
            ts = df.index[-1]
        elif "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"].iloc[-1], errors="coerce", utc=True)
        if ts is not None and not pd.isna(ts):
            try:
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                if int(ts.hour) not in c.allowed_utc_hours:
                    return True
            except Exception:
                pass
        return False


# ===========================================================================
# ExitController — 出场控制器
# ===========================================================================

class ExitController:
    """管理持仓出场，按优先级依次检查。

    优先级：
      1. Emergency Stop      — 紧急止损（ROI 硬止损，无视所有等待条件）
      2. Structural Reversal — 结构性反转（EMA + ATR buffer，无 min 约束）
      3. ATR Trailing Stop   — 浮动追踪止损（受 min_hold_before_trailing 约束）
      4. Max Hold Timeout    — 最大持仓超时（受 min_hold_before_timeout 约束）

    min_position_bars 已拆分：
      min_hold_before_trailing : trailing stop 激活前的最小 bar 数
                                 或者达到 trail_activate_atr * ATR 浮盈时提前激活
      min_hold_before_timeout  : timeout 触发前的最小 bar 数（默认等于 max_hold_bars）

    regime 参与 exit 参数融合：
      trend regime 顺势 → 放宽 exit_ema_atr_mult / max_hold / 允许更多 pyramiding
      ranging regime → 使用更紧的出场参数（可选）

    输出：
      info["exit_signal"] : ±1 平仓方向，或 None（继续持有）
    """

    def __init__(self, cfg: Any) -> None:
        self._c          = cfg
        self._best_price: Optional[float] = None

    def __call__(self, df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, Dict]:
        curr_side      = int(info.get("curr_side", 0))
        ft             = info["features"]
        ts: TradeState = info["trade_state"]

        if "decision_trace" not in info:
            info["decision_trace"] = []

        exit_sig = self._check(ft, curr_side, ts)
        info["exit_signal"] = exit_sig
        info["best_price"]  = self._best_price

        info["decision_trace"].append({
            "module":      "ExitController",
            "curr_side":   curr_side,
            "hold_bars":   ts.hold_bars,
            "exit_signal": exit_sig,
            "best_price":  self._best_price,
            "entry_price": ts.entry_price,
            "regime":      ft.get("regime", "unknown"),
        })
        return df, info

    def reset_best_price(self) -> None:
        self._best_price = None

    # ------------------------------------------------------------------

    def _check(self, ft: Dict, curr_side: int, ts: TradeState) -> Optional[float]:
        c = self._c
        if curr_side == 0 or str(c.exit_authority).lower() == "exchange":
            return None

        price     = ft["price"]
        ema_slow  = ft["ema_slow"]
        ema_macro = ft.get("ema_macro", float("nan"))
        atr       = ft["atr"]
        macro_dir = int(ft.get("macro_dir", 0) or 0)
        regime    = str(ft.get("regime", "ranging"))

        # 方向性 ATR 倍数 / 最大持仓 bar
        ema_mult = (
            float(c.exit_ema_atr_mult_long)  if (curr_side ==  1 and c.exit_ema_atr_mult_long  is not None)
            else float(c.exit_ema_atr_mult_short) if (curr_side == -1 and c.exit_ema_atr_mult_short is not None)
            else float(c.exit_ema_atr_mult)
        )
        trail_mult = (
            float(c.exit_trail_atr_long)  if (curr_side ==  1 and c.exit_trail_atr_long  is not None)
            else float(c.exit_trail_atr_short) if (curr_side == -1 and c.exit_trail_atr_short is not None)
            else float(c.exit_trail_atr)
        )
        # max_hold: Optional[int]，None 表示不启用超时强平
        _base_hold = (
            c.max_hold_bars_long  if (curr_side ==  1 and c.max_hold_bars_long  is not None)
            else c.max_hold_bars_short if (curr_side == -1 and c.max_hold_bars_short is not None)
            else c.max_hold_bars
        )
        max_hold: Optional[int] = int(_base_hold) if _base_hold is not None else None

        # 宏观 & regime 顺势：放宽出场参数
        in_trend_regime = (
            (regime == "trend_up"   and curr_side ==  1)
            or (regime == "trend_down" and curr_side == -1)
        )
        if macro_dir == curr_side or in_trend_regime:
            if c.max_hold_bars_trend is not None:
                max_hold = (max(max_hold, int(c.max_hold_bars_trend))
                            if max_hold is not None else int(c.max_hold_bars_trend))
            ema_mult = max(float(ema_mult), float(c.exit_ema_atr_mult_trend))
            if not np.isnan(ema_macro):
                ema_slow = float(ema_macro)

        # ranging regime（持仓中切换到 ranging）→ 收紧出场参数
        if regime == "ranging" and not in_trend_regime:
            _ranging_ema = getattr(c, "exit_ema_atr_mult_ranging", None)
            ema_mult = min(float(ema_mult),
                          float(_ranging_ema) if _ranging_ema is not None else float(ema_mult) * 0.7)
            _ranging_hold = getattr(c, "max_hold_bars_ranging", None)
            if _ranging_hold is not None:
                max_hold = (min(max_hold, int(_ranging_hold))
                            if max_hold is not None else int(_ranging_hold))
            elif max_hold is not None:
                max_hold = max_hold // 2
            # _ranging_hold is None and max_hold is None → ranging 也不超时

        # 浮动盈亏
        pnl_pct = 0.0
        if ts.entry_price is not None and ts.entry_price > 0:
            pnl_pct = (price - ts.entry_price) / ts.entry_price * curr_side

        # 追踪最佳价格
        if self._best_price is None:
            self._best_price = price
        elif curr_side == 1 and price > self._best_price: self._best_price = price
        elif curr_side == -1 and price < self._best_price: self._best_price = price

        # 1. Emergency Stop（无 min bar 约束）
        roi_pnl = pnl_pct * ts.entry_leverage
        if roi_pnl < -c.emergency_exit_pct:
            return -float(curr_side)

        # 2. Structural Reversal（受 min_hold_before_structural 约束）
        _struct_cfg = getattr(c, "min_hold_before_structural", None)
        _min_struct = int(_struct_cfg) if _struct_cfg is not None else 0
        if ts.hold_bars >= _min_struct and not np.isnan(ema_slow) and not np.isnan(atr) and atr > 0:
            buf = atr * ema_mult
            if curr_side ==  1 and price < ema_slow - buf: return -float(curr_side)
            if curr_side == -1 and price > ema_slow + buf: return -float(curr_side)

        # 拆分后的两个 min bar 参数（Optional 字段默认 None，需显式回退）
        _min_pos      = int(getattr(c, "min_position_bars", 0) or 0)
        _trail_cfg    = getattr(c, "min_hold_before_trailing", None)
        _timeout_cfg  = getattr(c, "min_hold_before_timeout", None)
        min_trail   = int(_trail_cfg)   if _trail_cfg   is not None else _min_pos
        min_timeout = int(_timeout_cfg) if _timeout_cfg is not None else _min_pos

        # 3. ATR Trailing Stop（受 min_hold_before_trailing 约束）
        dynamic_min   = self._dynamic_hold_bars(ft["adx"])
        effective_min = max(dynamic_min, min_trail)
        if ts.hold_bars >= effective_min and not np.isnan(atr) and atr > 0:
            if ts.entry_price is not None and ts.entry_price > 0 and self._best_price is not None:
                best_profit = ((self._best_price - ts.entry_price) if curr_side == 1
                               else (ts.entry_price - self._best_price))
                # 浮盈激活（替代纯 bar 计数）
                profit_activated = best_profit >= atr * float(c.trail_activate_atr)
                if profit_activated or ts.hold_bars >= effective_min:
                    if profit_activated:
                        trail_dist = atr * trail_mult
                        if curr_side ==  1 and price < self._best_price - trail_dist:
                            return -float(curr_side)
                        if curr_side == -1 and price > self._best_price + trail_dist:
                            return -float(curr_side)

        # 4. Max Hold Timeout（max_hold 为 None 时跳过，完全依赖止损/止盈自然出场）
        if max_hold is not None:
            effective_max = max(int(max_hold), int(min_timeout))
            if ts.hold_bars >= effective_max:
                return -float(curr_side)

        return None

    def _dynamic_hold_bars(self, adx: float) -> int:
        c    = self._c
        mult = min(adx / c.adx_threshold, 2.5) if adx > c.adx_threshold else 1.0
        return int(c.min_hold_bars * mult)


# ===========================================================================
# PositionSizer — 仓位管理器
# ===========================================================================

class PositionSizer:
    """根据风险参数和资金量确定开仓规模。

    支持模式：fixed / notional_pct / margin_pct / risk_pct / volatility / kelly
    结果经过硬性约束（min/max notional、leverage 上下限）后输出。

    自适应缩放因子（乘在原始 notional 上）：
      regime_scale   : ranging → sizer_ranging_scale（默认 0.5）
                       trend   → sizer_trend_scale（默认 1.0）
    输出：info["sizing"] = {quantity, leverage, notional, regime_scale, ...}
    """

    def __init__(self, cfg: Any) -> None:
        self._c = cfg

    def __call__(self, df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, Dict]:
        c      = self._c
        equity = float(info.get("equity", 0.0))
        price  = float(df[PRICE_COL].astype(float).iloc[-1])

        if c.sizing_mode == "fixed":
            raw = _SizingAlgorithms.fixed(c.quantity, c.leverage, price)
        elif c.sizing_mode == "notional_pct":
            raw = _SizingAlgorithms.notional_percent(
                equity=equity, notional_pct=c.notional_pct, price=price,
                leverage=c.leverage, max_leverage=c.max_leverage,
                min_quantity=c.min_quantity, quantity_precision=c.quantity_precision)
        elif c.sizing_mode == "margin_pct":
            raw = _SizingAlgorithms.margin_percent(
                equity=equity, margin_pct=c.margin_pct, price=price,
                leverage=c.leverage, max_leverage=c.max_leverage,
                min_quantity=c.min_quantity, quantity_precision=c.quantity_precision)
        else:
            atr_col = f"atr_{c.atr_period}"
            atr     = float(df[atr_col].iloc[-1]) if atr_col in df.columns else 0.0
            if np.isnan(atr) or atr <= 0:
                raw = _SizingAlgorithms.fixed(c.quantity, c.leverage, price)
            elif c.sizing_mode == "risk_pct":
                raw = _SizingAlgorithms.risk_percent(
                    equity=equity, risk_pct=c.risk_per_trade, price=price, atr=atr,
                    atr_sl_multiplier=c.atr_sl_multiplier, max_leverage=c.max_leverage,
                    min_quantity=c.min_quantity, quantity_precision=c.quantity_precision)
            elif c.sizing_mode == "volatility":
                raw = _SizingAlgorithms.volatility_target(
                    equity=equity, target_vol=c.target_vol, price=price, atr=atr,
                    max_leverage=c.max_leverage, min_quantity=c.min_quantity,
                    quantity_precision=c.quantity_precision)
            elif c.sizing_mode == "kelly":
                raw = _SizingAlgorithms.kelly(
                    equity=equity, win_rate=c.kelly_win_rate,
                    avg_win=c.kelly_avg_win, avg_loss=1.0, price=price,
                    fraction=c.kelly_fraction, max_leverage=c.max_leverage,
                    min_quantity=c.min_quantity, quantity_precision=c.quantity_precision)
            else:
                raw = _SizingAlgorithms.fixed(c.quantity, c.leverage, price)

        # ── Regime 自适应缩放 ─────────────────────────────────────────────
        ft     = info.get("features", {})
        regime = str(info.get("regime", ft.get("regime", "ranging")))
        signal_dir = int(info.get("signal", 0))

        regime_scale = 1.0

        if regime == "ranging":
            regime_scale = float(getattr(c, "sizer_ranging_scale", 0.5))
        elif regime in ("trend_up", "trend_down"):
            trend_matches_signal = (
                (regime == "trend_up"   and signal_dir ==  1)
                or (regime == "trend_down" and signal_dir == -1)
            )
            regime_scale = float(getattr(c, "sizer_trend_scale",
                                         1.2 if trend_matches_signal else 0.8))

        raw["quantity"] = round(float(raw.get("quantity", 0.0)) * max(0.1, regime_scale),
                                int(c.quantity_precision))
        raw["notional"]     = float(raw["quantity"]) * price
        raw["regime_scale"] = regime_scale

        info["sizing"] = self._apply_constraints(raw, price, equity)
        return df, info

    def _apply_constraints(self, sizing: Dict, price: float, equity: float) -> Dict:
        c        = self._c
        qty      = sizing["quantity"]
        lev      = max(int(c.min_leverage), min(int(c.max_leverage), int(sizing["leverage"])))
        notional = qty * price

        if equity > 0:
            min_n = equity * float(c.min_notional_pct or 0.0)
            max_n = equity * float(c.max_notional_pct or 0.0)
            max_n_by_lev = equity * float(c.max_leverage)
            if max_n > 0: max_n = min(max_n, max_n_by_lev)
            else:         max_n = max_n_by_lev

            if min_n > 0 and notional < min_n: notional = min_n
            if max_n > 0 and notional > max_n: notional = max_n

            qty = max(notional / price if price > 0 else qty, c.min_quantity)
            qty = round(qty, c.quantity_precision)
            notional = qty * price

            if max_n > 0 and notional > max_n and price > 0:
                step = 10 ** int(c.quantity_precision)
                qty  = max(float(np.floor((max_n / price) * step) / step), 0.0)
                notional = qty * price

            lev = max(int(c.min_leverage),
                      min(int(c.max_leverage), max(int(lev), int(np.ceil(notional / equity)))))

        sizing["quantity"]  = float(qty)
        sizing["leverage"]  = int(lev)
        sizing["notional"]  = float(notional)
        if equity > 0:
            sizing["notional_pct"] = float(round(notional / equity, 4))
        sizing["clamped"]   = True
        return sizing


# ===========================================================================
# OrderRouter — 订单路由器
# ===========================================================================

class OrderRouter:
    """将交易信号转化为提交到交易所的订单。

    职责：
    - 订单方向 / 数量 / 类型（MARKET 或 LIMIT post-only）
    - SL/TP 设置（ROI 模式 / 固定价格 / 结构止损）
    - 反向仓位处理（flip_mode: immediate / close_only / close_then_wait）
    - 追踪止损更新指令

    Maker/Taker 自适应（use_adaptive_order_type=True）：
      - alpha score >= maker_score_threshold → MARKET（taker，立即成交）
      - alpha score  < maker_score_threshold → LIMIT post-only（maker，挂单意图）
        实盘层需将 LIMIT 订单挂在 best bid/ask 附近；
        若无法成交可降级为 MARKET（由 caller 决策）。

    注意（BAR-CLOSE 系统）：
      TradeState.entry_price 不再由 OrderRouter 写入。
      调用方必须在收到 fill 回报后调用 ts.update_from_fill(fill_price, leverage)。

    输出：
      info["order_result"]  : {signal, order, meta}
      info["trailing_stop"] : 追踪止损参数字典
    """

    def __init__(self, cfg: Any, risk_filter: RiskFilter,
                 feature_engine: FeatureEngine) -> None:
        self._c  = cfg
        self._rf = risk_filter       # 调用 set_reverse_cooldown
        self._fe = feature_engine    # 读取 _last_swing_high/low

    def __call__(self, df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, Dict]:
        c           = self._c
        signal      = int(info.get("signal", 0))
        current_pos = float(info.get("current_pos", 0.0))
        sizing      = info["sizing"]
        ts: TradeState = info["trade_state"]

        closes  = df[PRICE_COL].astype(float)
        price   = float(closes.iloc[-1])
        atr_col = f"atr_{c.atr_period}"
        atr     = (float(df[atr_col].iloc[-1])
                   if atr_col in df.columns and not np.isnan(df[atr_col].iloc[-1]) else 0.0)

        # 将 alpha score 注入 sizing 供 _build_order 使用
        alpha_info = info.get("alpha", {})
        sizing["_alpha_score"] = float(alpha_info.get("score", 0.0))

        result = self._build_order(signal, current_pos, sizing, price, atr, ts)
        info["order_result"] = result

        # 计算追踪止损
        effective_pos = current_pos
        if result["signal"] in ("LONG", "SHORT"):
            effective_pos = (sizing["quantity"] if result["signal"] == "LONG"
                             else -sizing["quantity"])
        info["trailing_stop"] = self._compute_trailing_stop(
            effective_pos, price, atr, ts, info.get("best_price"))
        return df, info

    # ------------------------------------------------------------------

    def _build_order(self, signal: int, current_pos: float,
                     sizing: Dict, price: float, atr: float,
                     ts: TradeState) -> Dict:
        c        = self._c
        cur_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
        cur_amt  = abs(current_pos)

        if signal == 0:
            return {"signal": "HOLD", "order": None, "meta": {}}

        open_qty = sizing.get("quantity", 0.0)
        if open_qty is None or float(open_qty) <= 0:
            return {"signal": "HOLD", "order": None,
                    "meta": {"blocked": sizing.get("blocked", "invalid_quantity")}}

        # 反向信号 → 先平仓
        if cur_side != 0 and cur_side != signal and c.flip_mode in ("close_only", "close_then_wait"):
            close_side = "SELL" if current_pos > 0 else "BUY"
            close_qty  = round(cur_amt, c.quantity_precision)
            if close_qty <= 0:
                return {"signal": "HOLD", "order": None, "meta": {}}
            order = {"symbol": c.symbol, "side": close_side, "type": "MARKET",
                     "quantity": close_qty, "position_side": "BOTH", "reduce_only": True}
            meta = {"mode": "close_only", "flip_mode": c.flip_mode,
                    "close_qty": close_qty, "close_side": close_side,
                    "requested_signal": "LONG" if signal == 1 else "SHORT"}
            if c.flip_mode == "close_then_wait":
                self._rf.set_reverse_cooldown(c.reverse_cooldown_bars)
                meta["reverse_cooldown_bars"] = int(c.reverse_cooldown_bars)
            return {"signal": "CLOSE", "order": order, "meta": meta}

        # 开仓 / 加仓
        total_qty  = round(open_qty, c.quantity_precision)
        order_side = "BUY" if signal == 1 else "SELL"

        # Maker/Taker 自适应订单类型
        alpha_score = float(sizing.get("_alpha_score", 0.0))
        maker_thresh = float(getattr(c, "maker_score_threshold", 3.0))
        use_adaptive = bool(getattr(c, "use_adaptive_order_type", True))
        if use_adaptive and alpha_score < maker_thresh:
            order_type = "LIMIT"
            order_intent = "post_only"  # 实盘层处理；回测层降级为 MARKET
        else:
            order_type = "MARKET"
            order_intent = "taker"

        order: Dict[str, Any] = {
            "symbol": c.symbol, "side": order_side, "type": order_type,
            "quantity": total_qty, "position_side": "BOTH",
            "order_intent": order_intent,
        }

        sl_dist, tp_dist, sl_meta = self._compute_sl_tp(signal, price, atr)
        meta: Dict[str, Any] = {**sl_meta}

        lev = max(1, int(sizing.get("leverage", 1) or 1))

        if str(c.exit_authority).lower() == "strategy":
            meta["tp_sl_mode"] = "NONE"
        elif c.use_roi_tpsl:
            sl_pct_v = float(c.sl_roi_pct)
            tp_pct_v = float(c.tp_roi_pct)
            sl_pct_eff = sl_pct_v
            if c.roi_sl_use_structure and sl_dist > 0:
                sl_roi_struct = (sl_dist / price) * 100.0 * float(lev)
                if sl_roi_struct > 0:
                    sl_pct_eff = min(sl_pct_v, float(sl_roi_struct))
            sl_dist_eff = price * (sl_pct_eff / 100.0) / float(lev)
            tp_dist_eff = price * (tp_pct_v  / 100.0) / float(lev)
            sl_dist     = sl_dist_eff
            tp_dist     = tp_dist_eff
            order["tp_sl_mode"]  = "ROI"
            order["stop_loss"]   = round(sl_pct_eff, 6)
            order["take_profit"] = round(tp_pct_v, 6)
            meta.update({"tp_sl_mode": "ROI_PCT", "sl_pct": round(sl_pct_eff, 4),
                          "tp_pct": round(tp_pct_v, 4),
                          "sl_distance": round(sl_dist_eff, 2),
                          "tp_distance": round(tp_dist_eff, 2)})
        elif sl_dist > 0:
            sl_price = round(price - sl_dist if signal == 1 else price + sl_dist, 2)
            order["tp_sl_mode"] = "PRICE"
            order["stop_loss"]  = sl_price
            meta.update({"sl_price": sl_price, "sl_distance": round(sl_dist, 2)})
            if tp_dist > 0:
                tp_price = round(price + tp_dist if signal == 1 else price - tp_dist, 2)
                order["take_profit"] = tp_price
                meta["tp_price"]     = tp_price

        # !! entry_price / entry_leverage 不再由 OrderRouter 写入 !!
        # 调用方收到 fill 回报后须调用: ts.update_from_fill(fill_price, leverage)
        ts.pending_fill   = True
        ts.entry_leverage = lev  # 暂存 leverage（fill 回报时会被覆盖为实际值）

        meta["fee"]          = compute_fee_info(price, open_qty, sl_dist, tp_dist, c.fee_rate)
        meta["sizing"]       = sizing
        meta["intent_price"] = price   # 下单时参考价（非成交价）
        meta["open_qty"]     = open_qty
        meta["order_type"]   = order_type
        meta["order_intent"] = order_intent

        sig_name = "LONG" if signal == 1 else "SHORT"
        return {"signal": sig_name, "order": order, "meta": meta}

    def _compute_sl_tp(self, signal: int, price: float,
                        atr: float) -> Tuple[float, float, Dict]:
        c = self._c
        sl_dist = tp_dist = 0.0
        sl_meta: Dict[str, Any] = {}

        if c.use_pct_sl_tp:
            sl_dist = price * c.sl_pct
        elif atr > 0:
            sl_dist = atr * c.atr_sl_multiplier

        if c.use_structure_sl and sl_dist > 0:
            buf = (atr * c.structure_sl_buffer_atr) if atr > 0 else (price * 0.001)
            struct_sl: Optional[float] = None
            if signal == 1 and self._fe._last_swing_low is not None:
                struct_sl = float(self._fe._last_swing_low) - buf
                if struct_sl >= price: struct_sl = None
            if signal == -1 and self._fe._last_swing_high is not None:
                struct_sl = float(self._fe._last_swing_high) + buf
                if struct_sl <= price: struct_sl = None
            if struct_sl is not None:
                struct_dist = abs(price - struct_sl)
                sl_dist     = min(max(struct_dist, price * c.structure_sl_min_pct),
                                  price * c.structure_sl_max_pct)
                sl_meta["sl_source"] = "structure"

        if sl_dist > 0 and not c.use_roi_tpsl and not c.use_trailing_stop:
            tp_dist = price * c.tp_pct if c.use_pct_sl_tp else atr * c.atr_tp_multiplier

        return sl_dist, tp_dist, sl_meta

    def _compute_trailing_stop(self, current_pos: float, price: float, atr: float,
                                ts: TradeState, best_price: Optional[float]) -> Dict:
        c        = self._c
        inactive = {"active": False, "should_update": False,
                    "sl_price": None, "best_price": None, "entry_price": ts.entry_price}

        if str(c.exit_authority).lower() == "exchange":  return inactive
        if c.use_roi_tpsl and str(c.exit_authority).lower() != "strategy": return inactive
        if current_pos == 0 or not c.use_trailing_stop or (atr <= 0 and not c.use_pct_sl_tp):
            return inactive

        ref         = ts.entry_price if ts.entry_price else price
        sl_distance = ref * c.sl_pct if c.use_pct_sl_tp else atr * c.atr_sl_multiplier
        bp          = best_price

        if current_pos > 0:
            if bp is None or price > bp: bp = price
            trailing_sl = round(bp - sl_distance, 2)
        else:
            if bp is None or price < bp: bp = price
            trailing_sl = round(bp + sl_distance, 2)

        should_update = False
        if ts.entry_price is not None:
            if current_pos > 0:
                should_update = trailing_sl > (ts.entry_price - sl_distance + sl_distance * 0.1)
            else:
                should_update = trailing_sl < (ts.entry_price + sl_distance - sl_distance * 0.1)

        return {"active": True, "should_update": should_update,
                "sl_price": trailing_sl, "best_price": bp,
                "entry_price": ts.entry_price, "sl_distance": round(sl_distance, 2)}


# ===========================================================================
# AdaptiveTrendFusionStrategy — 策略编排器
# ===========================================================================

class AdaptiveTrendFusionStrategy(Strategy):
    """自适应趋势融合策略 — Binance USDT-M 永续合约。

    工业级流水线：
      FeatureEngine → RegimeClassifier → AlphaEngine → RiskFilter
      → ExitController → PositionSizer → OrderRouter

    信号优先级：exit_signal > trade_signal (alpha)
    """

    # ── 策略元信息 ──────────────────────────────────────────────────────
    name:         str       = Field(default="adaptive_trend_fusion")
    description:  str       = Field(default="自适应多因子趋势跟踪策略，Binance USDT-M 永续合约")
    factor_names: List[str] = Field(
        default=["ema", "rsi", "adx", "atr", "bb_width", "vwap", "er", "roc"])

    # ── 交易参数 ────────────────────────────────────────────────────────
    symbol:             str   = Field(default="BTCUSDT")
    quantity:           float = Field(default=0.02)
    leverage:           int   = Field(default=5)
    quantity_precision: int   = Field(default=3)
    min_quantity:       float = Field(default=0.001)

    # ── 仓位管理 ────────────────────────────────────────────────────────
    sizing_mode:    str   = Field(default="margin_pct")
    risk_per_trade: float = Field(default=0.02)
    atr_sl_multiplier: float = Field(default=2.0)
    target_vol:     float = Field(default=0.10)
    kelly_win_rate: float = Field(default=0.55)
    kelly_avg_win:  float = Field(default=1.5)
    kelly_fraction: float = Field(default=0.5)
    notional_pct:   float = Field(default=1.0)
    margin_pct:     float = Field(default=0.25)

    # ── 仓位约束 ────────────────────────────────────────────────────────
    min_leverage:    int   = Field(default=1)
    max_leverage:    int   = Field(default=10)
    min_notional_pct: float = Field(default=0.0)
    max_notional_pct: float = Field(default=10.0)

    # ── 出场权威 ────────────────────────────────────────────────────────
    exit_authority: str = Field(default="strategy",
        description="strategy | exchange — exchange 时由交易所 SL/TP 负责出场")

    # ── SL / TP ─────────────────────────────────────────────────────────
    use_trailing_stop:  bool  = Field(default=True)
    atr_tp_multiplier:  float = Field(default=5.0)
    use_pct_sl_tp:      bool  = Field(default=True)
    sl_pct:             float = Field(default=0.03)
    tp_pct:             float = Field(default=0.10)

    # ── ROI TP/SL ───────────────────────────────────────────────────────
    use_roi_tpsl:          bool  = Field(default=True)
    tp_roi_pct:            float = Field(default=20.0, description="止盈 ROI%（×杠杆后的盈利百分比）")
    sl_roi_pct:            float = Field(default=10.0, description="止损 ROI%（×杠杆后的亏损百分比）")
    roi_sl_use_structure:  bool  = Field(default=False)

    # ── 手续费 ──────────────────────────────────────────────────────────
    fee_rate: float = Field(default=DEFAULT_TAKER_FEE)

    # ── 技术指标参数 ────────────────────────────────────────────────────
    ema_fast_period:   int   = Field(default=50)
    ema_slow_period:   int   = Field(default=200)
    ema_macro_period:  int   = Field(default=240)
    ema_trend_buffer:  float = Field(default=0.0006)
    adx_period:        int   = Field(default=14)
    adx_threshold:     float = Field(default=28.0)
    adx_no_trade:      float = Field(default=12.0)
    rsi_period:        int   = Field(default=14)
    rsi_oversold:      float = Field(default=38.0)
    rsi_overbought:    float = Field(default=62.0)
    rsi_pullback_long_max:   float = Field(default=45.0)
    rsi_pullback_short_min:  float = Field(default=55.0)
    rsi_extreme_bonus:       float = Field(default=0.5)
    atr_period:        int   = Field(default=14)
    bb_width_period:   int   = Field(default=20)
    vwap_period:       int   = Field(default=100)
    er_period:         int   = Field(default=60)

    # ── Alpha 信号评分参数 ──────────────────────────────────────────────
    trend_align_score:     float = Field(default=0.4)
    env_align_score:       float = Field(default=0.4)
    min_entry_conditions:  float = Field(default=2.0)
    min_event_points:      float = Field(default=0.5)
    low_event_adx_override: float = Field(default=35.0)
    strong_trend_er_boost: float = Field(default=0.03)

    # ── 回踩入场 ────────────────────────────────────────────────────────
    pullback_ema_lo:        float = Field(default=-0.001)
    pullback_ema_hi:        float = Field(default=0.0015)
    pullback_rsi_long_max:  float = Field(default=42.0)
    pullback_rsi_short_min: float = Field(default=58.0)
    swing_lookback:         int   = Field(default=60)
    vol_surge_lookback:     int   = Field(default=30)
    vol_surge_multiplier:   float = Field(default=1.2)
    breakout_chase_atr:     float = Field(default=0.6)

    # ── Squeeze Breakout ────────────────────────────────────────────────
    use_squeeze_breakout:     bool  = Field(default=True)
    bb_width_min_entry:       float = Field(default=0.005)
    squeeze_min_bars:         int   = Field(default=20)
    squeeze_breakout_window:  int   = Field(default=20)
    bb_width_expansion_pct:   float = Field(default=0.15)
    breakout_confirm_window:  int   = Field(default=12)
    breakout_retest_buf_pct:  float = Field(default=0.0008)
    breakout_confirm_buf_pct: float = Field(default=0.0003)
    breakout_cancel_buf_pct:  float = Field(default=0.0010)

    # ── 趋势效率 / 震荡过滤 ─────────────────────────────────────────────
    er_min_entry:           float = Field(default=0.07,
        description="趋势入场 ER 下限（降低到 0.07 让震荡期的趋势信号不被全部过滤）")
    er_min_entry_breakout:  float = Field(default=0.10,
        description="breakout 入场 ER 下限。从 0.06→0.10：ranging 市场里 ER 0.06 几乎任何小波动都能触发，"
                    "导致在局部底部/顶部做假突破；提高到 0.10 要求有更明确的方向性效率。")
    er_min_reverse_close:   float = Field(default=0.30)
    bb_width_trend_min:     float = Field(default=0.003)

    # ── 宏观趋势（ROC）─────────────────────────────────────────────────
    use_macro_slope:    bool  = Field(default=True)
    macro_slope_period: int   = Field(default=120)
    macro_slope_buf:    float = Field(default=0.0002)
    use_macro_bias_entry: bool  = Field(default=True)
    macro_bias_er_min:    float = Field(default=0.12)
    macro_align_bonus:    float = Field(default=0.5, description="ROC 宏观方向对齐时给 Alpha 加分（软偏置）")
    macro_contra_penalty: float = Field(default=0.7, description="ROC 宏观方向相反时给 Alpha 扣分（软偏置）")

    # Breakout 宏观反向 veto（默认开启，用于减少震荡期假突破逆势单）
    use_macro_breakout_veto: bool = Field(default=True)
    macro_breakout_veto_er_min: float = Field(default=0.25,
        description="启用 breakout veto 的 ER 下限；设高一些避免 ranging 市场中 macro_dir 来回摆时把合理 breakout 全 veto 掉")
    macro_breakout_veto_apply_to_squeeze: bool = Field(default=True, description="是否对 squeeze breakout 同样启用宏观反向 veto")

    # ── 过度延伸过滤 ────────────────────────────────────────────────────
    use_extension_filter:   bool  = Field(default=True)
    extension_adaptive:     bool  = Field(default=True)
    extension_lookback:     int   = Field(default=240)
    extension_percentile:   float = Field(default=0.95)
    max_vwap_extension_atr: float = Field(default=4.0)
    max_ema_extension_atr:  float = Field(default=3.0)

    # ── RSI 极值禁入 ────────────────────────────────────────────────────
    rsi_block_long_over:    float = Field(default=70.0)
    rsi_block_short_under:  float = Field(default=30.0)
    rsi_block_long_squeeze: float = Field(default=73.0)
    rsi_block_short_squeeze: float = Field(default=27.0)
    rsi_anti_chase_long_max: float = Field(default=68.0, description="避免追多：若 RSI>=该阈值则不新开多")
    rsi_anti_chase_short_min: float = Field(default=40.0, description="避免追空：若 RSI<=该阈值则不新开空")

    # ── 工业级入场 timing gate（两种都要）───────────────────────────────
    use_entry_timing_gate: bool = Field(default=True, description="新开仓需满足回抽/反弹 + RSI 反转确认")
    timing_require_ema_pullback: bool = Field(default=True, description="要求价格回抽/反弹到 EMA_fast 附近")
    timing_require_rsi_reversal: bool = Field(default=True, description="要求 RSI 反转确认（避免追单）")

    # ── 手续费边际过滤 ──────────────────────────────────────────────────
    use_fee_edge_filter: bool  = Field(default=True)
    edge_over_fee_mult:  float = Field(default=0.5)
    extra_cost_bps:      float = Field(default=3.0)

    # ── 时间过滤 ────────────────────────────────────────────────────────
    use_time_filter:    bool       = Field(default=False)
    allowed_utc_hours:  List[int]  = Field(default=list(range(7, 23)))

    # ── 结构止损 ────────────────────────────────────────────────────────
    use_structure_sl:          bool  = Field(default=True)
    structure_sl_buffer_atr:   float = Field(default=0.2)
    structure_sl_min_pct:      float = Field(default=0.015)
    structure_sl_max_pct:      float = Field(default=0.03)

    # ── 风控参数 ────────────────────────────────────────────────────────
    min_hold_bars:         int   = Field(default=5)
    min_position_bars:     int   = Field(default=30,
        description="默认 trailing/timeout 的最小持仓 bar（1m 上不应设超过 60，否则无法及时止损）")
    max_hold_bars:         Optional[int] = Field(default=None,
        description="最大持仓 bar（None=不限制，依赖止损/止盈逻辑自然出场；设具体值时为超时强平上限）")
    max_hold_bars_trend:   Optional[int] = Field(default=None,
        description="宏观顺势时最大持仓 bars（None=不限制；设具体值时作为趋势单超时上限）")
    cooldown_bars:         int   = Field(default=20)
    max_consecutive_losses: int  = Field(default=3)
    loss_pause_timeout:    int   = Field(default=200)
    emergency_exit_pct:    float = Field(default=0.04)

    # ── 出场参数 ────────────────────────────────────────────────────────
    exit_ema_atr_mult:        float          = Field(default=3.5,
        description="Structural Reversal 基础 ATR 倍数。从 2.5→3.5：给价格更多呼吸空间，"
                    "避免在正常波动中被触发，尤其是 ranging 市场里价格会在 EMA 附近来回振荡。")
    exit_ema_atr_mult_trend:  float          = Field(default=7.0,
        description="顺势 regime 下 Structural Reversal ATR 倍数（放宽到 7.0 让趋势单充分发展）")
    exit_trail_atr:           float          = Field(default=4.0,
        description="trailing stop 距离（ATR 倍数）。1m BTC 均值 ATR≈43 USDT，4x≈173 USDT(0.26%)，"
                    "足够过滤 1m 噪声；原值 2.0x≈87 USDT 两根 K 线就能打掉。")
    trail_activate_atr:       float          = Field(default=3.0,
        description="trailing stop 激活门槛（ATR 倍数）。3x≈130 USDT(0.19%) 浮盈后才开始追踪，"
                    "原值 0.8x≈35 USDT 不足一根平均 K 线振幅就激活，噪声极大。")
    exit_ema_atr_mult_long:   Optional[float] = Field(default=None)
    exit_ema_atr_mult_short:  Optional[float] = Field(default=None)
    exit_trail_atr_long:      Optional[float] = Field(default=None)
    exit_trail_atr_short:     Optional[float] = Field(default=None)
    max_hold_bars_long:       Optional[int]   = Field(default=None)
    max_hold_bars_short:      Optional[int]   = Field(default=None)

    # ── 入场门槛 ────────────────────────────────────────────────────────
    require_trend_env_for_trend_entry: bool = Field(default=True)
    long_only: bool = Field(default=False)

    # ── 反向信号处理 ────────────────────────────────────────────────────
    flip_mode:             str = Field(default="close_then_wait")
    reverse_cooldown_bars: int = Field(default=15)

    # ── 金字塔加仓 ──────────────────────────────────────────────────────
    max_pyramid_adds:     int   = Field(default=0)
    pyramid_score_boost:  float = Field(default=1.0)
    pyramid_min_interval: int   = Field(default=5)
    pyramid_profit_atr:   float = Field(default=1.0)

    # ── Regime Gate（Issue#1）───────────────────────────────────────────
    use_regime_gate:              bool  = Field(default=True,
        description="ranging 时禁止趋势单入场（regime 真正参与融合）")
    allow_trend_in_ranging:       bool  = Field(default=False,
        description="False = ranging 时禁止趋势单；True = 降权但允许")
    breakout_ok_in_ranging:       bool  = Field(default=True,
        description="ranging 时仍允许 breakout/squeeze 信号（但受 macro veto 约束）")
    regime_trend_score_bonus:     float = Field(default=0.5,
        description="trend regime 顺势方向开仓时的 alpha 评分奖励")
    regime_trend_extra_pyramid:   int   = Field(default=1,
        description="trend regime 顺势时允许的额外金字塔加仓次数")

    # ── 出场 regime 参数（ranging 收紧 / trend 放宽）─────────────────────
    exit_ema_atr_mult_ranging:    Optional[float] = Field(default=2.5,
        description="ranging 时结构止损 ATR 倍数上限。从 None（0.7×=1.75）→ 2.5：ranging 市场价格振荡，"
                    "1.75× 太窄会把正常回调也止损出去；2.5× 给了足够缓冲。")
    max_hold_bars_ranging:        Optional[int]   = Field(default=None,
        description="ranging 时最大持仓 bar 上限（None=若 max_hold_bars 也为 None 则不限制；否则取 max_hold_bars//2）")

    # ── Maker/Taker 自适应（Issue#2）────────────────────────────────────
    use_adaptive_order_type:  bool  = Field(default=True,
        description="True=信号强用 MARKET，弱用 LIMIT post-only")
    maker_score_threshold:    float = Field(default=3.0,
        description="alpha score 低于此值时建议挂单（LIMIT）")
    slippage_atr_pct:         float = Field(default=0.1,
        description="单边滑点估算系数 k（预期滑点 = k * ATR / price）")

    # ── min_hold 拆分（Issue#4）─────────────────────────────────────────
    min_hold_before_trailing:    Optional[int] = Field(default=None,
        description="ATR trailing stop 激活前的最小 bar 数（None 时回退到 min_position_bars）")
    min_hold_before_timeout:     Optional[int] = Field(default=None,
        description="timeout 出场前的最小 bar 数（None 时回退到 min_position_bars）")
    min_hold_before_structural:  Optional[int] = Field(default=20,
        description="Structural Reversal（EMA±ATR buffer）触发前的最小 bar 数，"
                    "防止入场后立即被小幅反弹止损。None/0 = 无约束（原行为）。")

    # ── Sizer regime 缩放 ───────────────────────────────────────────────
    sizer_ranging_scale:  float = Field(default=0.5,
        description="ranging regime 时仓位缩放因子（0.5 = 半仓）")
    sizer_trend_scale:    float = Field(default=1.0,
        description="trend regime 顺势时仓位缩放因子（可>1 放大，但受 max_notional 约束）")

    # ── 兼容性保留字段 ──────────────────────────────────────────────────
    use_di_filter:          bool  = Field(default=False)
    di_min_diff:            float = Field(default=2.0)
    use_macro_trend:        bool  = Field(default=False)
    macro_ema_hard_buf_pct: float = Field(default=0.002)
    use_vwap_filter:        bool  = Field(default=True)
    vwap_filter_buf_pct:    float = Field(default=0.003)

    # ==================================================================
    # 初始化 — 实例化所有流水线模块
    # ==================================================================

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # ── 配置互斥校验（Issue#3）──────────────────────────────────────────
        # exit_authority="strategy" 时 use_roi_tpsl 若为 True 会造成认知误导，
        # 策略层并不实际下交易所 SL/TP 单，但 use_roi_tpsl=True 的 flag 容易被误读。
        # 行为规则：strategy 模式下强制关闭 use_roi_tpsl，或允许同时存在但打印警告。
        if str(self.exit_authority).lower() == "strategy" and bool(self.use_roi_tpsl):
            import warnings
            warnings.warn(
                "AdaptiveTrendFusionStrategy: exit_authority='strategy' + use_roi_tpsl=True "
                "— 交易所 ROI SL/TP 不会被实际下单（tp_sl_mode 强制为 NONE）。"
                "如需交易所硬止损请设置 exit_authority='exchange'；"
                "如只需策略层软出场请设置 use_roi_tpsl=False。"
                "当前行为：use_roi_tpsl 将被忽略。",
                stacklevel=2,
            )
            object.__setattr__(self, "use_roi_tpsl", False)

        self._trade_state = TradeState()

        # 实例化顺序反映依赖关系：
        # RiskFilter 先于 AlphaEngine（AlphaEngine 需引用 RiskFilter）
        # FeatureEngine 先于 OrderRouter（OrderRouter 需读取 swing 高低点）
        self._feature_engine    = FeatureEngine(self)
        self._regime_classifier = RegimeClassifier(self)
        self._risk_filter       = RiskFilter(self)
        self._alpha_engine      = AlphaEngine(self, self._risk_filter)
        self._exit_controller   = ExitController(self)
        self._position_sizer    = PositionSizer(self)
        self._order_router      = OrderRouter(self, self._risk_filter, self._feature_engine)

    # ==================================================================
    # __call__ — 流水线编排主入口
    # ==================================================================

    async def __call__(
        self,
        df: pd.DataFrame,
        current_pos: float = 0.0,
        equity: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """接收 K 线 DataFrame，输出 Binance 合约订单。"""

        # ── Step 0: 预计算因子 ────────────────────────────────────────
        await factor_manager("ema",      df)
        await factor_manager("rsi",      df)
        await factor_manager("atr",      df)
        await factor_manager("adx",      df)
        await factor_manager("bb_width", df, periods=[self.bb_width_period])
        await factor_manager("vwap",     df, periods=[self.vwap_period])
        await factor_manager("er",       df, periods=[self.er_period])
        if self.use_macro_slope:
            await factor_manager("roc",  df, windows=[self.macro_slope_period])

        for p in (self.ema_fast_period, self.ema_slow_period, self.ema_macro_period):
            col = f"ema_{p}"
            if col not in df.columns:
                df[col] = df["close"].ewm(span=int(p), adjust=False).mean()

        # 数据量检查
        min_bars = max(
            self.ema_slow_period, self.ema_macro_period,
            self.adx_period, self.rsi_period, self.atr_period,
            self.swing_lookback, self.bb_width_period,
            self.vwap_period, self.er_period,
            (self.macro_slope_period if self.use_macro_slope else 1),
        ) + 10
        if df is None or df.empty or len(df) < min_bars:
            return self._empty_result()
        if PRICE_COL not in df.columns or VOL_COL not in df.columns:
            return self._empty_result()

        # ── Step 1: 构建初始 info ─────────────────────────────────────
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
        info: Dict[str, Any] = {
            "current_pos":    current_pos,
            "equity":         equity,
            "curr_side":      curr_side,
            "trade_state":    self._trade_state,
            "last_exit_price": kwargs.get("last_exit_price"),
        }

        # ── Step 2: 流水线执行 ────────────────────────────────────────
        df, info = self._feature_engine(df, info)
        df, info = self._regime_classifier(df, info)

        # regime 传入 features dict（供后续模块使用 ft.get("regime")）
        info["features"]["regime"] = info.get("regime", "ranging")

        df, info = self._alpha_engine(df, info)     # 生成候选 Alpha 信号
        df, info = self._risk_filter(df, info)      # 验证 / 门控

        if info["risk_blocked"]:
            # 虽然被门控阻断，但仍要把已计算好的 regime / 指标写入 factors_json，
            # 确保 trading_log.csv 能正确记录每一个 bar 的状态（logging 修复）。
            _ts   = self._trade_state
            _ft   = info.get("features", {})
            _fcols = [
                f"ema_{self.ema_fast_period}", f"ema_{self.ema_slow_period}",
                f"ema_{self.ema_macro_period}",
                f"rsi_{self.rsi_period}", f"adx_{self.adx_period}",
                f"plus_di_{self.adx_period}", f"minus_di_{self.adx_period}",
                f"atr_{self.atr_period}", f"bb_width_{self.bb_width_period}",
                f"vwap_{self.vwap_period}", f"er_{self.er_period}",
            ]
            if self.use_macro_slope:
                _fcols.append(f"roc_{self.macro_slope_period}")
            _alpha_info = info.get("alpha", {})
            _blocked_fj = build_factors_json(df, _fcols, extra={
                "hold_bars":          _ts.hold_bars,
                "consecutive_losses": self._risk_filter._consecutive_losses,
                "cooldown_remaining": self._risk_filter._cooldown_remaining,
                "regime":             info.get("regime", ""),
                "macro_dir":          info.get("macro_dir", 0),
                "alpha_score":        float(_alpha_info.get("score", 0.0)),
                "alpha_type":         str(_alpha_info.get("signal_type", "none")),
                "funding_rate": _ft.get("funding_rate"),
                "regime_scale": None,
            })
            return self._empty_result(info, factors_json=_blocked_fj)

        df, info = self._exit_controller(df, info)  # 出场检查

        # ── Step 3: 信号分辨率（exit > alpha）────────────────────────
        ts        = self._trade_state
        exit_sig  = info.get("exit_signal")
        trade_sig = int(info.get("trade_signal", 0))

        if curr_side != 0 and exit_sig is not None:
            # 持仓中，出场条件触发
            info["signal"] = int(np.sign(exit_sig)) if exit_sig else 0
            if info["signal"] != 0:
                self._exit_controller.reset_best_price()
        elif trade_sig != 0:
            # 无持仓或未触发出场，执行 Alpha 信号
            info["signal"] = trade_sig
        else:
            info["signal"] = 0

        # ── Step 4: 仓位计算 + 订单路由 ──────────────────────────────
        df, info = self._position_sizer(df, info)
        df, info = self._order_router(df, info)

        # ── Step 5: 组装返回值 ────────────────────────────────────────
        result   = info["order_result"]
        sizing   = info["sizing"]
        trailing = info["trailing_stop"]

        atr_col     = f"atr_{self.atr_period}"
        current_atr = (float(df[atr_col].iloc[-1])
                       if atr_col in df.columns and not np.isnan(df[atr_col].iloc[-1]) else 0.0)

        factor_cols = [
            f"ema_{self.ema_fast_period}", f"ema_{self.ema_slow_period}",
            f"ema_{self.ema_macro_period}",
            f"rsi_{self.rsi_period}",
            f"adx_{self.adx_period}",
            f"plus_di_{self.adx_period}", f"minus_di_{self.adx_period}",
            f"atr_{self.atr_period}",
            f"bb_width_{self.bb_width_period}",
            f"vwap_{self.vwap_period}",
            f"er_{self.er_period}",
        ]
        if self.use_macro_slope:
            factor_cols.append(f"roc_{self.macro_slope_period}")

        alpha_info   = info.get("alpha", {})
        ft           = info.get("features", {})
        factors_json = build_factors_json(df, factor_cols, extra={
            "hold_bars":          ts.hold_bars,
            "consecutive_losses": self._risk_filter._consecutive_losses,
            "cooldown_remaining": self._risk_filter._cooldown_remaining,
            "regime":             info.get("regime", "unknown"),
            "macro_dir":          info.get("macro_dir", 0),
            "alpha_score":        float(alpha_info.get("score", 0.0)),
            "alpha_type":         str(alpha_info.get("signal_type", "none")),
            "funding_rate": ft.get("funding_rate"),
            "regime_scale": sizing.get("regime_scale"),
        })

        meta              = result.get("meta", {})
        meta["factors_json"] = factors_json

        return {
            "signal":       result["signal"],
            "order":        result["order"],
            "leverage":     sizing["leverage"],
            "sizing":       sizing,
            "atr":          current_atr,
            "meta":         meta,
            "factors_json": factors_json,
            "risk": {
                "cooldown_remaining": self._risk_filter._cooldown_remaining,
                "consecutive_losses": self._risk_filter._consecutive_losses,
                "loss_pause_bars":    self._risk_filter._loss_pause_bars,
                "hold_bars":          ts.hold_bars,
                "regime":             info.get("regime", "unknown"),
                "alpha_score":        float(alpha_info.get("score", 0.0)),
                "alpha_type":         str(alpha_info.get("signal_type", "none")),
            },
            "trailing_stop":   trailing,
            "decision_trace":  info.get("decision_trace", []),
        }

    def _empty_result(
        self,
        info: Optional[Dict] = None,
        *,
        factors_json: Optional[str] = None,
    ) -> Dict[str, Any]:
        ts = self._trade_state
        return {
            "signal": "HOLD", "order": None, "leverage": self.min_leverage,
            "sizing": {"quantity": 0.0, "leverage": self.min_leverage, "notional": 0.0},
            "atr": 0.0, "meta": {},
            # 优先使用调用方传入的 factors_json（含 regime/指标），
            # 回退到空 JSON（warmup 阶段无指标可记录）
            "factors_json": factors_json if factors_json is not None else "{}",
            "risk": {
                "cooldown_remaining": self._risk_filter._cooldown_remaining,
                "consecutive_losses": self._risk_filter._consecutive_losses,
                "loss_pause_bars":    self._risk_filter._loss_pause_bars,
                "hold_bars":          ts.hold_bars,
                "regime":             (info.get("regime", "unknown") if info else "unknown"),
                "alpha_score": 0.0, "alpha_type": "none",
            },
            "trailing_stop": {
                "active": False, "should_update": False,
                "sl_price": None, "best_price": None, "entry_price": ts.entry_price,
            },
        }
