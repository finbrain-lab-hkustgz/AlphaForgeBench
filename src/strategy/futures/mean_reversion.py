"""
均值回归策略 (Mean Reversion) — v2.0
适用市场：Binance USDT-M 永续合约（默认 BTCUSDT）

═══════════════════════════════════════════════════════════════════════
整体流水线
═══════════════════════════════════════════════════════════════════════
  FeatureEngine → EnterController → ExitController → PositionSizer → OrderRouter

  · FeatureEngine    提取 RSI / BB / ATR / ADX / VWAP / Z-Score / ER / CHOP / Hurst /
                     EMA / KDJ / MFI / ATR 分位数 / Funding 等指标
  · EnterController  市场状态检查（Regime + Trend） → 多因子梯度评分 →
                     硬条件门控（BB 触碰 + 极值确认）→ 盈利性检查（手续费覆盖 + ATR 距离）
  · ExitController   动态均值止盈 + 保本机制 + 最大持仓 + DI/ADX 动量衰竭 + 风控管理
  · PositionSizer    按评分动态调整 risk_per_trade + notional_pct 缩放
  · OrderRouter      信号合并、附加 ROI TP/SL 条件单、构建最终订单

═══════════════════════════════════════════════════════════════════════
入场逻辑
═══════════════════════════════════════════════════════════════════════

  环境检查:
    1. 自适应 Regime — ADX / ER / CHOP / Hurst / BB Width
       · ADX < 25 (loose): CHOP OR Hurst 任一通过即可
       · ADX ≥ 25 (strict): CHOP AND Hurst 必须都通过
    2. 趋势方向过滤 — EMA50/200 spread + slope
       · 强趋势（slope ≥ 2.5× 阈值）→ 禁止所有方向入场
       · 中度趋势 → 禁止逆势方向入场
    3. ATR 分位数 — 高波动（> 50 分位）禁入

  多因子评分（需 ≥ 2.2 分）:
    - RSI 梯度     (极端 1.5 / 超买超卖 1.0 / 轻度 0.3)
    - KDJ 超买超卖  (0.8) + 金/死叉加成 (0.4)
    - MFI 资金流    (0.8)
    - 偏离桶        BB 触碰 (0.8~1.2) + Z-Score (1.0~1.5) + VWAP 偏离 (0.5)
                    取 top1 + 0.4× top2，封顶 2.5
    - Funding 极端  (0.3)
    - Regime 惩罚   ADX 高 (-0.5) / ER 高 (-0.4)

  硬条件门控:
    · BB 外轨触碰（含 0.4% 缓冲）
    · 极值确认（RSI 超买超卖 / Z-Score 阈值 / KDJ 超买超卖）
    · 盈利性检查（预期回归距离 ≥ 4× 手续费 且 ≥ 1.0× ATR）

═══════════════════════════════════════════════════════════════════════
出场逻辑
═══════════════════════════════════════════════════════════════════════

  1. 动态均值止盈   价格触碰 BB 中轨 或 VWAP → 回归完成，平仓
  2. 保本机制       浮盈 > 1.2× ATR → 标记保本；价格回到开仓价附近 → 止盈离场
  3. 最大持仓       180 bar → 强制平仓
  4. 动量衰竭       DI 不利差值 > 20 或 ADX > 30 → 平仓（需持仓 ≥ 15 bar）
  5. ROI TP/SL      保底止盈 20% / 止损 10%（交易所条件单兜底）

═══════════════════════════════════════════════════════════════════════
风控
═══════════════════════════════════════════════════════════════════════
  · 亏损冷却 15 bar / 盈利冷却 15 bar
  · 连续 3 次亏损 → 暂停 120 bar（2 小时）
  · close_then_wait 反向冷却 15 bar
  · 评分动态仓位调整（信号强度 → risk 加成，notional 缩放）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import Field

from src.strategy.types import Strategy
from src.factor import factor_manager
from src.strategy.futures._utils import (
    PRICE_COL, VOL_COL, DEFAULT_TAKER_FEE, safe_float,
    calc_position_size, build_order, compute_trailing_stop, build_factors_json,
)
from src.factor.factors.funding_percentile import FUNDING_COL


# ===========================================================================
# TradeState — 交易状态跟踪
# ===========================================================================

@dataclass
class TradeState:
    """跟踪当前交易的实时状态。"""
    hold_bars:            int             = 0
    last_side:            int             = 0
    cooldown_remaining:   int             = 0
    consecutive_losses:   int             = 0
    loss_pause_bars:      int             = 0
    last_position:        float           = 0.0
    entry_price:          Optional[float] = None
    entry_leverage:       int             = 1
    best_price:           Optional[float] = None
    last_entry_score:     float           = 0.0
    reverse_cooldown:     int             = 0
    last_diag:            Dict[str, Any]  = field(default_factory=dict)
    breakeven_triggered:  bool            = False
    pending_fill:         bool            = False
    tpsl_closed:          bool            = False

    def update_from_fill(self, fill_price: float, leverage: int) -> None:
        """成交回报到达后由 caller 调用。"""
        self.entry_price = float(fill_price)
        self.entry_leverage = max(1, int(leverage))
        self.pending_fill = False


# ===========================================================================
# FeatureEngine — 指标提取
# ===========================================================================

class FeatureEngine:
    """从 DataFrame 提取均值回归策略所需的全部指标特征。"""

    def __init__(self, cfg: Any) -> None:
        self._c = cfg

    def __call__(self, df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, Dict]:
        c = self._c

        def _last(col: str, default: float = float("nan")) -> float:
            if col not in df.columns:
                return default
            try:
                v = df[col].iloc[-1]
                return float(v) if v is not None and not pd.isna(v) else default
            except (TypeError, ValueError, IndexError):
                return default

        price = float(df[PRICE_COL].iloc[-1])
        atr = _last(f"atr_{c.atr_period}")

        # BB 相关
        bb_upper = _last(f"bb_upper_{c.bb_period}")
        bb_lower = _last(f"bb_lower_{c.bb_period}")
        bb_middle = _last(f"bb_middle_{c.bb_period}")
        bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0.0

        # Z-Score（可能需要就地计算）
        zs_col = f"zscore_{c.zscore_period}"
        zscore = _last(zs_col, 0.0)

        # VWAP
        vwap = _last(f"vwap_{c.vwap_period}")

        # EMA
        ema_fast = _last(f"ema_{c.ema_fast_period}", float("nan"))
        ema_slow = _last(f"ema_{c.ema_slow_period}", float("nan"))

        # EMA slope & spread
        spread = 0.0
        slope = 0.0
        if not np.isnan(ema_fast) and not np.isnan(ema_slow) and ema_slow > 0:
            spread = (ema_fast - ema_slow) / ema_slow
            if c.ema_slope_window > 0 and len(df) > c.ema_slope_window:
                slow_col = f"ema_{c.ema_slow_period}"
                if slow_col in df.columns:
                    prev = safe_float(df[slow_col].astype(float).iloc[-1 - c.ema_slope_window], float("nan"))
                    if not np.isnan(prev) and prev > 0:
                        slope = (ema_slow - prev) / prev

        # Regime 因子
        er_col = f"er_{c.er_period}"
        er_val = _last(er_col, float("nan")) if er_col in df.columns else float("nan")

        chop_col = f"chop_{c.chop_period}"
        chop_val = _last(chop_col, float("nan")) if chop_col in df.columns else float("nan")

        hurst_col = f"hurst_{c.hurst_window}"
        hurst_val = _last(hurst_col, float("nan")) if hurst_col in df.columns else float("nan")

        # ATR percentile
        atr_pct_col = f"atr_pct_{c.atr_period}_{c.atr_percentile_window}"
        atr_pct = _last(atr_pct_col, float("nan")) if atr_pct_col in df.columns else float("nan")

        # Funding
        funding = _last(FUNDING_COL, 0.0) if FUNDING_COL in df.columns else 0.0

        info["features"] = {
            "price":       price,
            "atr":         atr,
            "bb_upper":    bb_upper,
            "bb_lower":    bb_lower,
            "bb_middle":   bb_middle,
            "bb_width":    bb_width,
            "rsi":         _last(f"rsi_{c.rsi_period}", 50.0),
            "stoch_k":     _last(f"stoch_k_{c.kdj_period}", 50.0),
            "stoch_d":     _last(f"stoch_d_{c.kdj_period}", 50.0),
            "mfi":         _last(f"mfi_{c.mfi_period}", 50.0),
            "zscore":      zscore,
            "vwap":        vwap,
            "adx":         _last(f"adx_{c.adx_period}", 30.0),
            "plus_di":     _last(f"plus_di_{c.adx_period}", 0.0),
            "minus_di":    _last(f"minus_di_{c.adx_period}", 0.0),
            "ema_fast":    ema_fast,
            "ema_slow":    ema_slow,
            "ema_spread":  spread,
            "ema_slope":   slope,
            "er":          er_val,
            "chop":        chop_val,
            "hurst":       hurst_val,
            "atr_pct":     atr_pct,
            "funding":     funding,
        }
        return df, info


# ===========================================================================
# EnterController — 入场信号
# ===========================================================================

class EnterController:
    """
    均值回归入场系统。

    流程：
      § 1  守卫检查   — 冷却 / 暂停 / 已持仓 / 反向冷却
      § 2  Regime 检查 — ADX / ER / CHOP / Hurst / BB Width
      § 3  趋势过滤   — EMA slope + spread
      § 4  ATR 分位数门控
      § 5  多因子评分 — RSI / KDJ / MFI / 偏离桶 / Funding
      § 6  硬条件检查 — BB 触碰 + 极值确认
      § 7  方向决策
      § 8  盈利性检查 — 手续费覆盖 + ATR 距离
    """

    def __init__(self, cfg: Any) -> None:
        self._c = cfg

    # ------------------------------------------------------------------
    # § 2  Regime 检查
    # ------------------------------------------------------------------

    def _check_regime(self, info: Dict) -> Dict[str, Any]:
        """自适应检查市场状态 (Adaptive Regime Filter)。"""
        c = self._c
        ft = info["features"]
        result: Dict[str, Any] = {"pass": True, "reason": "", "details": {}}

        adx = ft["adx"]
        result["details"]["adx"] = adx

        if adx > c.adx_max_threshold:
            result["details"]["adx_penalty"] = True

        # ER
        er_val = ft["er"]
        if not np.isnan(er_val) and er_val > c.er_max_entry:
            result["details"]["er_penalty"] = True
            result["details"]["er"] = er_val

        # CHOP & Hurst — 自适应逻辑
        chop_val = ft["chop"]
        hurst_val = ft["hurst"]
        chop_ok = True if np.isnan(chop_val) else (chop_val >= c.chop_min_entry)
        hurst_ok = True if np.isnan(hurst_val) else (hurst_val <= c.hurst_max_entry)
        result["details"]["chop"] = chop_val
        result["details"]["hurst"] = hurst_val

        is_strict_mode = adx > c.adx_strict_threshold
        result["details"]["mode"] = "strict" if is_strict_mode else "loose"

        if is_strict_mode:
            if not chop_ok or not hurst_ok:
                result["pass"] = False
                result["reason"] = "regime_strict_fail"
        else:
            if not chop_ok and not hurst_ok:
                result["pass"] = False
                result["reason"] = "regime_loose_fail"

        # BB Width
        bb_width = ft["bb_width"]
        if c.bb_width_min > 0 and bb_width < c.bb_width_min:
            result["pass"] = False
            result["reason"] = "bb_width_too_low"
        if bb_width > c.bb_width_max:
            result["pass"] = False
            result["reason"] = "bb_width_too_high"
        result["details"]["bb_width"] = bb_width

        return result

    # ------------------------------------------------------------------
    # § 3  趋势过滤
    # ------------------------------------------------------------------

    def _check_trend_filter(self, info: Dict) -> Dict[str, Any]:
        """检查趋势方向。"""
        c = self._c
        ft = info["features"]
        result: Dict[str, Any] = {"block_long": False, "block_short": False, "details": {}}

        if not c.trend_filter_enabled:
            return result

        ema_fast = ft["ema_fast"]
        ema_slow = ft["ema_slow"]
        if np.isnan(ema_fast) or np.isnan(ema_slow) or ema_slow <= 0:
            return result

        spread = ft["ema_spread"]
        slope = ft["ema_slope"]
        abs_slope = abs(slope)
        trending = abs_slope >= c.ema_slope_threshold
        strong_trend = trending and abs_slope >= c.ema_slope_threshold * c.strong_trend_slope_mult

        trend_dir = 0
        if trending:
            if spread > 0 and slope > 0:
                trend_dir = 1
            elif spread < 0 and slope < 0:
                trend_dir = -1

        result["details"] = {
            "slope": slope, "spread": spread, "trending": trending,
            "strong_trend": strong_trend, "dir": trend_dir,
        }

        if strong_trend and c.trend_block_strong:
            result["block_long"] = True
            result["block_short"] = True
            return result

        if c.trend_block_opposite:
            if trend_dir == 1:
                result["block_short"] = True
            elif trend_dir == -1:
                result["block_long"] = True

        return result

    # ------------------------------------------------------------------
    # § 5  多因子评分
    # ------------------------------------------------------------------

    def _calculate_scores(
        self, df: pd.DataFrame, info: Dict, regime_penalty: float,
    ) -> Dict[str, float]:
        """计算多因子梯度评分。"""
        c = self._c
        ft = info["features"]
        price = ft["price"]
        atr = ft["atr"]

        bull_score = 0.0
        bear_score = 0.0

        # [1] RSI 梯度
        rsi = ft["rsi"]
        if rsi < c.rsi_extreme:          bull_score += 1.5
        elif rsi < c.rsi_oversold:       bull_score += 1.0
        elif rsi < c.rsi_mild:           bull_score += 0.3

        if rsi > (100 - c.rsi_extreme):  bear_score += 1.5
        elif rsi > c.rsi_overbought:     bear_score += 1.0
        elif rsi > (100 - c.rsi_mild):   bear_score += 0.3

        # [2] KDJ
        k = ft["stoch_k"]
        d = ft["stoch_d"]
        if k < c.kdj_oversold and d < c.kdj_oversold:
            bull_score += 0.8
            if k > d:
                bull_score += 0.4
        if k > c.kdj_overbought and d > c.kdj_overbought:
            bear_score += 0.8
            if k < d:
                bear_score += 0.4

        # [3] MFI
        mfi = ft["mfi"]
        if mfi < c.mfi_oversold:
            bull_score += 0.8
        if mfi > c.mfi_overbought:
            bear_score += 0.8

        # [4] 偏离桶（BB + Z-Score + VWAP，取 top1 + 0.4× top2，封顶 2.5）
        bb_lower = ft["bb_lower"]
        bb_upper = ft["bb_upper"]
        zscore = ft["zscore"]
        vwap = ft["vwap"]

        bb_long_s = 0.0
        bb_short_s = 0.0
        if bb_lower > 0:
            if price <= bb_lower:
                bb_long_s = 1.2
            elif price <= bb_lower * (1 + c.bb_touch_buffer):
                bb_long_s = 0.8
        if bb_upper > 0:
            if price >= bb_upper:
                bb_short_s = 1.2
            elif price >= bb_upper * (1 - c.bb_touch_buffer):
                bb_short_s = 0.8

        z_long_s = 0.0
        z_short_s = 0.0
        if zscore < -c.zscore_extreme:      z_long_s = 1.5
        elif zscore < -c.zscore_threshold:  z_long_s = 1.0
        if zscore > c.zscore_extreme:       z_short_s = 1.5
        elif zscore > c.zscore_threshold:   z_short_s = 1.0

        vwap_long_s = 0.0
        vwap_short_s = 0.0
        if vwap > 0 and atr > 0:
            dev = (vwap - price) / atr
            if dev > c.vwap_atr_mult:
                vwap_long_s = 0.5
            elif dev < -c.vwap_atr_mult:
                vwap_short_s = 0.5

        def _combine(scores: list) -> float:
            s = sorted([x for x in scores if x > 0], reverse=True)
            if not s:
                return 0.0
            top = s[0]
            second = s[1] if len(s) > 1 else 0.0
            return min(c.deviation_bucket_cap, top + c.deviation_secondary_weight * second)

        bull_score += _combine([bb_long_s, z_long_s, vwap_long_s])
        bear_score += _combine([bb_short_s, z_short_s, vwap_short_s])

        # [5] Funding 偏差
        if c.funding_filter_enabled:
            fval = ft["funding"]
            if abs(fval) >= c.funding_extreme_abs:
                if fval > 0:
                    bear_score += 0.3
                elif fval < 0:
                    bull_score += 0.3

        # Regime 惩罚
        bull_score = max(0.0, bull_score - regime_penalty)
        bear_score = max(0.0, bear_score - regime_penalty)

        return {
            "bull": bull_score, "bear": bear_score,
            "rsi": rsi, "k": k, "d": d, "mfi": mfi, "zscore": zscore,
        }

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def __call__(
        self, df: pd.DataFrame, info: Dict,
    ) -> Tuple[pd.DataFrame, Dict]:
        c = self._c
        ft = info["features"]
        ts: TradeState = info["trade_state"]

        info["signal_direction"] = 0
        info["chosen_score"] = 0.0

        # ══════════════════════════════════════════════════════════════
        # § 1  守卫检查
        # ══════════════════════════════════════════════════════════════

        if ts.last_side != 0:
            return df, info

        if ts.reverse_cooldown > 0:
            ts.reverse_cooldown -= 1
            return df, info

        if ts.cooldown_remaining > 0:
            ts.cooldown_remaining -= 1
            return df, info

        if ts.consecutive_losses >= c.max_consecutive_losses:
            ts.loss_pause_bars += 1
            if ts.loss_pause_bars >= c.loss_pause_timeout:
                ts.consecutive_losses = 0
                ts.loss_pause_bars = 0
            else:
                return df, info

        price = ft["price"]
        atr = ft["atr"]
        if price <= 0:
            return df, info

        # ══════════════════════════════════════════════════════════════
        # § 2  Regime 检查
        # ══════════════════════════════════════════════════════════════

        regime = self._check_regime(info)
        if not regime["pass"]:
            ts.last_diag = {
                "entry_gate_pass": False,
                "reason": regime["reason"],
                "details": regime["details"],
            }
            return df, info

        # ══════════════════════════════════════════════════════════════
        # § 3  趋势过滤
        # ══════════════════════════════════════════════════════════════

        trend = self._check_trend_filter(info)
        if trend.get("block_long") and trend.get("block_short"):
            ts.last_diag = {
                "entry_gate_pass": False,
                "reason": "trend_strong_blocked",
                "details": trend,
            }
            return df, info

        # ══════════════════════════════════════════════════════════════
        # § 4  ATR 分位数门控
        # ══════════════════════════════════════════════════════════════

        if c.atr_percentile_enabled:
            atr_pct = ft["atr_pct"]
            if not np.isnan(atr_pct) and atr_pct > c.atr_percentile_max_entry:
                ts.last_diag = {
                    "entry_gate_pass": False,
                    "reason": "atr_percentile_too_high",
                    "val": atr_pct,
                }
                return df, info

        # ══════════════════════════════════════════════════════════════
        # § 5  多因子评分
        # ══════════════════════════════════════════════════════════════

        penalty = 0.0
        if regime["details"].get("adx_penalty"):
            penalty += 0.5
        if regime["details"].get("er_penalty"):
            penalty += 0.4

        scores = self._calculate_scores(df, info, penalty)
        bull_score = scores["bull"]
        bear_score = scores["bear"]

        # ══════════════════════════════════════════════════════════════
        # § 6  硬条件检查
        # ══════════════════════════════════════════════════════════════

        bb_upper = ft["bb_upper"]
        bb_lower = ft["bb_lower"]

        touch_long = (bb_lower > 0) and (price <= bb_lower * (1 + c.bb_touch_buffer))
        touch_short = (bb_upper > 0) and (price >= bb_upper * (1 - c.bb_touch_buffer))

        extreme_long = (
            (scores["rsi"] <= c.rsi_oversold)
            or (scores["zscore"] <= -c.zscore_threshold)
            or (scores["k"] < c.kdj_oversold)
        )
        extreme_short = (
            (scores["rsi"] >= c.rsi_overbought)
            or (scores["zscore"] >= c.zscore_threshold)
            or (scores["k"] > c.kdj_overbought)
        )

        allowed_long = (
            (not c.require_bb_outer_touch or touch_long)
            and (not c.require_extreme_confirm or extreme_long)
        )
        allowed_short = (
            (not c.require_bb_outer_touch or touch_short)
            and (not c.require_extreme_confirm or extreme_short)
        )

        # ══════════════════════════════════════════════════════════════
        # § 7  方向决策
        # ══════════════════════════════════════════════════════════════

        if bull_score >= c.min_entry_score and bear_score >= c.min_entry_score:
            if abs(bull_score - bear_score) < 0.5:
                ts.last_diag = {"reason": "ambiguous", "bull": bull_score, "bear": bear_score}
                return df, info

        final_signal = 0
        chosen_score = 0.0

        if bull_score >= c.min_entry_score and bull_score > bear_score:
            if not trend.get("block_long") and allowed_long:
                final_signal = 1
                chosen_score = bull_score
        elif bear_score >= c.min_entry_score and bear_score > bull_score:
            if not trend.get("block_short") and allowed_short:
                final_signal = -1
                chosen_score = bear_score

        if final_signal == 0:
            ts.last_diag = {"reason": "score_low_or_blocked", "bull": bull_score, "bear": bear_score}
            return df, info

        # ══════════════════════════════════════════════════════════════
        # § 8  盈利性检查
        # ══════════════════════════════════════════════════════════════

        fee_est = 2.0 * c.fee_rate * price
        bb_middle = ft["bb_middle"]
        vwap = ft["vwap"]
        mean_price = vwap if vwap > 0 else bb_middle
        expected_profit_dist = abs(price - mean_price) if mean_price > 0 else 0.0

        if c.fee_guard_enabled and expected_profit_dist < fee_est * c.fee_safety_mult:
            ts.last_diag = {
                "reason": "fee_safety_fail_real",
                "exp_dist": expected_profit_dist,
                "fee_est": fee_est,
                "mult": c.fee_safety_mult,
            }
            return df, info

        if c.min_tp_atr_mult > 0 and expected_profit_dist < atr * c.min_tp_atr_mult:
            ts.last_diag = {
                "reason": "tp_atr_fail_real",
                "exp_dist": expected_profit_dist,
                "atr": atr,
                "min_mult": c.min_tp_atr_mult,
            }
            return df, info

        # ── 成功 ──
        info["signal_direction"] = final_signal
        info["chosen_score"] = chosen_score
        ts.last_entry_score = chosen_score
        ts.last_diag = {"entry_gate_pass": True, "score": chosen_score, "side": final_signal}
        return df, info


# ===========================================================================
# ExitController — 出场信号 + 风控状态管理
# ===========================================================================

class ExitController:
    """
    持仓管理 & 风控:
      1. 最大持仓时间 → 强制平仓
      2. 保本机制（浮盈 > breakeven_atr × ATR → 标记；回到开仓价 → 平仓）
      3. 动态均值止盈（触碰 BB 中轨 / VWAP → 平仓）
      4. 动量衰竭（DI/ADX）→ 平仓
      5. 盈亏推断 → 冷却/暂停状态更新
    """

    def __init__(self, cfg: Any) -> None:
        self._c = cfg

    def update_risk_state(
        self, ts: TradeState, current_pos: float, price: float,
    ) -> None:
        """推断盈亏并更新风控状态。"""
        c = self._c
        old_exited = (
            (ts.last_position != 0.0 and current_pos == 0.0)
            or (ts.last_position > 0 and current_pos < 0)
            or (ts.last_position < 0 and current_pos > 0)
        )

        if old_exited and ts.entry_price is not None:
            if ts.last_position > 0:
                pnl = (price - ts.entry_price) * abs(ts.last_position)
            else:
                pnl = (ts.entry_price - price) * abs(ts.last_position)

            if pnl < 0:
                ts.consecutive_losses += 1
                ts.cooldown_remaining = c.cooldown_bars
                ts.loss_pause_bars = 0
            else:
                ts.consecutive_losses = 0
                ts.loss_pause_bars = 0
                ts.cooldown_remaining = c.win_cooldown_bars

        if current_pos != 0 and ts.last_position == 0:
            ts.best_price = price
            ts.breakeven_triggered = False
        elif old_exited:
            ts.best_price = None
            ts.entry_price = None
            ts.last_entry_score = 0.0
            ts.breakeven_triggered = False

        ts.tpsl_closed = old_exited and ts.last_position != 0.0
        ts.last_position = current_pos

    def __call__(self, df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, Dict]:
        c = self._c
        ts: TradeState = info["trade_state"]
        curr_side = int(info["curr_side"])

        info["exit_signal"] = None

        if curr_side == 0:
            return df, info

        ft = info["features"]
        price = ft["price"]
        atr = ft["atr"]

        # ── 1. 最大持仓 → 强制平仓 ──────────────────────────
        if ts.hold_bars >= c.max_hold_bars:
            info["exit_signal"] = float(-curr_side)
            return df, info

        # ── 2. 保本机制 ─────────────────────────────────────
        if ts.entry_price and atr > 0:
            if curr_side == 1:
                pnl_atr = (price - ts.entry_price) / atr
            else:
                pnl_atr = (ts.entry_price - price) / atr

            if pnl_atr > c.breakeven_atr:
                ts.breakeven_triggered = True

            if ts.breakeven_triggered:
                buffer = atr * 0.1
                if curr_side == 1 and price < ts.entry_price + buffer:
                    info["exit_signal"] = float(-curr_side)
                    return df, info
                if curr_side == -1 and price > ts.entry_price - buffer:
                    info["exit_signal"] = float(-curr_side)
                    return df, info

        # ── 3. 均值止盈（触碰 BB 中轨 / VWAP） ───────────────
        if c.exit_on_mean_touch:
            bb_middle = ft["bb_middle"]
            vwap = ft["vwap"]
            if curr_side == 1:
                if (bb_middle > 0 and price >= bb_middle) or (vwap > 0 and price >= vwap):
                    info["exit_signal"] = float(-curr_side)
                    return df, info
            elif curr_side == -1:
                if (bb_middle > 0 and price <= bb_middle) or (vwap > 0 and price <= vwap):
                    info["exit_signal"] = float(-curr_side)
                    return df, info

        # ── 4. 动量衰竭（DI/ADX）— 需持仓 ≥ min_hold_bars ──
        if ts.hold_bars >= c.min_hold_bars:
            plus_di = ft["plus_di"]
            minus_di = ft["minus_di"]
            adx = ft["adx"]

            if plus_di > 0 and minus_di > 0:
                if curr_side == 1 and (minus_di - plus_di) > c.di_adverse_gap:
                    info["exit_signal"] = float(-curr_side)
                    return df, info
                if curr_side == -1 and (plus_di - minus_di) > c.di_adverse_gap:
                    info["exit_signal"] = float(-curr_side)
                    return df, info
            if adx > c.adx_force_exit:
                info["exit_signal"] = float(-curr_side)
                return df, info

        return df, info


# ===========================================================================
# PositionSizer — 仓位计算（含评分动态调整）
# ===========================================================================

class PositionSizer:
    """按评分动态调整 risk_per_trade + notional_pct 缩放。"""

    def __init__(self, cfg: Any) -> None:
        self._c = cfg

    def _score_adjusted_risk(self, ts: TradeState) -> float:
        """根据入场评分动态调整单笔风险比例。"""
        c = self._c
        if ts.last_entry_score <= c.min_entry_score or c.score_sizing_boost <= 0:
            return c.risk_per_trade

        score_excess = min(
            (ts.last_entry_score - c.min_entry_score) / c.min_entry_score,
            1.0,
        )
        boosted = c.risk_per_trade * (1.0 + score_excess * c.score_sizing_boost)
        return min(boosted, c.max_risk_per_trade)

    def __call__(self, df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, Dict]:
        c = self._c
        ts: TradeState = info["trade_state"]
        equity = float(info.get("equity") or 0.0)
        price = info["features"]["price"]
        atr = info["features"]["atr"]
        sig = int(info.get("signal_direction", 0))

        actual_risk = self._score_adjusted_risk(ts) if sig != 0 else c.risk_per_trade

        eff_min_notional = float(c.min_notional_pct)
        eff_max_notional = float(c.max_notional_pct)

        if sig != 0 and c.notional_score_scaling_enabled and c.min_entry_score > 0:
            strength = min(
                (ts.last_entry_score - c.min_entry_score) / c.min_entry_score, 1.0,
            )
            if abs(eff_min_notional - eff_max_notional) < 1e-9:
                floor = eff_max_notional * c.notional_floor_mult_when_fixed
                cap = floor + (eff_max_notional - floor) * strength
                eff_min_notional = min(floor, cap)
                eff_max_notional = max(eff_min_notional, cap)
            else:
                cap = eff_min_notional + (eff_max_notional - eff_min_notional) * strength
                eff_max_notional = max(eff_min_notional, cap)

        sizing = calc_position_size(
            equity=equity, price=price, atr=atr,
            risk_pct=actual_risk,
            atr_sl_mult=c.atr_sl_multiplier,
            min_leverage=c.min_leverage, max_leverage=c.max_leverage,
            min_notional_pct=eff_min_notional, max_notional_pct=eff_max_notional,
            min_quantity=c.min_quantity, quantity_precision=c.quantity_precision,
        )
        info["sizing"] = sizing
        return df, info


# ===========================================================================
# MeanReversionStrategy — 配置 + 主流程
# ===========================================================================

class MeanReversionStrategy(Strategy):
    """均值回归策略 — 多因子梯度评分 + 动态均值止盈 + 自适应 Regime。"""

    # ── 策略元信息 ────────────────────────────────────────────
    name: str = Field(default="mean_reversion", description="策略名称")
    description: str = Field(
        default="均值回归策略 v2 — 动态均值止盈、保本机制、自适应Regime",
    )
    factor_names: List[str] = Field(
        default=[
            "rsi", "bb", "atr", "adx", "vwap", "zscore_price", "er", "chop", "hurst", "ema",
            "atr_percentile", "funding_percentile", "kdj", "mfi",
        ],
    )

    # ── 交易参数 ──────────────────────────────────────────────
    symbol: str = Field(default="BTCUSDT")
    quantity_precision: int = Field(default=3)
    min_quantity: float = Field(default=0.02)

    # ── 仓位管理 ──────────────────────────────────────────────
    risk_per_trade: float = Field(default=0.015, description="基础每笔风险 1.5%")
    max_risk_per_trade: float = Field(default=0.025, description="单笔风险上限 2.5%")
    score_sizing_boost: float = Field(default=0.5, description="信号强度仓位加成系数")
    atr_sl_multiplier: float = Field(default=2.5, description="止损 = 2.5× ATR")
    min_leverage: int = Field(default=5)
    max_leverage: int = Field(default=10)
    min_notional_pct: float = Field(default=1.0)
    max_notional_pct: float = Field(default=1.0)

    # ── 手续费 ────────────────────────────────────────────────
    fee_rate: float = Field(default=DEFAULT_TAKER_FEE)

    # ── 市场状态过滤 (Regime) ────────────────────────────────
    adx_max_threshold: float = Field(default=25.0, description="ADX 上限")
    adx_strict_threshold: float = Field(default=25.0, description="ADX 严格阈值")

    # ── 动态止盈与保本 ────────────────────────────────────────
    exit_on_mean_touch: bool = Field(default=True, description="触碰均值(BB中轨/VWAP)时平仓")
    breakeven_atr: float = Field(default=1.2, description="保本触发: 浮盈 > 1.2 ATR")

    # ── 退出逻辑 ──────────────────────────────────────────────
    adx_force_exit: float = Field(default=30.0, description="ADX 强退阈值")
    di_adverse_gap: float = Field(default=20.0, description="DI 不利差值阈值")
    bb_width_max: float = Field(default=0.12, description="BB 带宽上限")
    bb_width_min: float = Field(default=0.005, description="BB 带宽下限")

    # ── RSI ───────────────────────────────────────────────────
    rsi_period: int = Field(default=14)
    rsi_extreme: float = Field(default=20.0)
    rsi_oversold: float = Field(default=32.0)
    rsi_overbought: float = Field(default=68.0)
    rsi_mild: float = Field(default=40.0)

    # ── KDJ & MFI ────────────────────────────────────────────
    kdj_period: int = Field(default=14)
    kdj_oversold: float = Field(default=25.0)
    kdj_overbought: float = Field(default=75.0)
    mfi_period: int = Field(default=14)
    mfi_oversold: float = Field(default=20.0)
    mfi_overbought: float = Field(default=80.0)

    # ── BB & Z-Score & VWAP ──────────────────────────────────
    bb_period: int = Field(default=20)
    bb_touch_buffer: float = Field(default=0.004)
    zscore_period: int = Field(default=50)
    zscore_threshold: float = Field(default=1.8)
    zscore_extreme: float = Field(default=2.5)
    vwap_period: int = Field(default=20)
    vwap_atr_mult: float = Field(default=1.5)

    # ── ATR / ADX ────────────────────────────────────────────
    atr_period: int = Field(default=14)
    adx_period: int = Field(default=14)

    # ── 风控与冷却 ───────────────────────────────────────────
    min_entry_score: float = Field(default=2.2, description="最低入场分数")
    min_hold_bars: int = Field(default=15)
    max_hold_bars: int = Field(default=180)
    cooldown_bars: int = Field(default=15)
    win_cooldown_bars: int = Field(default=15)
    max_consecutive_losses: int = Field(default=3)
    loss_pause_timeout: int = Field(default=120)

    # ── 入场硬条件 ───────────────────────────────────────────
    require_bb_outer_touch: bool = Field(default=True, description="要求触碰 BB 外轨")
    require_extreme_confirm: bool = Field(default=True, description="要求极值确认")

    # ── 反向处理 ─────────────────────────────────────────────
    flip_mode: str = Field(default="close_then_wait")
    reverse_cooldown_bars: int = Field(default=15)

    # ── Regime 因子 ──────────────────────────────────────────
    er_period: int = Field(default=60)
    er_max_entry: float = Field(default=0.3)
    chop_period: int = Field(default=14)
    chop_min_entry: float = Field(default=50.0)
    hurst_window: int = Field(default=120)
    hurst_max_entry: float = Field(default=0.5)

    # ── ROI TP/SL ────────────────────────────────────────────
    use_roi_tpsl: bool = Field(default=True)
    tp_roi_pct: float = Field(default=20.0, description="止盈 ROI（保底）")
    sl_roi_pct: float = Field(default=10.0, description="止损 ROI（保底）")

    # ── 手续费盈利性门控 ─────────────────────────────────────
    fee_guard_enabled: bool = Field(default=True)
    fee_safety_mult: float = Field(default=4.0)
    min_tp_atr_mult: float = Field(default=1.0)

    # ── 趋势/方向过滤 ────────────────────────────────────────
    trend_filter_enabled: bool = Field(default=True)
    ema_fast_period: int = Field(default=50)
    ema_slow_period: int = Field(default=200)
    ema_slope_window: int = Field(default=20)
    ema_slope_threshold: float = Field(default=0.001)
    trend_block_opposite: bool = Field(default=True)
    trend_block_strong: bool = Field(default=True)
    strong_trend_slope_mult: float = Field(default=2.5)

    atr_percentile_enabled: bool = Field(default=True)
    atr_percentile_window: int = Field(default=200)
    atr_percentile_max_entry: float = Field(default=0.50)

    funding_filter_enabled: bool = Field(default=True)
    funding_extreme_abs: float = Field(default=0.0008)

    deviation_bucket_cap: float = Field(default=2.5)
    deviation_secondary_weight: float = Field(default=0.4)

    notional_score_scaling_enabled: bool = Field(default=True)
    notional_floor_mult_when_fixed: float = Field(default=0.30)

    # ==================================================================
    # 初始化
    # ==================================================================

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._trade_state = TradeState()
        self._feature_engine = FeatureEngine(self)
        self._enter_controller = EnterController(self)
        self._exit_controller = ExitController(self)
        self._position_sizer = PositionSizer(self)

    # ==================================================================
    # __call__ — 主流程
    # ==================================================================

    async def __call__(
        self,
        df: pd.DataFrame,
        current_pos: float = 0.0,
        equity: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """均值回归策略主入口。"""

        # ── Step 0: 因子计算 ───────────────────────────────────
        await factor_manager("rsi", df, periods=[self.rsi_period])
        await factor_manager("bb", df, periods=[self.bb_period])
        await factor_manager("atr", df, periods=[self.atr_period])
        await factor_manager("adx", df, periods=[self.adx_period])
        await factor_manager("vwap", df, periods=[self.vwap_period])
        await factor_manager("ema", df, periods=[self.ema_fast_period, self.ema_slow_period])
        await factor_manager("kdj", df, periods=[self.kdj_period])
        await factor_manager("mfi", df, periods=[self.mfi_period])
        await factor_manager("atr_percentile", df, periods=[self.atr_percentile_window])
        await factor_manager("funding_percentile", df, periods=[500])
        await factor_manager("er", df, periods=[self.er_period])
        await factor_manager("chop", df, periods=[self.chop_period])
        await factor_manager("hurst", df, windows=[self.hurst_window])

        zs_col = f"zscore_{self.zscore_period}"
        if zs_col not in df.columns:
            roll_mean = df[PRICE_COL].rolling(self.zscore_period).mean()
            roll_std = df[PRICE_COL].rolling(self.zscore_period).std()
            df[zs_col] = (df[PRICE_COL] - roll_mean) / roll_std.replace(0, np.nan)

        # ── Step 1: 特征提取 ──────────────────────────────────
        ts = self._trade_state
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
        info: Dict[str, Any] = {
            "current_pos": current_pos,
            "equity": equity,
            "curr_side": curr_side,
            "trade_state": ts,
        }

        df, info = self._feature_engine(df, info)
        ft = info["features"]
        price = ft["price"]
        atr = ft["atr"]

        # ── Step 2: 风控状态更新 ──────────────────────────────
        self._exit_controller.update_risk_state(ts, current_pos, price)

        # ── Step 3: 更新 hold_bars / last_side ────────────────
        if curr_side != 0:
            if ts.last_side == curr_side:
                ts.hold_bars += 1
            else:
                ts.hold_bars = 1
                ts.last_side = curr_side
        else:
            ts.hold_bars = 0
            ts.last_side = 0

        # ── Step 4: 出场检查（持仓中）────────────────────────
        df, info = self._exit_controller(df, info)
        exit_sig = info.get("exit_signal")

        # ── Step 5: 入场信号（无持仓时）──────────────────────
        df, info = self._enter_controller(df, info)
        new_sig = int(info.get("signal_direction", 0))

        # ── Step 6: 仓位计算 ─────────────────────────────────
        df, info = self._position_sizer(df, info)
        sizing = info["sizing"]

        # ── Step 7: 信号合并 + 构建订单 ──────────────────────
        signal_name = "CLOSE" if ts.tpsl_closed else "HOLD"
        order: Optional[Dict] = None
        result: Optional[Dict] = None

        sig_int_for_order = 0

        if curr_side != 0 and exit_sig is not None:
            sig_int_for_order = int(np.sign(exit_sig))
            result = build_order(
                symbol=self.symbol, signal=sig_int_for_order, current_pos=current_pos,
                sizing=sizing, price=price, atr=atr,
                atr_sl_multiplier=self.atr_sl_multiplier, atr_tp_multiplier=0.0,
                use_trailing_stop=False,
                flip_mode=self.flip_mode,
                tp_sl_mode="PRICE",
                fee_rate=self.fee_rate,
                quantity_precision=self.quantity_precision,
            )
            signal_name = result["signal"]
            order = result["order"]

        elif curr_side == 0 and new_sig != 0:
            sig_int_for_order = new_sig
            result = build_order(
                symbol=self.symbol, signal=sig_int_for_order, current_pos=current_pos,
                sizing=sizing, price=price, atr=atr,
                atr_sl_multiplier=self.atr_sl_multiplier, atr_tp_multiplier=0.0,
                use_trailing_stop=False,
                flip_mode=self.flip_mode,
                tp_sl_mode=("ROI" if self.use_roi_tpsl else "PRICE"),
                take_profit_roi_pct=(self.tp_roi_pct if self.use_roi_tpsl else None),
                stop_loss_roi_pct=(self.sl_roi_pct if self.use_roi_tpsl else None),
                fee_rate=self.fee_rate,
                quantity_precision=self.quantity_precision,
            )
            signal_name = result["signal"]
            order = result["order"]

            if signal_name in ("LONG", "SHORT"):
                ts.entry_price = price
                ts.best_price = price

        if signal_name == "CLOSE" and self.flip_mode == "close_then_wait":
            ts.reverse_cooldown = int(self.reverse_cooldown_bars)

        # ── Step 8: Trailing Stop Info ────────────────────────
        effective_pos = current_pos
        if signal_name == "LONG":
            effective_pos = sizing["quantity"]
        elif signal_name == "SHORT":
            effective_pos = -sizing["quantity"]

        trailing_info, ts.best_price = compute_trailing_stop(
            effective_pos, price, atr, self.atr_sl_multiplier,
            ts.entry_price, ts.best_price,
        )

        # ── Step 9: 日志字段 ─────────────────────────────────
        factors_json = build_factors_json(df, [
            f"rsi_{self.rsi_period}",
            f"adx_{self.adx_period}", f"plus_di_{self.adx_period}", f"minus_di_{self.adx_period}",
            f"atr_{self.atr_period}",
            f"bb_upper_{self.bb_period}", f"bb_lower_{self.bb_period}", f"bb_middle_{self.bb_period}",
            f"vwap_{self.vwap_period}",
            f"ema_{self.ema_fast_period}", f"ema_{self.ema_slow_period}",
            f"stoch_k_{self.kdj_period}", f"stoch_d_{self.kdj_period}",
            f"mfi_{self.mfi_period}",
            f"er_{self.er_period}",
            f"chop_{self.chop_period}",
            f"hurst_{self.hurst_window}",
            f"zscore_{self.zscore_period}",
            f"atr_pct_{self.atr_period}_{self.atr_percentile_window}",
            "funding_percentile_500",
        ], extra={
            "hold_bars": ts.hold_bars,
            "consecutive_losses": ts.consecutive_losses,
            "cooldown_remaining": ts.cooldown_remaining,
            "entry_score": ts.last_entry_score,
        })

        meta = result.get("meta", {}) if result is not None else {}
        meta["mean_reversion"] = {
            "entry_score": round(ts.last_entry_score, 2),
            "diag": ts.last_diag,
            "breakeven": ts.breakeven_triggered,
        }
        meta["factors_json"] = factors_json

        return {
            "signal": signal_name,
            "order": order,
            "leverage": sizing["leverage"],
            "sizing": sizing,
            "atr": atr,
            "meta": meta,
            "factors_json": factors_json,
            "risk": {
                "cooldown": ts.cooldown_remaining,
                "losses": ts.consecutive_losses,
                "hold": ts.hold_bars,
            },
            "trailing_stop": trailing_info,
        }
