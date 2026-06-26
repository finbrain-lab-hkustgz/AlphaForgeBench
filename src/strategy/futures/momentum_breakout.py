"""
动量突破策略 (Momentum Breakout) — v5 简化版
适用市场：Binance USDT-M 永续合约（默认 BTCUSDT）

═══════════════════════════════════════════════════════════════════════
整体流水线
═══════════════════════════════════════════════════════════════════════
  FeatureEngine → EnterController → ExitController → PositionSizer → OrderRouter

  · FeatureEngine    提取 BB / ADX / ATR / EMA / RSI / TTM Squeeze / BBWidth / ROC 等指标
  · EnterController  前置门控（ATR 活跃度 + TTM Squeeze + BBWidth 分位压缩 + BB 突破 + 放量）+
                     追空/追多末端保护（RSI 耗竭 + 低位/高位禁入 + 反弹检测）+
                     4 因子评分系统（≥ 2.5 分方可入场）+ EMA 趋势过滤
  · ExitController   保证金 ROI% 止盈止损 + ATR 波动率飙升退出 + 最大持仓时间
  · PositionSizer    保证金百分比仓位计算: notional = equity × margin_pct × leverage
  · OrderRouter      信号合并、附加 TP/SL 条件单、构建最终订单

═══════════════════════════════════════════════════════════════════════
入场逻辑
═══════════════════════════════════════════════════════════════════════

  前置门控（必须全部通过）:
    1. ATR ≥ ATR_MA × 0.85 且 ATR 正在扩张（squeeze 释放阶段）
    2. TTM Squeeze 在最近 5 bar 内发生过（波动率压缩期）
    3. BBWidth 落在近 N bar 的低分位（例如 20% 分位）→ 只做“真压缩”
    4. BB 外轨突破（price > BB_upper 或 price < BB_lower）
    5. 成交量 ≥ 1.5× 均值（无量突破不入场）

  评分（需 ≥ 2.5 分，满分 3.0）:
    - BB 突破确认                             → 1.0 分
    - 成交量 > 1.5× 均值                      → 0.5 分
    - ROC 动量方向一致且增强                  → 1.0 分
    - ADX > 25 + DI 方向一致                  → 0.5 分

  末端保护（防止在极端行情末端追单）:
    - 追空：RSI 极端超卖 + 宏观斜率不足 → 禁止追空（趋势即将反转）
    - 追空：近期超卖后 RSI 急反弹 + 1m 回升 → 禁止追空（反弹延续）
    - 追空：近期出现超卖 + 价格处于 2h 区间底部 → 禁止做空（低位禁入）
    - 追多：RSI 极端超买 + 宏观斜率不足 → 禁止追多
    - 追多：近期出现超买 + 价格处于 2h 区间顶部 → 禁止做多（高位禁入）

  方向过滤:
    - EMA200 价格位置过滤（做多需 price > EMA200）
    - EMA50/200 双均线趋势一致性过滤

═══════════════════════════════════════════════════════════════════════
出场逻辑
═══════════════════════════════════════════════════════════════════════

  三条退出规则（任一触发即 CLOSE）:
    1. TP/SL — 保证金 ROI% 硬止盈止损（默认 TP=20%，SL=10%）
    2. ATR 波动率飙升 — 当前 ATR > 入场 ATR × 3.5，市场失控强制退出
    3. 最大持仓 720 bar（12 小时）→ 强制平仓

═══════════════════════════════════════════════════════════════════════
风控
═══════════════════════════════════════════════════════════════════════
  · 亏损冷却 60 bar / 盈利冷却 30 bar
  · 连续 2 次亏损 → 暂停 200 bar（3.3 小时）
  · close_then_wait 反向冷却 60 bar
  · 双向交易（long_only 默认关闭）
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
    PRICE_COL, VOL_COL, DEFAULT_TAKER_FEE, safe_float,
    build_factors_json,
)


# ===========================================================================
# MTF helper — 高时间框架 EMA 方向（参考 adaptive_trend_fusion）
# ===========================================================================

def _compute_mtf_ema_dir(df: pd.DataFrame, freq: str, period: int) -> int:
    """
    将 1m df 重采样到 freq（如 '5min'/'15min'），计算 EMA，返回最后一根的方向。
    +1 = EMA 上升（多头），-1 = 下降（空头），0 = 数据不足 / 解析失败
    """
    try:
        ts_col = "timestamp" if "timestamp" in df.columns else None
        if ts_col:
            s = df[ts_col]
            try:
                if hasattr(s, "dtype") and not str(s.dtype).startswith("int") and not str(s.dtype).startswith("float"):
                    idx = pd.to_datetime(
                        s,
                        format="%Y-%m-%d %H:%M:%S",
                        utc=True,
                        errors="coerce",
                    )
                else:
                    idx = pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
            except Exception:
                idx = pd.to_datetime(s, utc=True, errors="coerce")
        elif hasattr(df.index, "dtype") and str(df.index.dtype).startswith("datetime"):
            idx = df.index
        else:
            return 0

        close = df["close"].astype(float).copy()
        close.index = idx
        resampled = close.resample(freq).last().dropna()
        if len(resampled) < period + 2:
            return 0
        ema = resampled.ewm(span=period, adjust=False).mean()
        return 1 if float(ema.iloc[-1]) > float(ema.iloc[-2]) else -1
    except Exception:
        return 0


# ===========================================================================
# TradeState — 交易状态跟踪
# ===========================================================================

@dataclass
class TradeState:
    """跟踪当前交易的实时状态。"""
    hold_bars:         int             = 0
    last_side:         int             = 0
    cooldown_remaining: int            = 0
    consecutive_losses: int            = 0
    loss_pause_bars:   int             = 0
    last_position:     float           = 0.0
    entry_price:       Optional[float] = None
    entry_leverage:    int             = 1
    entry_atr:         float           = 0.0
    entry_regime:      str             = "n/a"   # "trend" | "range" | "transition" | "n/a"
    reverse_cooldown:  int             = 0
    pending_fill:      bool            = False
    tpsl_closed:       bool            = False

    def update_from_fill(self, fill_price: float, leverage: int) -> None:
        """成交回报到达后由 caller 调用。"""
        self.entry_price = float(fill_price)
        self.entry_leverage = max(1, int(leverage))
        self.pending_fill = False


# ===========================================================================
# FeatureEngine — 指标提取
# ===========================================================================

class FeatureEngine:
    """从 DataFrame 提取 BB / MACD / ADX / ATR / EMA / Volume，
    并计算 Keltner Channel Squeeze 状态。"""

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

        def _prev(col: str, default: float = float("nan")) -> float:
            if col not in df.columns or len(df) < 2:
                return default
            try:
                v = df[col].iloc[-2]
                return float(v) if v is not None and not pd.isna(v) else default
            except (TypeError, ValueError, IndexError):
                return default

        price = float(df[PRICE_COL].iloc[-1])
        atr = _last(f"atr_{c.atr_period}")
        ema_slow = _last(f"ema_{c.ema_trend_period}", 0.0)
        dist_atr = float("nan")
        try:
            _atr = float(atr)
            if (not np.isnan(_atr)) and _atr > 1e-12 and ema_slow > 0:
                dist_atr = abs(float(price) - float(ema_slow)) / _atr
        except Exception:
            dist_atr = float("nan")

        # TTM Squeeze + BBWidth + ROC
        squeeze_on_col = f"squeeze_on_{c.squeeze_period}"
        squeeze_mom_col = f"squeeze_mom_{c.squeeze_period}"
        bb_width_col = f"bb_width_{c.bb_width_period}"
        roc_col = f"roc_{c.roc_window}"

        # Supertrend direction (stable confirmation, like adaptive_trend_fusion)
        stable_st_dir: float = float("nan")
        try:
            st_tag = f"{int(c.supertrend_period)}_{int(float(c.supertrend_mult) * 10)}"
            st_col = f"supertrend_dir_{st_tag}"
            confirm = max(1, int(getattr(c, "supertrend_confirm_bars", 3)))
            if st_col in df.columns and len(df) >= confirm:
                recent = df[st_col].iloc[-confirm:]
                if len(recent) == confirm and recent.notna().all():
                    vals = recent.astype(float).values
                    if (vals == vals[0]).all() and vals[0] in (1.0, -1.0):
                        stable_st_dir = float(vals[0])
        except Exception:
            stable_st_dir = float("nan")

        had_recent_squeeze = False
        if squeeze_on_col in df.columns and len(df) >= c.squeeze_recent_bars + 1:
            squeeze_on_s = df[squeeze_on_col].astype(float)
            recent_n = min(int(c.squeeze_recent_bars), len(squeeze_on_s))
            had_recent_squeeze = bool((squeeze_on_s.iloc[-recent_n:] > 0.0).any())

        roc_ratio = _last(roc_col, float("nan"))
        roc_ratio_prev = _prev(roc_col, float("nan"))
        eps = 1e-12
        roc_mom = (1.0 / (roc_ratio + eps) - 1.0) if not np.isnan(roc_ratio) else 0.0
        roc_mom_prev = (1.0 / (roc_ratio_prev + eps) - 1.0) if not np.isnan(roc_ratio_prev) else 0.0

        info["features"] = {
            "price":              price,
            "atr":                atr,
            "atr_prev":           _prev(f"atr_{c.atr_period}"),
            "bb_upper":           _last(f"bb_upper_{c.bb_period}"),
            "bb_lower":           _last(f"bb_lower_{c.bb_period}"),
            "adx":                _last(f"adx_{c.adx_period}", 15.0),
            "plus_di":            _last(f"plus_di_{c.adx_period}", 0.0),
            "minus_di":           _last(f"minus_di_{c.adx_period}", 0.0),
            "ema_slow":           ema_slow,
            "ema_fast":           _last(f"ema_{c.ema_fast_period}", 0.0),
            "dist_atr":           dist_atr,
            "supertrend_dir":     stable_st_dir,
            "had_recent_squeeze": had_recent_squeeze,
            "squeeze_on":         _last(squeeze_on_col, 0.0),
            "squeeze_mom":        _last(squeeze_mom_col, 0.0),
            "bb_width":           _last(bb_width_col, float("nan")),
            "roc_mom":            float(roc_mom),
            "roc_mom_prev":       float(roc_mom_prev),
            "rsi":                _last(f"rsi_{c.rsi_period}", float("nan")),
        }
        return df, info


# ===========================================================================
# EnterController — 入场信号
# ===========================================================================

class EnterController:
    """
    动量突破入场系统。

    流程：
      § 1    守卫检查     — 冷却 / 暂停 / 已持仓 / 反向冷却
      § 2    门控检查     — ATR 活跃度 / TTM Squeeze / BBWidth 分位压缩 / BB 突破 / 放量
      § 2.5  末端保护     — RSI 超卖/超买 + 低位/高位 + 反弹检测 → 防止在极端行情末端追单
      § 3    多因子评分   — 4 项因子累加（满分 3.0）
      § 4    EMA 趋势过滤 + 方向确认
    """

    def __init__(self, cfg: Any) -> None:
        self._c = cfg

    def __call__(self, df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, Dict]:
        c = self._c
        ft = info["features"]
        ts: TradeState = info["trade_state"]

        info["signal_direction"] = 0
        info["entry_score"] = None
        info["effective_min_entry_score"] = None
        info["atr_gate_ok"] = None

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

        # ── Regime 过滤（参考 adaptive_trend_fusion：transition 期跳过）──
        regime = str(info.get("regime") or "n/a")
        if c.regime_skip_transition and regime == "transition":
            return df, info

        price = ft["price"]
        atr = ft["atr"]
        if price <= 0 or np.isnan(atr):
            return df, info

        # ── 过度延伸过滤（宁可少做也不追高/追低）：|price-EMA200|/ATR 过大直接不入场 ──
        dist_atr = float(ft.get("dist_atr", float("nan")))
        info["dist_atr"] = dist_atr
        if c.entry_dist_filter_enabled and float(c.max_entry_dist_atr) > 0:
            if np.isnan(dist_atr) or dist_atr > float(c.max_entry_dist_atr):
                return df, info

        # ── ATR 最低门槛（参考 adaptive_trend_fusion：手续费覆盖/波动充裕）──
        if c.atr_min_gate_enabled and float(c.atr_min_usdt) > 0:
            if float(atr) < float(c.atr_min_usdt):
                return df, info

        # ══════════════════════════════════════════════════════════════
        # § 2  门控检查
        # ══════════════════════════════════════════════════════════════

        # ── ATR 活跃度（ATR expanding + above floor）──────────────
        atr_gate_ok = True
        if c.atr_filter_enabled and len(df) >= c.atr_filter_period + 1:
            atr_series = df[f"atr_{c.atr_period}"]
            atr_ma = float(atr_series.iloc[-c.atr_filter_period:].mean())
            atr_prev = ft["atr_prev"]
            atr_expanding = atr > atr_prev
            atr_above_floor = atr_ma > 0 and atr >= atr_ma * c.atr_filter_slack
            atr_gate_ok = bool(atr_above_floor and atr_expanding)
            if (not atr_gate_ok) and str(getattr(c, "atr_filter_mode", "gate")) == "gate":
                return df, info
        info["atr_gate_ok"] = bool(atr_gate_ok)

        # ── BB 突破 ──────────────────────────────────────────────
        bb_upper = ft["bb_upper"]
        bb_lower = ft["bb_lower"]
        bb_bull_break = price > bb_upper
        bb_bear_break = price < bb_lower
        if not bb_bull_break and not bb_bear_break:
            return df, info

        # ── 稳健“趋势突破”模式：趋势环境通过时，不强制 squeeze/bb_width 压缩门控 ──
        mtf_5m = int(info.get("mtf_5m_dir", 0))
        mtf_15m = int(info.get("mtf_15m_dir", 0))
        adx = float(ft.get("adx", 0.0))
        st_dir = ft.get("supertrend_dir")
        st_bull = (st_dir == 1.0)
        st_bear = (st_dir == -1.0)

        # MTF 方向一致性（参考 adaptive_trend_fusion 的 relaxed/strict 思路）
        def _mtf_ok(direction: int) -> bool:
            if mtf_5m != direction:
                return False
            if c.mtf_strict:
                return mtf_15m == direction
            # relaxed: 15m 允许中立(0)，但不允许反向
            return mtf_15m != -direction

        trend_long_ok = (
            c.trend_breakout_enabled
            and bb_bull_break
            and st_bull
            and _mtf_ok(1)
            and adx >= float(c.trend_adx_threshold)
        )
        trend_short_ok = (
            c.trend_breakout_enabled
            and c.trend_breakout_allow_short
            and bb_bear_break
            and st_bear
            and _mtf_ok(-1)
            and adx >= float(c.trend_adx_threshold)
        )
        # 稳健版：默认只允许 LONG 放宽；可选开启 SHORT 放宽
        bypass_compression_gates = bool(trend_long_ok or trend_short_ok)

        if not bypass_compression_gates:
            # ── TTM Squeeze（压缩期）────────────────────────────────
            if not ft["had_recent_squeeze"]:
                return df, info

            # ── BBWidth 分位压缩门控（只做“真压缩”）──────────────────
            if c.bb_width_gate_enabled:
                bw = float(ft.get("bb_width", float("nan")))
                if np.isnan(bw):
                    return df, info

                bw_col = f"bb_width_{c.bb_width_period}"
                lookback = int(c.bb_width_lookback)
                if bw_col not in df.columns or lookback <= 0 or len(df) < lookback + 2:
                    return df, info

                hist = df[bw_col].astype(float).iloc[-lookback - 1:-1].dropna()
                if len(hist) < max(20, int(lookback * 0.5)):
                    return df, info

                thr = float(hist.quantile(float(c.bb_width_quantile)))
                if bw > thr:
                    return df, info

        # SHORT 默认更严格（至少 5m 必须同向；15m 不反向）
        if bb_bear_break and not _mtf_ok(-1):
            return df, info
        if bb_bull_break and not _mtf_ok(1):
            return df, info

        # ── squeeze_mom 方向过滤（可选，用于减少假突破）───────────
        if c.squeeze_mom_filter_enabled:
            sm = float(ft.get("squeeze_mom", 0.0))
            if bb_bull_break and sm <= 0:
                return df, info
            if bb_bear_break and sm >= 0:
                return df, info

        # ── 成交量放大门控 ─────────────────────────────────────
        vol_surging = False
        if VOL_COL in df.columns and len(df) > c.vol_lookback + 1:
            volumes = df[VOL_COL].astype(float)
            curr_vol = safe_float(volumes.iloc[-1])
            vol_avg = float(volumes.iloc[-c.vol_lookback - 1:-1].mean())
            vol_surging = vol_avg > 0 and curr_vol >= vol_avg * c.vol_surge_mult

        if c.vol_gate_enabled and not vol_surging:
            return df, info

        # ══════════════════════════════════════════════════════════════
        # § 2.5  追空/追多末端保护
        #
        #   防止在趋势末端追单（例如暴跌底部追空、暴涨顶部追多）。
        #   三层保护：(a) 低位/高位禁入  (b) RSI 极端耗竭  (c) 超卖/超买反弹
        # ══════════════════════════════════════════════════════════════

        rsi = ft["rsi"]
        rsi_col = f"rsi_{c.rsi_period}"
        ema_col = f"ema_{c.ema_trend_period}"

        if bb_bear_break and c.short_protection_enabled and not np.isnan(rsi):
            # 近期 RSI 最小值（用于判断是否出现过超卖）
            rsi_min_recent = float("nan")
            if rsi_col in df.columns and len(df) >= c.rebound_window:
                rsi_min_recent = float(
                    np.nanmin(df[rsi_col].iloc[-c.rebound_window:].values)
                )

            bypass_lowpos = bool(
                getattr(c, "bypass_lowpos_protection_in_trend_breakout", True)
                and trend_short_ok
            )
            # (a) 低位追空禁入：近期出现过超卖 + 价格处于区间底部 → 禁止做空
            if (not bypass_lowpos
                    and (not np.isnan(rsi_min_recent))
                    and rsi_min_recent <= c.lowpos_min_rsi_recent
                    and len(df) >= c.lowpos_lookback):
                lo = float(np.nanmin(df["low"].iloc[-c.lowpos_lookback:].values))
                hi = float(np.nanmax(df["high"].iloc[-c.lowpos_lookback:].values))
                if hi > lo:
                    pos_in_range = (price - lo) / (hi - lo)
                    if pos_in_range <= c.lowpos_max:
                        return df, info

            # (b) RSI 极端超卖耗竭：RSI 极低 + 宏观下行斜率不足 → 趋势即将反转
            if rsi <= c.exhaust_rsi_short_max:
                macro_slope = float("nan")
                if (ema_col in df.columns
                        and len(df) > c.exhaust_slope_lookback):
                    ema_now = float(df[ema_col].iloc[-1])
                    ema_prev = float(df[ema_col].iloc[-1 - c.exhaust_slope_lookback])
                    if not np.isnan(ema_now) and not np.isnan(ema_prev):
                        macro_slope = ema_now - ema_prev
                if np.isnan(macro_slope) or macro_slope > -c.exhaust_slope_weak_usdt:
                    return df, info

            # (c) 超卖反弹追空保护：近期超卖 → RSI 急反弹 + 1m 回升 → 禁止追空
            if (not np.isnan(rsi_min_recent)
                    and rsi_min_recent <= c.rebound_min_rsi
                    and rsi >= c.rebound_curr_rsi_min
                    and len(df) >= 2):
                c0 = float(df["close"].iloc[-1])
                c1 = float(df["close"].iloc[-2])
                if c1 > 0 and (c0 / c1 - 1.0) >= c.rebound_ret_1m_min:
                    return df, info

        if bb_bull_break and c.long_protection_enabled and not np.isnan(rsi):
            # 近期 RSI 最大值（用于判断是否出现过超买）
            rsi_max_recent = float("nan")
            if rsi_col in df.columns and len(df) >= c.rebound_window:
                rsi_max_recent = float(
                    np.nanmax(df[rsi_col].iloc[-c.rebound_window:].values)
                )

            bypass_highpos = bool(
                getattr(c, "bypass_highpos_protection_in_trend_breakout", True)
                and trend_long_ok
            )
            # (a) 高位追多禁入：近期出现过超买 + 价格处于区间顶部 → 禁止做多
            if (not bypass_highpos
                    and (not np.isnan(rsi_max_recent))
                    and rsi_max_recent >= c.highpos_max_rsi_recent
                    and len(df) >= c.highpos_lookback):
                lo = float(np.nanmin(df["low"].iloc[-c.highpos_lookback:].values))
                hi = float(np.nanmax(df["high"].iloc[-c.highpos_lookback:].values))
                if hi > lo:
                    pos_in_range = (price - lo) / (hi - lo)
                    if pos_in_range >= c.highpos_min:
                        return df, info

            # (b) RSI 极端超买耗竭：RSI 极高 + 宏观上行斜率不足 → 趋势即将反转
            if rsi >= c.exhaust_rsi_long_min:
                macro_slope = float("nan")
                if (ema_col in df.columns
                        and len(df) > c.exhaust_slope_lookback):
                    ema_now = float(df[ema_col].iloc[-1])
                    ema_prev = float(df[ema_col].iloc[-1 - c.exhaust_slope_lookback])
                    if not np.isnan(ema_now) and not np.isnan(ema_prev):
                        macro_slope = ema_now - ema_prev
                if np.isnan(macro_slope) or macro_slope < c.exhaust_slope_weak_usdt:
                    return df, info

        # ══════════════════════════════════════════════════════════════
        # § 3  多因子评分（满分 3.0，阈值 min_entry_score）
        # ══════════════════════════════════════════════════════════════

        # ATR soft mode: 不再一票否决，而是提高准入分数（让其它强因子补偿）
        effective_min = float(c.min_entry_score)
        if (not atr_gate_ok) and str(getattr(c, "atr_filter_mode", "gate")) != "gate":
            effective_min += float(getattr(c, "atr_soft_penalty", 0.25))
        info["effective_min_entry_score"] = float(effective_min)

        bull_score = 0.0
        bear_score = 0.0

        # [1] BB 突破确认（1.0 分）
        if bb_bull_break:
            bull_score += 1.0
        if bb_bear_break:
            bear_score += 1.0

        # [2] 成交量放大（0.5 分）
        if vol_surging:
            bull_score += 0.5
            bear_score += 0.5

        # [3] ROC 动量方向一致且增强（1.0 分）
        roc_mom = float(ft.get("roc_mom", 0.0))
        roc_mom_prev = float(ft.get("roc_mom_prev", 0.0))
        if roc_mom > 0 and roc_mom > roc_mom_prev:
            bull_score += 1.0
        if roc_mom < 0 and roc_mom < roc_mom_prev:
            bear_score += 1.0

        # [4] ADX 趋势强度 + DI 方向（0.5 分）
        adx = ft["adx"]
        if adx > c.adx_threshold:
            plus_di = ft["plus_di"]
            minus_di = ft["minus_di"]
            if plus_di > 0 and minus_di > 0:
                if plus_di > minus_di:
                    bull_score += 0.5
                else:
                    bear_score += 0.5

        # ══════════════════════════════════════════════════════════════
        # § 4  EMA 趋势过滤 + 方向确认
        # ══════════════════════════════════════════════════════════════

        ema_fast = ft["ema_fast"]
        ema_slow = ft["ema_slow"]
        htf_bull = ema_fast > 0 and ema_slow > 0 and ema_fast > ema_slow
        htf_bear = ema_fast > 0 and ema_slow > 0 and ema_fast < ema_slow

        if bull_score >= effective_min:
            if c.ema_trend_filter and ema_slow > 0 and price <= ema_slow:
                pass
            elif c.ema_crossover_filter and ema_fast > 0 and ema_slow > 0 and not htf_bull:
                pass
            else:
                info["signal_direction"] = 1
                info["entry_score"] = float(bull_score)
                return df, info

        if bear_score >= effective_min and not c.long_only:
            if c.ema_trend_filter and ema_slow > 0 and price >= ema_slow:
                pass
            elif c.ema_crossover_filter and ema_fast > 0 and ema_slow > 0 and not htf_bear:
                pass
            else:
                info["signal_direction"] = -1
                info["entry_score"] = float(bear_score)
                return df, info

        return df, info


# ===========================================================================
# ExitController — 出场信号 + 风控状态管理
# ===========================================================================

class ExitController:
    """
    出场信号（任一条件触发即 CLOSE）：
      1. TP/SL — 保证金 ROI% 硬止盈止损（TP=20% / SL=10%，5x杠杆≈价格4%/2%）
      2. 最大持仓 bar 数 → 强制平仓（720 bar = 12 小时）
      3. ATR 波动率飙升 — 当前 ATR > 入场 ATR × 3.5，市场失控
    """

    def __init__(self, cfg: Any) -> None:
        self._c = cfg

    def update_risk_state(
        self, ts: TradeState, current_pos: float,
        price: float, exit_price: Optional[float] = None,
    ) -> None:
        """推断盈亏并更新风控状态（冷却/连续亏损等）。"""
        c = self._c
        old_exited = (
            (ts.last_position != 0.0 and current_pos == 0.0)
            or (ts.last_position > 0 and current_pos < 0)
            or (ts.last_position < 0 and current_pos > 0)
        )

        if old_exited and ts.entry_price is not None:
            fill = exit_price if exit_price is not None else price
            if ts.last_position > 0:
                pnl = (fill - ts.entry_price) * abs(ts.last_position)
            else:
                pnl = (ts.entry_price - fill) * abs(ts.last_position)

            if pnl < 0:
                ts.consecutive_losses += 1
                ts.cooldown_remaining = c.cooldown_bars
                ts.loss_pause_bars = 0
            else:
                ts.consecutive_losses = 0
                ts.loss_pause_bars = 0
                ts.cooldown_remaining = c.win_cooldown_bars

        if old_exited:
            ts.entry_price = None
            ts.entry_atr = 0.0

        ts.tpsl_closed = old_exited and ts.last_position != 0.0
        ts.last_position = current_pos

    def __call__(self, df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, Dict]:
        ts: TradeState = info["trade_state"]
        curr_side = int(info["curr_side"])

        info["exit_signal"] = None

        if curr_side == 0:
            return df, info

        ft = info["features"]
        info["exit_signal"] = self._check(
            self._c, ts, curr_side, ft["price"], ft["atr"],
        )
        return df, info

    def _check(self, c, ts: TradeState, curr_side: int,
               price: float, atr: float) -> Optional[float]:
        if ts.entry_price is None or ts.entry_price <= 0:
            return None

        close_sig = -float(curr_side)

        # ── 1. TP/SL (ROI%) ───────────────────────────────────────
        pnl_pct = (price - ts.entry_price) / ts.entry_price * curr_side
        roi = pnl_pct * max(1, ts.entry_leverage)

        regime = getattr(ts, "entry_regime", "n/a")
        if regime == "range":
            tp = float(c.range_tp_roi_pct) / 100.0
            sl = float(c.range_sl_roi_pct) / 100.0
        else:
            tp = float(c.tp_roi_pct) / 100.0
            sl = float(c.sl_roi_pct) / 100.0

        if roi >= tp:
            return close_sig
        if roi <= -sl:
            return close_sig

        # ── 2. 最大持仓 → 强制平仓 ───────────────────────────────
        if ts.hold_bars >= c.max_hold_bars:
            return close_sig

        # ── 3. ATR 波动率飙升 ─────────────────────────────────────
        if (ts.entry_atr > 0 and not np.isnan(atr)
                and float(c.vol_spike_atr_mult) > 0):
            if atr > ts.entry_atr * float(c.vol_spike_atr_mult):
                return close_sig

        return None


# ===========================================================================
# PositionSizer — 仓位计算
# ===========================================================================

class PositionSizer:
    """
    保证金百分比仓位计算
      notional = equity × margin_pct × leverage
      quantity = notional / price
    """

    def __init__(self, cfg: Any) -> None:
        self._c = cfg

    def __call__(self, df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, Dict]:
        c      = self._c
        equity = float(info.get("equity") or 0.0)
        price  = info["features"]["price"]

        lev = max(1, min(int(c.leverage or 1), int(c.max_leverage)))

        if equity <= 0 or price <= 0:
            info["sizing"] = {
                "quantity": 0.0, "leverage": lev,
                "notional": 0.0, "margin_usdt": 0.0,
            }
            return df, info

        pct      = max(0.0, float(c.margin_pct or 0.0))
        notional = equity * pct * float(lev)
        quantity = notional / price
        quantity = max(quantity, float(c.min_quantity))
        quantity = round(quantity, int(c.quantity_precision))
        notional = quantity * price

        info["sizing"] = {
            "quantity":    float(quantity),
            "leverage":    lev,
            "notional":    float(notional),
            "margin_usdt": float(notional / lev if lev > 0 else 0.0),
        }
        return df, info


# ===========================================================================
# MomentumBreakoutStrategy — 配置 + 主流程
# ===========================================================================

class MomentumBreakoutStrategy(Strategy):
    """动量突破策略 — 波动率压缩突破 + 4 因子评分 + ROI% TP/SL + ATR 波动率退出。"""

    # ── 策略元信息 ────────────────────────────────────────────
    name: str = Field(default="momentum_breakout", description="策略名称")
    description: str = Field(
        default="动量突破策略 v6 — TTM Squeeze + BBWidth 分位压缩 + ROC 动量确认 + ROI% TP/SL",
    )
    factor_names: List[str] = Field(
        default=["bb", "ttm_squeeze", "bb_width", "roc", "atr", "adx", "ema", "rsi", "supertrend", "chop"],
    )

    # ── 交易参数 ──────────────────────────────────────────────
    symbol: str = Field(default="BTCUSDT")
    quantity_precision: int = Field(default=3)
    min_quantity: float = Field(default=0.02)

    # ── 仓位管理 ──────────────────────────────────────────────
    leverage: int = Field(default=5, description="杠杆倍数")
    max_leverage: int = Field(default=10, description="杠杆上限")
    margin_pct: float = Field(
        default=0.5,
        description="每笔保证金占净值比例；notional = equity × margin_pct × leverage",
    )
    fee_rate: float = Field(default=DEFAULT_TAKER_FEE)

    # ── 止盈止损（保证金 ROI%）+ 波动率 ──────────────────────
    tp_roi_pct: float = Field(default=20.0, description="止盈: 保证金 ROI 达到 20%（5x杠杆≈价格变动4%）")
    sl_roi_pct: float = Field(default=10.0, description="止损: 保证金 ROI 亏损 10%（5x杠杆≈价格变动2%）")
    range_tp_roi_pct: float = Field(default=10.0, description="震荡/压缩突破专用止盈（ROI%），默认更快止盈")
    range_sl_roi_pct: float = Field(default=8.0, description="震荡/压缩突破专用止损（ROI%），默认略紧")
    vol_spike_atr_mult: float = Field(default=3.5, description="ATR 飙升退出: 当前ATR > 入场ATR × 3.5")

    # ── Regime（参考 adaptive_trend_fusion：CHOP 制度切换）───────────────
    chop_period: int = Field(default=14, description="CHOP 周期")
    chop_trend_max: float = Field(default=46.0, description="chop<=该值判定为趋势")
    chop_range_min: float = Field(default=60.0, description="chop>=该值判定为震荡")
    regime_skip_transition: bool = Field(default=True, description="CHOP 过渡区(transition)是否跳过交易")

    # ── 波动充裕门槛（参考 adaptive_trend_fusion：atr_min_usdt）──────────
    atr_min_gate_enabled: bool = Field(default=False, description="启用 ATR 最低门槛（低波动期不交易）")
    atr_min_usdt: float = Field(default=45.0, description="ATR 最低阈值(USDT)")

    # ── 压缩/蓄势（TTM Squeeze + BBWidth 分位）─────────────────
    bb_period: int = Field(default=20)
    squeeze_period: int = Field(default=20, description="TTM Squeeze 周期")
    squeeze_recent_bars: int = Field(default=5, description="最近 N bar 内出现过 squeeze_on 即可")
    squeeze_mom_filter_enabled: bool = Field(default=True, description="使用 squeeze_mom 做突破方向过滤")

    bb_width_period: int = Field(default=20, description="BBWidth 周期")
    bb_width_gate_enabled: bool = Field(default=True, description="BBWidth 分位压缩作为前置门控")
    bb_width_lookback: int = Field(default=240, description="BBWidth 分位统计回看窗口（bar）")
    bb_width_quantile: float = Field(default=0.2, description="低分位阈值（例如 0.2 表示 20% 分位）")

    # ── 稳健趋势突破增强（参考 adaptive_trend_fusion 的方向一致性）──────────
    trend_breakout_enabled: bool = Field(
        default=True,
        description="趋势突破增强：趋势环境通过时，放宽 squeeze/bb_width 硬门控以抓趋势延续",
    )
    trend_adx_threshold: float = Field(default=25.0, description="趋势环境最低 ADX（稳健版）")
    mtf_5m_period: int = Field(default=20, description="5m 重采样 EMA 周期（方向一致性）")
    mtf_15m_period: int = Field(default=14, description="15m 重采样 EMA 周期（方向一致性）")
    trend_breakout_allow_short: bool = Field(default=True, description="趋势突破放宽是否允许做空（默认开启，允许双向趋势突破）")

    # ── MTF 过滤强度（参考 adaptive_trend_fusion）────────────────────────
    mtf_strict: bool = Field(default=False, description="True=5m/15m必须同向；False=5m同向且15m不反向")

    # ── 末端保护的趋势旁路（避免强趋势突破被“高位/低位禁入”卡死）───────────
    bypass_highpos_protection_in_trend_breakout: bool = Field(
        default=True,
        description="当趋势突破增强(trend_breakout)判定通过时，旁路高位追多禁入（避免错过强趋势突破）",
    )
    bypass_lowpos_protection_in_trend_breakout: bool = Field(
        default=True,
        description="当趋势突破增强(trend_breakout)判定通过且允许做空时，旁路低位追空禁入（避免错过强趋势下破）",
    )

    # ── Supertrend 方向锚定（参考 adaptive_trend_fusion）──────────────────
    supertrend_period: int = Field(default=10, description="Supertrend ATR 周期")
    supertrend_mult: float = Field(default=3.0, description="Supertrend ATR 倍数（越大翻转越少）")
    supertrend_confirm_bars: int = Field(default=3, description="Supertrend 方向需连续保持 N 根 bar 才有效")

    # ── ROC 动量确认 ───────────────────────────────────────────
    roc_window: int = Field(default=10, description="ROC 窗口（bar）")

    # ── ATR & 活跃度 ─────────────────────────────────────────
    atr_period: int = Field(default=14)
    atr_filter_enabled: bool = Field(default=True)
    atr_filter_period: int = Field(default=20)
    atr_filter_slack: float = Field(default=0.85, description="ATR ≥ ATR_MA × 0.85")
    atr_filter_mode: str = Field(
        default="soft",
        description="ATR 活跃度过滤模式：gate=不满足直接拒绝；soft=不满足则提高入场分数阈值",
    )
    atr_soft_penalty: float = Field(
        default=0.25,
        description="ATR gate 未满足时，soft 模式下对 min_entry_score 的附加惩罚（0.25 意味着需更强信号才入场）",
    )

    # ── ADX ───────────────────────────────────────────────────
    adx_period: int = Field(default=14)
    adx_threshold: float = Field(default=25.0)

    # ── 成交量 ────────────────────────────────────────────────
    vol_lookback: int = Field(default=20)
    vol_surge_mult: float = Field(default=1.5)
    vol_gate_enabled: bool = Field(default=True, description="成交量放大作为前置门控")

    # ── RSI ────────────────────────────────────────────────────
    rsi_period: int = Field(default=14)

    # ── 追空末端保护（防止在暴跌底部追空被反弹打脸）─────────
    short_protection_enabled: bool = Field(default=True, description="启用追空末端/低位/反弹保护")
    exhaust_rsi_short_max: float = Field(default=25.0, description="RSI ≤ 该值视为极端超卖")
    exhaust_slope_lookback: int = Field(default=60, description="EMA 宏观斜率回看 bars")
    exhaust_slope_weak_usdt: float = Field(default=150.0, description="EMA 下行斜率不足(> -该值)时禁止在超卖区追空")
    rebound_window: int = Field(default=20, description="反弹检测：回看 RSI 窗口 bars")
    rebound_min_rsi: float = Field(default=30.0, description="反弹检测：近期 RSI min ≤ 该值视为出现过超卖")
    rebound_curr_rsi_min: float = Field(default=42.0, description="反弹检测：当前 RSI ≥ 该值视为明显反弹")
    rebound_ret_1m_min: float = Field(default=0.0005, description="反弹检测：1m 回升幅度阈值")
    lowpos_lookback: int = Field(default=240, description="低位保护：价格区间回看 bars（240=4小时）")
    lowpos_max: float = Field(default=0.40, description="低位保护：处于区间底部该分位以下禁止做空")
    lowpos_min_rsi_recent: float = Field(default=30.0, description="低位保护：需近期出现过 RSI ≤ 该值才触发")

    # ── 追多末端保护（防止在暴涨顶部追多被回调打脸）─────────
    long_protection_enabled: bool = Field(default=True, description="启用追多末端/高位保护")
    exhaust_rsi_long_min: float = Field(default=75.0, description="RSI ≥ 该值视为极端超买")
    highpos_lookback: int = Field(default=120, description="高位保护：价格区间回看 bars")
    highpos_min: float = Field(default=0.72, description="高位保护：处于区间顶部该分位以上禁止做多")
    highpos_max_rsi_recent: float = Field(default=70.0, description="高位保护：需近期出现过 RSI ≥ 该值才触发")

    # ── 评分 & EMA 趋势 ──────────────────────────────────────
    min_entry_score: float = Field(default=2.5, description="最低入场分数（满分 3.0）")
    ema_trend_period: int = Field(default=200, description="EMA 宏观趋势周期")
    ema_trend_filter: bool = Field(default=True)
    ema_fast_period: int = Field(default=50, description="短期 EMA 周期")
    ema_crossover_filter: bool = Field(default=True, description="EMA50/200 双均线趋势过滤")
    long_only: bool = Field(default=False, description="只做多模式（默认关闭，双向交易）")

    # ── 过度延伸过滤（宁可少做也不追高/追低）───────────────────────────
    entry_dist_filter_enabled: bool = Field(
        default=True,
        description="启用过度延伸过滤：|price-EMA200|/ATR 过大则拒绝入场",
    )
    max_entry_dist_atr: float = Field(
        default=3.0,
        description="允许入场的最大 dist_atr=|price-EMA200|/ATR；调小→更少追单",
    )

    # ── 风控 ──────────────────────────────────────────────────
    max_hold_bars: int = Field(default=720, description="最大持仓 bar 数（720=12小时），给趋势充分发展空间")
    cooldown_bars: int = Field(default=60)
    win_cooldown_bars: int = Field(default=30)
    max_consecutive_losses: int = Field(default=2)
    loss_pause_timeout: int = Field(default=200)

    # ── 反向处理 ──────────────────────────────────────────────
    flip_mode: str = Field(default="close_then_wait")
    reverse_cooldown_bars: int = Field(default=60)

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
        """动量突破策略主入口。"""

        # ── Step 0: 因子计算 ───────────────────────────────────
        await factor_manager("bb", df)
        await factor_manager("ttm_squeeze", df, periods=[self.squeeze_period])
        await factor_manager("bb_width", df, periods=[self.bb_width_period])
        await factor_manager("roc", df, windows=[self.roc_window])
        await factor_manager("atr", df)
        await factor_manager("adx", df)
        await factor_manager("ema", df, periods=[self.ema_trend_period, self.ema_fast_period])
        await factor_manager("rsi", df)
        await factor_manager("chop", df, periods=[self.chop_period])
        await factor_manager(
            "supertrend", df,
            periods=[self.supertrend_period],
            multipliers=[self.supertrend_mult],
        )

        # ── Step 1: 特征提取 ──────────────────────────────────
        ts = self._trade_state
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
        info: Dict[str, Any] = {
            "current_pos": current_pos,
            "equity": equity,
            "curr_side": curr_side,
            "trade_state": ts,
            # 多时间框架 EMA 方向（稳健趋势突破用）
            "mtf_5m_dir": _compute_mtf_ema_dir(df, "5min", self.mtf_5m_period),
            "mtf_15m_dir": _compute_mtf_ema_dir(df, "15min", self.mtf_15m_period),
        }

        df, info = self._feature_engine(df, info)
        ft = info["features"]
        price = ft["price"]
        atr = ft["atr"]

        # ── Regime: trend / range / transition（CHOP）──────────────
        regime = "n/a"
        chop_col = f"chop_{self.chop_period}"
        try:
            if chop_col in df.columns:
                chop_v = float(df[chop_col].iloc[-1])
                if not np.isnan(chop_v):
                    if chop_v <= float(self.chop_trend_max):
                        regime = "trend"
                    elif chop_v >= float(self.chop_range_min):
                        regime = "range"
                    else:
                        regime = "transition"
        except Exception:
            regime = "n/a"
        info["regime"] = regime

        # ── Step 2: 风控状态更新 ──────────────────────────────
        last_exit_price: Optional[float] = kwargs.get("last_exit_price")
        self._exit_controller.update_risk_state(ts, current_pos, price, exit_price=last_exit_price)

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
        meta: Dict[str, Any] = {}

        if curr_side != 0 and exit_sig is not None:
            signal_name = "CLOSE"
            qty = abs(current_pos)
            order = {
                "symbol":        self.symbol,
                "side":          "BUY" if curr_side == -1 else "SELL",
                "type":          "MARKET",
                "quantity":      qty,
                "position_side": "BOTH",
                "reduce_only":   True,
            }

        elif curr_side == 0 and new_sig != 0:
            signal_name = "LONG" if new_sig == 1 else "SHORT"
            qty = sizing["quantity"]

            ts.entry_price = price
            ts.entry_atr = float(atr) if not np.isnan(atr) else 0.0
            ts.entry_regime = str(info.get("regime") or "n/a")

            order = {
                "symbol":        self.symbol,
                "side":          "BUY" if new_sig == 1 else "SELL",
                "type":          "MARKET",
                "quantity":      qty,
                "position_side": "BOTH",
            }

            _lev = max(1, int(self.leverage))
            if ts.entry_regime == "range":
                _sl = float(self.range_sl_roi_pct)
                _tp = float(self.range_tp_roi_pct)
            else:
                _sl = float(self.sl_roi_pct)
                _tp = float(self.tp_roi_pct)
            sl_dist  = price * _sl / 100.0 / _lev
            tp_dist  = price * _tp / 100.0 / _lev
            sl_price = round(price - sl_dist if new_sig == 1 else price + sl_dist, 2)
            tp_price = round(price + tp_dist if new_sig == 1 else price - tp_dist, 2)
            order["tp_sl_mode"]  = "PRICE"
            order["stop_loss"]   = sl_price
            order["take_profit"] = tp_price
            meta.update({
                "sl_price": sl_price, "tp_price": tp_price,
                "sl_distance": round(sl_dist, 2),
                "tp_distance": round(tp_dist, 2),
            })

        if signal_name == "CLOSE" and self.flip_mode == "close_then_wait":
            ts.reverse_cooldown = int(self.reverse_cooldown_bars)

        # ── Step 8: 日志字段 ──────────────────────────────────
        factors_json = build_factors_json(df, [
            f"adx_{self.adx_period}", f"plus_di_{self.adx_period}", f"minus_di_{self.adx_period}",
            f"atr_{self.atr_period}",
            f"bb_upper_{self.bb_period}", f"bb_lower_{self.bb_period}", f"bb_middle_{self.bb_period}",
            f"squeeze_on_{self.squeeze_period}", f"squeeze_mom_{self.squeeze_period}",
            f"bb_width_{self.bb_width_period}",
            f"roc_{self.roc_window}",
            f"ema_{self.ema_trend_period}", f"ema_{self.ema_fast_period}",
            f"rsi_{self.rsi_period}",
            f"chop_{self.chop_period}",
            f"supertrend_dir_{self.supertrend_period}_{int(float(self.supertrend_mult) * 10)}",
        ], extra={
            "hold_bars": ts.hold_bars,
            "consecutive_losses": ts.consecutive_losses,
            "cooldown_remaining": ts.cooldown_remaining,
            "entry_score": info.get("entry_score"),
            "effective_min_entry_score": info.get("effective_min_entry_score"),
            "atr_gate_ok": info.get("atr_gate_ok"),
            "dist_atr": info.get("dist_atr", info.get("features", {}).get("dist_atr")),
            "regime": info.get("regime"),
            "mtf_5m_dir": info.get("mtf_5m_dir", 0),
            "mtf_15m_dir": info.get("mtf_15m_dir", 0),
            "bb_width": info.get("features", {}).get("bb_width"),
            "squeeze_on": info.get("features", {}).get("squeeze_on"),
            "squeeze_mom": info.get("features", {}).get("squeeze_mom"),
            "roc_mom": info.get("features", {}).get("roc_mom"),
        })
        meta["factors_json"] = factors_json

        return {
            "signal": signal_name,
            "order": order,
            "leverage": int(self.leverage),
            "sizing": sizing,
            "atr": atr,
            "meta": meta,
            "factors_json": factors_json,
            "risk": {
                "cooldown_remaining": ts.cooldown_remaining,
                "consecutive_losses": ts.consecutive_losses,
                "loss_pause_bars": ts.loss_pause_bars,
                "hold_bars": ts.hold_bars,
            },
        }
