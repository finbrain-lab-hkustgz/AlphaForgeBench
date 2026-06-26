"""
Adaptive Trend Fusion Strategy — Simplified v5
Binance USDT-M Perpetual Futures (BTCUSDT)

Core pipeline:
  FeatureEngine → EnterController → ExitController → PositionSizer → OrderRouter

Entry（三重过滤，假信号大幅减少）:
  LONG  : Supertrend_dir == +1 (1m)
          AND  MTF 5m EMA 方向 == +1
          AND  MTF 15m EMA 方向 == +1
          AND  ADX ≥ threshold
          AND  rsi_long_min ≤ RSI ≤ rsi_long_max
  SHORT : 反向

Exit:
  Pure TP/SL — ROI ≥ tp_roi_pct → take profit / ROI ≤ -sl_roi_pct → stop loss
  (trend / range / chase 各用各的 TP/SL 参数)
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
# MTF helper — 高时间框架 EMA 方向
# ===========================================================================

def _compute_mtf_ema_dir(df: pd.DataFrame, freq: str, period: int) -> int:
    """
    将 1m df 重采样到 freq（如 '5min'/'15min'），计算 EMA，返回最后一根的方向。
    +1 = EMA 上升（多头），-1 = 下降（空头），0 = 数据不足
    """
    try:
        ts_col = "timestamp" if "timestamp" in df.columns else None
        if ts_col:
            s = df[ts_col]
            try:
                # common backtest timestamp format: "YYYY-mm-dd HH:MM:SS"
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
# TradeState — 跨 bar 共享的持仓状态
# ===========================================================================

@dataclass
class TradeState:
    entry_price:    Optional[float] = None
    entry_leverage: int             = 1
    entry_alpha_type: str           = "none"   # "trend" | "range" | "none"
    entry_regime:     str           = "n/a"    # "trend" | "range" | "transition" | "n/a"
    entry_atr:      float           = 0.0
    last_position:  float           = 0.0
    hold_bars:      int             = 0
    last_side:      int             = 0
    pending_fill:   bool            = False

    def update_from_fill(self, fill_price: float, leverage: int) -> None:
        """成交回报到达后由 caller 调用。"""
        self.entry_price    = float(fill_price)
        self.entry_leverage = max(1, int(leverage))
        self.pending_fill   = False


# ===========================================================================
# FeatureEngine — 指标提取
# ===========================================================================

class FeatureEngine:
    """从 DataFrame 提取 Supertrend / EMA / RSI / ADX / ATR / CHOP / Momentum(ROC)。"""

    def __init__(self, cfg: Any) -> None:
        self._c = cfg

    def __call__(self, df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, Dict]:
        c = self._c

        def _last(col: str) -> float:
            if col not in df.columns:
                return float("nan")
            try:
                v = df[col].iloc[-1]
                return float(v) if v is not None else float("nan")
            except (TypeError, ValueError):
                return float("nan")

        st_tag = f"{c.supertrend_period}_{int(float(c.supertrend_mult) * 10)}"
        # Supertrend 稳定方向：最近 confirm_bars 根必须全部一致才有效，否则返回 nan
        st_col = f"supertrend_dir_{st_tag}"
        confirm = max(1, int(c.supertrend_confirm_bars))
        if st_col in df.columns and len(df) >= confirm:
            recent = df[st_col].iloc[-confirm:]
            vals = recent.dropna().values
            if len(vals) == confirm and len(set(vals)) == 1:
                stable_st_dir = float(vals[-1])
            else:
                stable_st_dir = float("nan")
        else:
            stable_st_dir = float("nan")

        info["features"] = {
            "price":          float(df[PRICE_COL].iloc[-1]),
            # ema_entry 用于入场确认（更敏感）；ema_slow 用于结构止损/大势参考（更稳）
            "ema_entry":      _last(f"ema_{c.ema_entry_period}"),
            "ema_slow":       _last(f"ema_{c.ema_slow_period}"),
            "adx":            _last(f"adx_{c.adx_period}"),
            "rsi":            _last(f"rsi_{c.rsi_period}"),
            "atr":            _last(f"atr_{c.atr_period}"),
            "chop":           _last(f"chop_{c.chop_period}"),
            # roc_{w} = close.shift(w) / close  (Alpha158 口径)
            "roc":            _last(f"roc_{c.chase_mom_window}"),
            "supertrend_dir": stable_st_dir,
        }
        return df, info


# ===========================================================================
# EnterController — 入场信号
# ===========================================================================

class EnterController:
    """
    多因子连续评分入场系统。

    所有条件（ATR / CHOP / MTF / EMA位置 / dist / ADX / RSI）均转化为 0~1 连续分，
    加权归一化后与阈值比较。没有硬 gate——弱信号不被直接拒绝，而是拉低总分；
    只有总分不达标时才不开仓，赋予策略更均衡的多空机会。

    流程：
      § 1  守卫检查   — 冷却 / 已持仓（唯二硬条件，不参与评分）
      § 2  特征解包   — 基础量计算、数据完整性检查
      § 3  Chase 路径 — 极端动量快速追单（独立评分，绕过 §5-7）
      § 4  Regime     — transition 期跳过（不明确方向不交易）
      § 5  因子评分   — 7 个因子各自输出 0~1 连续分
      § 6  加权合并   — 自动归一化，输出最终入场得分
      § 7  方向确认   — Supertrend 定方向 + 得分阈值准入
    """

    def __init__(self, cfg: Any) -> None:
        self._c        = cfg
        self._cooldown = 0

    def __call__(self, df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, Dict]:
        c  = self._c
        ft = info["features"]
        ts: TradeState = info["trade_state"]

        info["signal_direction"] = 0
        info["entry_score"]      = None
        info["alpha_type"]       = "none"

        # ══════════════════════════════════════════════════════════════════
        # § 1  守卫检查
        #      冷却期 / 已持仓 → 直接跳过，不参与任何评分逻辑
        # ══════════════════════════════════════════════════════════════════
        if self._cooldown > 0:
            self._cooldown -= 1
            return df, info

        if ts.last_side != 0:
            return df, info

        # ══════════════════════════════════════════════════════════════════
        # § 2  特征解包 & 基础量计算
        # ══════════════════════════════════════════════════════════════════
        st_dir    = ft["supertrend_dir"]   # +1 / -1 / nan（Supertrend 方向锚）
        adx       = ft["adx"]
        rsi       = ft["rsi"]
        atr       = ft["atr"]
        chop      = ft["chop"]
        roc_inv   = ft.get("roc", float("nan"))
        ema_entry = ft.get("ema_entry", float("nan"))
        ema_slow  = ft["ema_slow"]
        price     = ft["price"]

        # 任一核心指标未 warm-up → 不可交易
        if any(np.isnan(v) for v in [st_dir, adx, rsi, atr, chop, ema_entry, ema_slow]):
            return df, info

        regime  = str(info.get("regime") or "n/a")
        mtf_5m  = int(info.get("mtf_5m_dir",  0))
        mtf_15m = int(info.get("mtf_15m_dir", 0))

        _atr     = float(atr) if float(atr) > 1e-12 else 1e-12
        dist_atr = abs(float(price) - float(ema_entry)) / _atr
        ema_gap  = float(price) - float(ema_entry)  # 正值 = 价格在 EMA 上方

        # EMA 斜率（宏观趋势强弱）：用 ema_entry 作为“更敏感”的趋势代理
        #  - 用于 Range 模式防止强下行里抄底做多（#4）
        #  - 用于 Trend 模式防止极端超卖时继续追空（#5）
        macro_lb = max(1, int(getattr(c, "macro_slope_lookback_bars", 60)))
        macro_slope = float("nan")
        try:
            ema_col = f"ema_{c.ema_entry_period}"
            if ema_col in df.columns and len(df) > macro_lb:
                ema_prev = float(df[ema_col].iloc[-1 - macro_lb])
                if not np.isnan(ema_prev):
                    macro_slope = float(ema_entry) - ema_prev
        except Exception:
            pass
        info["ema_entry_slope"] = float(macro_slope) if not np.isnan(macro_slope) else float("nan")

        # 1m 即时涨跌幅（Chase 路径 & dist 分析用）
        ret_1m = down_1m = up_1m = float("nan")
        try:
            c0, c1 = float(df["close"].iloc[-1]), float(df["close"].iloc[-2])
            if c1 > 1e-12:
                ret_1m  = c0 / c1 - 1.0
                down_1m = float(df["low"].iloc[-1])  / c1 - 1.0
                up_1m   = float(df["high"].iloc[-1]) / c1 - 1.0
        except Exception:
            pass
        info["ret_1m"]  = ret_1m
        info["down_1m"] = down_1m
        info["up_1m"]   = up_1m
        info["dist_atr"] = float(dist_atr)

        # ROC → 5-bar 动量（chase 路径使用）
        mom = float("nan")
        if roc_inv and not np.isnan(float(roc_inv)) and abs(float(roc_inv)) > 1e-12:
            mom = 1.0 / float(roc_inv) - 1.0
        info["momentum"] = mom

        def _clip01(x: float) -> float:
            return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)

        # ══════════════════════════════════════════════════════════════════
        # § 3  Chase 快速追单（极端动量专属路径）
        #
        #   触发前提（硬性，保证只在真正极端行情时才进）：
        #     · dist_atr >= chase_dist_atr_min   价格已大幅偏离 EMA
        #     · down_1m / up_1m / |mom| 超过各自门槛（强烈单根动量）
        #   满足前提后，用独立 chase_score 判断强度，通过即返回，
        #   不走 §5-7 的常规多因子评分。
        # ══════════════════════════════════════════════════════════════════
        chase_enabled  = bool(getattr(c, "chase_enabled", True))
        chase_dist_min = float(getattr(c, "chase_dist_atr_min", 6.0))
        chase_dist_ok  = float(dist_atr) >= chase_dist_min

        chase_down_ok = (not np.isnan(down_1m)) and down_1m <= -float(getattr(c, "chase_down_1m_min_pct", 0.0035))
        chase_up_ok   = (not np.isnan(up_1m))   and up_1m   >=  float(getattr(c, "chase_up_1m_min_pct",   0.0035))
        chase_mom_ok  = (not np.isnan(mom))      and abs(mom) >= float(getattr(c, "chase_mom_min_pct",     0.006))
        chase_adx_ok  = (not bool(getattr(c, "chase_use_adx", False))
                         or float(adx) >= float(getattr(c, "chase_adx_min", 25.0)))

        # Chase 评分 = 1m 插针幅度(45%) + 5-bar 动量(35%) + EMA 距离(20%)
        s_c_1m  = _clip01(
            (max(abs(down_1m) if not np.isnan(down_1m) else 0.0,
                 abs(up_1m)   if not np.isnan(up_1m)   else 0.0)
             - float(getattr(c, "chase_down_1m_min_pct", 0.0035)))
            / max(1e-9, float(getattr(c, "chase_1m_span", 0.01)))
        )
        s_c_mom = (_clip01((abs(mom) - float(getattr(c, "chase_mom_min_pct", 0.006)))
                           / max(1e-9, float(getattr(c, "chase_mom_span", 0.015))))
                   if not np.isnan(mom) else 0.0)
        s_c_dst = _clip01((float(dist_atr) - chase_dist_min)
                          / max(1e-9, float(getattr(c, "chase_dist_span", 6.0))))
        chase_score = 0.45 * s_c_1m + 0.35 * s_c_mom + 0.20 * s_c_dst

        def _mtf_relaxed_ok(direction: int) -> bool:
            return mtf_5m == direction and mtf_15m != -direction

        if chase_enabled and chase_dist_ok and (chase_down_ok or chase_up_ok or chase_mom_ok) and chase_adx_ok:
            rsi_ok_long  = rsi <= float(getattr(c, "chase_rsi_long_max",  75.0))
            rsi_ok_short = rsi >= float(getattr(c, "chase_rsi_short_min", 25.0))
            thr_chase    = float(getattr(c, "chase_score_threshold", 0.60))

            if (chase_down_ok or (not np.isnan(mom) and mom < 0.0 and chase_mom_ok)):
                if (not bool(c.long_only)) and st_dir == -1.0 and _mtf_relaxed_ok(-1) and rsi_ok_short:
                    if chase_score >= thr_chase:
                        info["signal_direction"] = -1
                        info["entry_score"]      = float(chase_score)
                        info["alpha_type"]       = "chase"
                        return df, info
            if (chase_up_ok or (not np.isnan(mom) and mom > 0.0 and chase_mom_ok)):
                if st_dir == 1.0 and _mtf_relaxed_ok(1) and rsi_ok_long:
                    if chase_score >= thr_chase:
                        info["signal_direction"] = 1
                        info["entry_score"]      = float(chase_score)
                        info["alpha_type"]       = "chase"
                        return df, info

        # ══════════════════════════════════════════════════════════════════
        # § 4  Regime 过滤
        #      transition 区间：CHOP 介于趋势/震荡之间，指标互相矛盾，放弃
        # ══════════════════════════════════════════════════════════════════
        if regime == "transition":
            return df, info

        # ══════════════════════════════════════════════════════════════════
        # § 5  多因子连续评分（0 ~ 1，无硬 gate）
        #
        #   设计原则：
        #     · 弱信号不清零，只是降低分值；
        #     · 单一因子偏弱可被其余因子补偿；
        #     · 只有加权总分不达阈值时才拒绝入场。
        # ══════════════════════════════════════════════════════════════════

        # ── 5a. ADX 趋势强度分 ─────────────────────────────────────────
        #   < adx_gate_min       : 0.0（趋势太弱，完全不支持）
        #   gate_min ~ threshold : 0.0 → 0.5 线性（弱趋势小分，但不清零）
        #   threshold ~ +span    : 0.5 → 1.0 线性（强趋势满分）
        adx_gate = float(c.adx_gate_min)
        adx_thr  = float(c.adx_threshold)
        adx_span = max(1e-9, float(c.adx_score_span))
        if float(adx) < adx_gate:
            s_adx = 0.0
        elif float(adx) < adx_thr:
            s_adx = 0.5 * (float(adx) - adx_gate) / max(1e-9, adx_thr - adx_gate)
        else:
            s_adx = _clip01(0.5 + 0.5 * (float(adx) - adx_thr) / adx_span)

        # ── 5b. ATR 波动充裕分 ────────────────────────────────────────
        #   < 0.5×atr_min : 0.0（波动极小，手续费都覆盖不了）
        #   0.5× ~ 1.0×   : 0.0 → 0.5 线性（偏弱，仍给低分）
        #   1.0× ~ 2.0×   : 0.5 → 1.0 线性（理想波动区间）
        atr_min = float(c.atr_min_usdt)
        if float(atr) < atr_min * 0.5:
            s_atr = 0.0
        elif float(atr) < atr_min:
            s_atr = 0.5 * (float(atr) - atr_min * 0.5) / max(1e-9, atr_min * 0.5)
        else:
            s_atr = _clip01(0.5 + 0.5 * (float(atr) - atr_min) / max(1e-9, atr_min))

        # ── 5c. CHOP 趋势纯净分 ───────────────────────────────────────
        #   <= chop_trend_max (46) : 1.0（干净单向趋势）
        #   >= chop_range_min (60) : 0.0（纯震荡，趋势跟踪失效）
        #   中间过渡区              : 线性插值（不是 0/1 切换）
        chop_lo, chop_hi = float(c.chop_trend_max), float(c.chop_range_min)
        if float(chop) <= chop_lo:
            s_chop = 1.0
        elif float(chop) >= chop_hi:
            s_chop = 0.0
        else:
            s_chop = _clip01(1.0 - (float(chop) - chop_lo) / max(1e-9, chop_hi - chop_lo))

        # ── 5d. MTF 方向一致分 ────────────────────────────────────────
        #   5m + 15m 均同向  : 1.0（多框架共振，最强确认）
        #   5m 同向，15m 中立: 0.6（数据不足，给较高分）
        #   5m 同向，15m 反向: 0.3（低框架顺势但高框架反对，给低分但不清零）
        #   5m 反向          : 0.0（短周期已逆势，无支持）
        def _s_mtf(direction: int) -> float:
            if mtf_5m != direction:
                return 0.0
            if mtf_15m == direction:
                return 1.0
            if mtf_15m == 0:
                return 0.6
            return 0.3  # mtf_15m == -direction

        # ── 5e. EMA 方向位置分 ────────────────────────────────────────
        #   价格在正确方向偏离 > 0.1 ATR : 0.5 → 1.0（偏离越大越确认趋势）
        #   价格在 EMA ±0.1 ATR 内       : 0.5（刚穿越，给中性分）
        #   价格在错误方向               : 0.0（方向性否定）
        def _s_ema_pos(direction: int) -> float:
            signed = (ema_gap / _atr) * direction  # > 0 = 价格在期望方向
            if signed > 0.1:
                return _clip01(0.5 + 0.5 * signed / max(1e-9, float(c.max_ema_dist_atr)))
            if signed >= -0.1:
                return 0.5
            return 0.0

        # ── 5f. EMA 距离甜点分 ────────────────────────────────────────
        #   过近 (< 0.3 ATR)     : 0.3（偏离不足，追踪空间小）
        #   上升段 0.3 → mid ATR : 0.3 → 1.0 线性
        #   甜点 mid ~ max_dist  : 1.0（最佳入场偏离区间）
        #   过远 > max_dist      : 1.0 → 0.0 线性（追单反打风险上升）
        dist_lo  = 0.3
        dist_max = float(c.max_ema_dist_atr)
        dist_mid = dist_max * 0.5
        if float(dist_atr) < dist_lo:
            s_dist = 0.3
        elif float(dist_atr) <= dist_mid:
            s_dist = _clip01(0.3 + 0.7 * (float(dist_atr) - dist_lo) / max(1e-9, dist_mid - dist_lo))
        elif float(dist_atr) <= dist_max:
            s_dist = 1.0
        else:
            s_dist = _clip01(1.0 - (float(dist_atr) - dist_max) / max(1e-9, dist_max))

        # ── 5g. RSI 动能健康分 ────────────────────────────────────────
        #   偏离中心值（多=52 / 空=48）越远分越低；
        #   ADX 极强时旁路（极端趋势中 RSI 超买/超卖无反转含义，给固定 0.8）
        adx_bypass_rsi = adx >= float(c.adx_rsi_bypass_min)

        def _s_rsi(center: float) -> float:
            if adx_bypass_rsi:
                return 0.8  # 极强趋势中 RSI 不再主导，给偏高固定分
            return _clip01(1.0 - abs(float(rsi) - center) / max(1e-9, float(c.rsi_score_span)))

        # ══════════════════════════════════════════════════════════════════
        # § 6  加权合并 → 最终入场得分
        #
        #   权重从配置读取，运行时自动归一化（无需保证配置里加和=1）。
        #   可通过调整权重改变各因子重要性，不影响阈值语义。
        # ══════════════════════════════════════════════════════════════════
        w_adx     = float(c.score_w_adx)
        w_atr     = float(getattr(c, "score_w_atr",     0.10))
        w_chop    = float(getattr(c, "score_w_chop",    0.15))
        w_mtf     = float(getattr(c, "score_w_mtf",     0.20))
        w_ema_pos = float(getattr(c, "score_w_ema_pos", 0.15))
        w_dist    = float(c.score_w_dist)
        w_rsi     = float(c.score_w_rsi)
        w_total   = max(1e-9, w_adx + w_atr + w_chop + w_mtf + w_ema_pos + w_dist + w_rsi)

        # ══════════════════════════════════════════════════════════════════
        # § 7  方向确认 & 阈值准入（按 Regime 分流）
        #
        #   Trend 路径：Supertrend 方向锚定 → 对该方向计算 §5 各因子分 → 合并
        #   Range 路径：均值回归专用评分（ADX 低 + EMA 偏离 + RSI 极端）
        # ══════════════════════════════════════════════════════════════════

        if regime == "range":
            # ── Range 子策略：均值回归多因子评分 ─────────────────────
            #   做多：价格显著低于 EMA_slow（超卖 → 期待回归均值）
            #   做空：价格显著高于 EMA_slow（超买 → 期待回归均值）

            # MR-a. ADX 低位分（震荡做法要求 ADX 不能过高）
            #   <= mr_adx_max (25) : 1.0（ADX 低，纯震荡最理想）
            #   > mr_adx_max       : 线性衰减至 0（趋势化风险增加）
            s_mr_adx = (_clip01(1.0 - (float(adx) - float(c.mr_adx_max)) / max(1e-9, 10.0))
                        if float(adx) > float(c.mr_adx_max) else 1.0)

            # MR-b. ATR 充裕分（防止波动太小无利润空间，逻辑同 5b）
            mr_atr_min = float(c.mr_atr_min_usdt)
            if float(atr) < mr_atr_min * 0.5:
                s_mr_atr = 0.0
            elif float(atr) < mr_atr_min:
                s_mr_atr = 0.5 * (float(atr) - mr_atr_min * 0.5) / max(1e-9, mr_atr_min * 0.5)
            else:
                s_mr_atr = _clip01(0.5 + 0.5 * (float(atr) - mr_atr_min) / max(1e-9, mr_atr_min))

            # MR-c. EMA 距离分（偏离 EMA_slow 越大，回归空间越充足）
            s_mr_dist = _clip01(
                (float(dist_atr) - float(c.mr_entry_dist_atr))
                / max(1e-9, float(c.mr_dist_span))
            )

            # MR-d. RSI 极端分（越超卖/超买，回归概率越高）
            s_mr_rsi_long  = _clip01((float(c.mr_rsi_long_max)  - float(rsi)) / max(1e-9, float(c.mr_rsi_span)))
            s_mr_rsi_short = _clip01((float(rsi) - float(c.mr_rsi_short_min)) / max(1e-9, float(c.mr_rsi_span)))

            # MR-e. EMA 方向位置分（做多期望价格低于 EMA_slow；做空期望高于）
            ema_slow_gap   = float(price) - float(ema_slow)
            s_mr_pos_long  = _clip01((-ema_slow_gap) / max(1e-9, _atr))
            s_mr_pos_short = _clip01(  ema_slow_gap  / max(1e-9, _atr))

            w_d  = float(c.mr_score_w_dist)
            w_r  = float(c.mr_score_w_rsi)
            w_ma = float(getattr(c, "mr_score_w_adx", 0.15))
            w_mt = float(getattr(c, "mr_score_w_atr", 0.10))
            w_mp = float(getattr(c, "mr_score_w_pos", 0.20))
            w_mr = max(1e-9, w_d + w_r + w_ma + w_mt + w_mp)

            mr_score_long  = (w_d * s_mr_dist + w_r * s_mr_rsi_long
                              + w_ma * s_mr_adx + w_mt * s_mr_atr + w_mp * s_mr_pos_long)  / w_mr
            mr_score_short = (w_d * s_mr_dist + w_r * s_mr_rsi_short
                              + w_ma * s_mr_adx + w_mt * s_mr_atr + w_mp * s_mr_pos_short) / w_mr

            # 宏观趋势过滤（仅作用于 Range/MR）：强下行不抄底做多；强上行不逆势做空
            mr_macro_block = float(getattr(c, "mr_macro_slope_block_usdt", 150.0))
            if (mr_macro_block > 0) and (not np.isnan(macro_slope)):
                if float(macro_slope) <= -mr_macro_block:
                    mr_score_long = -1.0
                elif float(macro_slope) >= mr_macro_block:
                    mr_score_short = -1.0

            thr_mr = float(c.mr_entry_score_threshold)
            if mr_score_long >= thr_mr:
                info["signal_direction"] = 1
                info["entry_score"]      = float(mr_score_long)
                info["alpha_type"]       = "range"
            elif (not bool(c.long_only)) and mr_score_short >= thr_mr:
                info["signal_direction"] = -1
                info["entry_score"]      = float(mr_score_short)
                info["alpha_type"]       = "range"

        else:
            # ── Trend-Follow 多因子评分 ───────────────────────────────
            #   Supertrend 作为方向锚：+1 → 考虑做多，-1 → 考虑做空。
            #   方向相关因子（MTF / EMA位置 / RSI中心）按方向参数化后统一计算。
            #   选分值更高的方向（通常每 bar Supertrend 只有一个方向）。

            # 追空末端保护（仅影响 Trend 做空）：极端超卖 + 宏观下行不足时禁止追空（#5）
            if float(st_dir) == -1.0:
                # 反弹追空保护：RSI 从超卖急反弹、且 1m/动量转正时，禁止“回抽一脚”追空
                # 目的：过滤 2/24 03:53 这类先跌后弹、指标一致但位置很差的追空单
                rb_win = max(1, int(getattr(c, "tf_short_rebound_window_bars", 20)))
                rb_min_rsi = float(getattr(c, "tf_short_rebound_min_rsi", 30.0))
                rb_curr_rsi = float(getattr(c, "tf_short_rebound_curr_rsi_min", 42.0))
                rb_ret_1m = float(getattr(c, "tf_short_rebound_ret_1m_min", 0.0005))
                rb_mom = float(getattr(c, "tf_short_rebound_mom_min", 0.0))

                rsi_col = f"rsi_{c.rsi_period}"
                rsi_min_recent = float("nan")
                try:
                    if rsi_col in df.columns and len(df) >= rb_win:
                        recent = df[rsi_col].iloc[-rb_win:]
                        rsi_min_recent = float(np.nanmin(recent.values))
                except Exception:
                    pass

                # 低位追空禁入：最近窗口出现过超卖（min RSI 低），且价格仍处于近 2 小时区间的低位
                # 目的：过滤 2/24 03:54、2/25 03:25 这类“接近局部低点还继续追空”的单子
                low_lb = max(1, int(getattr(c, "tf_short_lowpos_lookback_bars", 120)))
                low_pos_max = float(getattr(c, "tf_short_lowpos_max", 0.28))
                low_rsi_min = float(getattr(c, "tf_short_lowpos_min_rsi_recent", 30.0))
                try:
                    if (not np.isnan(rsi_min_recent)) and float(rsi_min_recent) <= low_rsi_min and len(df) >= low_lb:
                        lo = float(np.nanmin(df["low"].iloc[-low_lb:].values))
                        hi = float(np.nanmax(df["high"].iloc[-low_lb:].values))
                        if hi > lo:
                            pos_in_range = (float(price) - lo) / (hi - lo)
                            if pos_in_range <= low_pos_max:
                                return df, info
                except Exception:
                    pass

                if (
                    (not np.isnan(rsi_min_recent))
                    and float(rsi_min_recent) <= rb_min_rsi
                    and float(rsi) >= rb_curr_rsi
                    and (not np.isnan(ret_1m)) and float(ret_1m) >= rb_ret_1m
                    and (not np.isnan(mom)) and float(mom) >= rb_mom
                ):
                    return df, info

                ex_rsi_max = float(getattr(c, "tf_exhaust_rsi_short_max", 25.0))
                ex_slope_weak_down = float(getattr(c, "tf_exhaust_slope_weak_down_usdt", 150.0))
                if (float(rsi) <= ex_rsi_max) and (np.isnan(macro_slope) or float(macro_slope) > -ex_slope_weak_down):
                    return df, info

            best_dir   = 0
            best_score = float(c.entry_score_threshold) - 1e-9

            for direction in (1, -1):
                if direction == -1 and bool(c.long_only):
                    continue
                if float(st_dir) != float(direction):
                    continue  # Supertrend 方向不符，该方向不评分

                s_mtf_val     = _s_mtf(direction)
                s_ema_pos_val = _s_ema_pos(direction)
                rsi_center    = (float(c.rsi_score_center_long) if direction == 1
                                 else float(c.rsi_score_center_short))
                s_rsi_val     = _s_rsi(rsi_center)

                total_score = (
                    w_adx     * s_adx          +
                    w_atr     * s_atr           +
                    w_chop    * s_chop          +
                    w_mtf     * s_mtf_val       +
                    w_ema_pos * s_ema_pos_val   +
                    w_dist    * s_dist          +
                    w_rsi     * s_rsi_val
                ) / w_total

                if total_score > best_score:
                    best_dir   = direction
                    best_score = total_score

            if best_dir != 0:
                info["signal_direction"] = best_dir
                info["entry_score"]      = float(best_score)
                info["alpha_type"]       = "trend"

        return df, info

    def set_cooldown(self, bars: int) -> None:
        self._cooldown = max(0, int(bars))


# ===========================================================================
# ExitController — 出场信号 (TP/SL + 波动率控制)
# ===========================================================================

class ExitController:
    """
    出场信号（任一条件触发即 CLOSE）：
      1. TP/SL  — 保证金 ROI% 硬止盈止损（安全网）
      2. ATR 波动率飙升 — 当前 ATR > N × 入场 ATR，市场失控
    """

    def __init__(self, cfg: Any) -> None:
        self._c = cfg

    def reset(self) -> None:
        pass

    def __call__(self, df: pd.DataFrame, info: Dict) -> Tuple[pd.DataFrame, Dict]:
        ts: TradeState = info["trade_state"]
        curr_side = int(ts.last_side)

        if curr_side == 0:
            info["exit_signal"] = None
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

        # ── 1. TP/SL (ROI%) ──────────────────────────────────────────
        pnl_pct = (price - ts.entry_price) / ts.entry_price * curr_side
        roi = pnl_pct * max(1, ts.entry_leverage)

        alpha = getattr(ts, "entry_alpha_type", "trend")
        if alpha == "range":
            tp = float(c.range_tp_roi_pct) / 100.0
            sl = float(c.range_sl_roi_pct) / 100.0
        elif alpha == "chase":
            tp = float(c.chase_tp_roi_pct) / 100.0
            sl = float(c.chase_sl_roi_pct) / 100.0
        else:
            tp = float(c.tp_roi_pct) / 100.0
            sl = float(c.sl_roi_pct) / 100.0

        if roi >= tp:
            return close_sig
        if roi <= -sl:
            return close_sig

        if alpha == "chase" and ts.hold_bars >= int(c.chase_max_hold_bars):
            return close_sig


        # ── 3. ATR 波动率飙升 ─────────────────────────────────────────
        if (ts.entry_atr > 0 and not np.isnan(atr)
                and float(c.vol_spike_atr_mult) > 0):
            if atr > ts.entry_atr * float(c.vol_spike_atr_mult):
                return close_sig

        return None


# ===========================================================================
# PositionSizer — 仓位计算（margin_pct）
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
# AdaptiveTrendFusionStrategy — 配置 + 主流程
# ===========================================================================

class AdaptiveTrendFusionStrategy(Strategy):
    """
    Adaptive Trend Fusion — Supertrend + MTF EMA(5m/15m) + ADX + ATR trailing + TP/SL.

    入场：Supertrend(1m) 方向 + 5m/15m EMA 方向全部一致 + ADX + RSI 四重过滤
    出场：Hard TP/SL (ROI%) → EMA Structural → ATR Trailing Stop
    """

    # ── 策略元信息 ──────────────────────────────────────────────────────
    name:         str       = Field(default="adaptive_trend_fusion")
    description:  str       = Field(default="Supertrend + MTF EMA 趋势跟踪：减少假信号")
    factor_names: List[str] = Field(default=["ema", "rsi", "adx", "atr", "supertrend", "chop"])

    # ── 交易参数 ────────────────────────────────────────────────────────
    symbol:             str   = Field(default="BTCUSDT")
    leverage:           int   = Field(default=5)
    quantity:           float = Field(default=0.02)
    quantity_precision: int   = Field(default=3)
    min_quantity:       float = Field(default=0.001)
    long_only:          bool  = Field(default=False)
    fee_rate:           float = Field(default=DEFAULT_TAKER_FEE)

    # ── 仓位管理 ────────────────────────────────────────────────────────
    margin_pct:  float = Field(
        default=0.5,
        description="每笔保证金占净值比例；notional = equity × margin_pct × leverage",
    )
    max_leverage: int  = Field(default=10, description="杠杆上限，防止计算结果超出交易所限制")

    # ── 技术指标 ────────────────────────────────────────────────────────
    ema_entry_period:   int   = Field(default=100, description="入场确认 EMA（更敏感，用于更早确认趋势反转）")
    ema_slow_period:    int   = Field(default=200, description="EMA慢线，用于结构止损判断（更稳）")
    adx_period:         int   = Field(default=14)
    adx_threshold:      float = Field(default=30.0, description="ADX 最低趋势强度阈值，30+ 即可入场评分（原40太高导致做多score永远不够）")
    adx_gate_min:       float = Field(default=25.0, description="ADX gate 最低门槛；低于此值趋势太弱，直接禁止开仓")
    adx_rsi_bypass_min: float = Field(default=60.0, description="ADX 超过该值时旁路 RSI 过滤；极端趋势中 RSI 超买/超卖不再有反转含义")
    adx_score_span:     float = Field(default=20.0, description="ADX 评分跨度；(adx-adx_threshold)/span 映射到 0~1")
    rsi_period:         int   = Field(default=14)
    atr_period:         int   = Field(default=14)
    atr_min_usdt:            float = Field(default=45.0, description="ATR 最低阈值(USDT)；调低→更容易开仓（注意手续费）")
    # Fast chase uses ROC factor
    chase_mom_window:        int   = Field(default=5,   description="Fast chase 动量窗口（ROC window, 1m）")
    max_ema_dist_atr:         float = Field(default=3.0, description="开仓时 price 与 EMA_slow 最大允许距离(ATR倍数)；过度延伸(>3ATR)禁止追单")
    chop_period:             int   = Field(default=14,  description="CHOP 周期；通常用 14")
    chop_threshold:          float = Field(default=50.0, description="(兼容) 旧的 CHOP 阈值；当前实际使用 chop_trend_max")
    chop_trend_max:          float = Field(default=46.0, description="制度切换：trend 条件 chop<=该值（调高→趋势段更宽→交易更多）")
    chop_range_min:          float = Field(default=60.0, description="制度切换：range 条件 chop>=该值（调低→震荡段更宽→交易更多）")
    supertrend_period:       int   = Field(default=10,  description="Supertrend ATR 周期")
    supertrend_mult:         float = Field(default=3.0, description="Supertrend ATR 倍数（Binance/TV 常用 3）；越大翻转越少")
    supertrend_confirm_bars: int   = Field(default=3,   description="Supertrend 方向需连续保持 N 根 bar 才允许入场；调低→更频繁")
    mtf_5m_period:           int   = Field(default=20,  description="5m EMA 周期；严格模式：0=不通过")
    mtf_15m_period:          int   = Field(default=14,  description="15m EMA 周期；400根窗口÷15=26根，需 period+2≤26；严格模式：0=不通过")
    mtf_strict:              bool  = Field(default=False, description="MTF 过滤强度：True=5m/15m必须同向；False=5m同向且15m不反向（更频繁）")

    # ── 入场条件 ────────────────────────────────────────────────────────
    rsi_long_min:   float = Field(default=42.0, description="做多 RSI 下限（需有上行动能，避免超卖反弹假信号）")
    rsi_long_max:   float = Field(default=62.0, description="做多 RSI 上限（不追顶，RSI>62 时多头动能可能过热）")
    rsi_short_min:  float = Field(default=38.0, description="做空 RSI 下限（不追底，RSI<38 时空头动能可能衰竭）")
    rsi_short_max:  float = Field(default=58.0, description="做空 RSI 上限（需有下行动能）")

    # ── Gate + Score 入场评分 ───────────────────────────────────────────
    entry_score_threshold: float = Field(default=0.65, description="入场评分阈值（原0.75配合旧权重导致做多不可能达标）")
    score_w_adx:           float = Field(default=0.35, description="评分权重：ADX 趋势强度")
    score_w_atr:           float = Field(default=0.10, description="评分权重：ATR 波动充裕度（新增）")
    score_w_chop:          float = Field(default=0.15, description="评分权重：CHOP 趋势纯净度（新增，替代原 chop 硬 gate）")
    score_w_mtf:           float = Field(default=0.20, description="评分权重：MTF 方向一致性（新增，替代原 _mtf_ok 硬 gate）")
    score_w_ema_pos:       float = Field(default=0.15, description="评分权重：EMA 方向位置分（新增，替代原 price_above/below_ema 硬 gate）")
    score_w_dist:          float = Field(default=0.35, description="评分权重：EMA 距离甜点分")
    score_w_rsi:           float = Field(default=0.30, description="评分权重：RSI 动能健康度")
    rsi_score_center_long: float = Field(default=52.0, description="RSI 评分中心（做多）")
    rsi_score_center_short:float = Field(default=48.0, description="RSI 评分中心（做空）")
    rsi_score_span:        float = Field(default=18.0, description="RSI 评分跨度；|rsi-center|/span 映射到 0~1")

    # ── 宏观趋势/末端追单保护（定点修补 #4/#5）──────────────────────────
    macro_slope_lookback_bars: int = Field(default=60, description="EMA_entry 斜率回看 bars（宏观趋势强弱代理）")
    tf_exhaust_rsi_short_max: float = Field(
        default=25.0,
        description="趋势跟踪追空末端保护：RSI 低于该值视为极端超卖（配合宏观斜率过滤）",
    )
    tf_exhaust_slope_weak_down_usdt: float = Field(
        default=150.0,
        description="趋势跟踪追空末端保护：当 EMA_entry 在 lookback 内下行不足(斜率 > -该值) 时，禁止在极端超卖继续追空",
    )
    tf_short_rebound_window_bars: int = Field(
        default=20,
        description="趋势跟踪追空反弹保护：回看 RSI 最近 N 根的最小值，用于识别“超卖急反弹后追空”",
    )
    tf_short_rebound_min_rsi: float = Field(
        default=30.0,
        description="趋势跟踪追空反弹保护：若最近窗口内 RSI 低于该值，视为出现过超卖",
    )
    tf_short_rebound_curr_rsi_min: float = Field(
        default=42.0,
        description="趋势跟踪追空反弹保护：当前 RSI 高于该值，视为从超卖明显反弹",
    )
    tf_short_rebound_ret_1m_min: float = Field(
        default=0.0005,
        description="趋势跟踪追空反弹保护：1m 收盘回升幅度阈值（避免在反弹回抽中追空）",
    )
    tf_short_rebound_mom_min: float = Field(
        default=0.0,
        description="趋势跟踪追空反弹保护：动量(roc)为正的最小阈值；动量转正时更易出现反弹延续",
    )
    tf_short_lowpos_lookback_bars: int = Field(
        default=120,
        description="低位追空禁入：用于计算近 N 根(bar)的价格区间位置（默认 120=近2小时）",
    )
    tf_short_lowpos_max: float = Field(
        default=0.28,
        description="低位追空禁入：若价格处于近 N 根区间的底部 x 分位（<=该值），且近期出现过超卖，则禁止追空",
    )
    tf_short_lowpos_min_rsi_recent: float = Field(
        default=30.0,
        description="低位追空禁入：最近 rebound_window 内 RSI 最小值 <= 该阈值，视为出现过超卖",
    )

    # ── Range(震荡) 子策略参数：Mean-Reversion ─────────────────────────
    mr_adx_max:             float = Field(default=25.0, description="震荡子策略：ADX 上限；高于此值认为可能趋势化，禁止逆势均值回归")
    mr_atr_min_usdt:        float = Field(default=45.0, description="震荡子策略：ATR 最低阈值(USDT)")
    mr_entry_dist_atr:      float = Field(default=1.4, description="震荡子策略：入场需 price 偏离 EMA_slow 至少 N×ATR（调低→更频繁）")
    mr_entry_score_threshold: float = Field(default=0.65, description="震荡子策略：入场评分阈值（调低→更频繁）")
    mr_dist_span:           float = Field(default=2.0,  description="震荡子策略：距离评分跨度；(dist-阈值)/span → 0~1")
    mr_rsi_long_max:        float = Field(default=38.0, description="震荡做多：RSI 必须 <= 该值（超卖）")
    mr_rsi_short_min:       float = Field(default=62.0, description="震荡做空：RSI 必须 >= 该值（超买）")
    mr_rsi_span:            float = Field(default=12.0, description="震荡 RSI 评分跨度")
    mr_macro_slope_block_usdt: float = Field(
        default=150.0,
        description="震荡子策略宏观过滤：EMA_entry 斜率绝对值超过该值时，仅允许顺宏观方向的 MR（强下行不抄底做多）",
    )
    mr_score_w_dist:        float = Field(default=0.40, description="震荡评分权重：EMA 距离（回归空间）")
    mr_score_w_rsi:         float = Field(default=0.25, description="震荡评分权重：RSI 极端程度")
    mr_score_w_adx:         float = Field(default=0.15, description="震荡评分权重：ADX 低位分（新增，震荡要求 ADX 低）")
    mr_score_w_atr:         float = Field(default=0.10, description="震荡评分权重：ATR 充裕分（新增）")
    mr_score_w_pos:         float = Field(default=0.20, description="震荡评分权重：EMA_slow 方向位置分（新增，替代原 price>/<ema_slow 硬 gate）")

    # 震荡子策略出场 ROI（更小更快）
    range_tp_roi_pct:       float = Field(default=6.0, description="震荡子策略硬止盈：保证金 ROI %")
    range_sl_roi_pct:       float = Field(default=6.0, description="震荡子策略硬止损：保证金 ROI %")

    # ── Fast chase（快速追涨/追跌）参数：极端动量才触发 ────────────────
    chase_enabled:          bool  = Field(default=True, description="是否启用 fast chase（极端动量追涨/追跌）")
    chase_dist_atr_min:     float = Field(default=6.0,  description="触发门槛：|price-EMA|/ATR ≥ 该值 才允许追")
    chase_dist_span:        float = Field(default=6.0,  description="距离评分跨度")
    chase_mom_min_pct:      float = Field(default=0.006,description="触发门槛：|return_w| ≥ 该值（例如 0.6%/5min）")
    chase_mom_span:         float = Field(default=0.015,description="动量评分跨度")
    chase_ret_1m_min_pct:   float = Field(default=0.0035, description="触发门槛：|1m return| ≥ 该值（例如 0.35%/1min）")
    chase_ret_1m_span:      float = Field(default=0.01,   description="1m return 评分跨度")
    chase_down_1m_min_pct:  float = Field(default=0.0035, description="触发门槛：low/prev_close 下破幅度 ≥ 该值（崩盘插针）")
    chase_up_1m_min_pct:    float = Field(default=0.0035, description="触发门槛：high/prev_close 上破幅度 ≥ 该值（拉升插针）")
    chase_1m_span:          float = Field(default=0.01,   description="1m 插针幅度评分跨度")
    chase_use_adx:          bool  = Field(default=True,   description="追单是否使用 ADX 作为硬 gate；防止弱趋势时追单")
    chase_adx_min:          float = Field(default=25.0, description="追单时 ADX 最低门槛（chase_use_adx=True 时生效）")
    chase_rsi_long_max:     float = Field(default=75.0, description="追多 RSI 上限；超买极值禁止追涨")
    chase_rsi_short_min:    float = Field(default=25.0, description="追空 RSI 下限；超卖极值禁止追跌")
    chase_score_threshold:  float = Field(default=0.60, description="追单评分阈值（越高越少做）")
    chase_tp_roi_pct:       float = Field(default=12.0, description="追单硬止盈：保证金 ROI %（快进快出）")
    chase_sl_roi_pct:       float = Field(default=4.0,  description="追单硬止损：保证金 ROI %（更紧）")
    chase_max_hold_bars:    int   = Field(default=20,   description="追单最大持仓 bars（超时平仓）")

    # ── 出场参数 ────────────────────────────────────────────────────────
    tp_roi_pct:         float = Field(default=20.0,
        description="硬性止盈：保证金 ROI 上限 %（UPnL/margin × 100）；无 min_hold 约束；价格距离 = tp_roi_pct/100/leverage")
    sl_roi_pct:         float = Field(default=10.0,
        description="硬性止损：保证金 ROI 下限 %（UPnL/margin × 100）；价格距离 = sl_roi_pct/100/leverage")

    # ── 波动率控制出场 ──────────────────────────────────────────────────
    vol_spike_atr_mult: float = Field(default=3.5,
        description="ATR 波动率飙升退出：当前 ATR > 入场 ATR × 该倍数时平仓（市场失控）；0=禁用")

    # ── 风控 ────────────────────────────────────────────────────────────
    cooldown_bars: int = Field(default=120,
        description="平仓后冷却 bar 数（1m bar）；调低→更频繁进出（注意手续费）")

    # ── 兼容字段（_empty_result 依赖）──────────────────────────────────
    min_leverage: int = Field(default=1)

    # ==================================================================
    # 初始化
    # ==================================================================

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._trade_state     = TradeState()
        self._feature_engine  = FeatureEngine(self)
        self._enter_controller = EnterController(self)
        self._exit_controller = ExitController(self)
        self._position_sizer  = PositionSizer(self)

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

        # ── Step 0: 计算技术指标 ─────────────────────────────────────
        await factor_manager("ema", df, periods=[self.ema_entry_period, self.ema_slow_period])
        await factor_manager("rsi", df)
        await factor_manager("atr", df)
        await factor_manager("adx", df)
        await factor_manager("chop", df, periods=[self.chop_period])
        await factor_manager("roc", df, windows=[self.chase_mom_window])
        await factor_manager(
            "supertrend", df,
            periods=[self.supertrend_period],
            multipliers=[self.supertrend_mult],
        )

        # Warmup 检查
        min_bars = max(self.ema_entry_period, self.ema_slow_period, self.adx_period,
                       self.rsi_period, self.atr_period,
                       self.supertrend_period) + 10
        if df is None or df.empty or len(df) < min_bars:
            return self._empty_result()
        if PRICE_COL not in df.columns:
            return self._empty_result()

        # ── Step 1: 构建 info ────────────────────────────────────────
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
        ts        = self._trade_state
        info: Dict[str, Any] = {
            "current_pos":  current_pos,
            "equity":       equity,
            "curr_side":    curr_side,
            "trade_state":  ts,
            # 多时间框架 EMA 方向（+1 / -1 / 0=数据不足）
            "mtf_5m_dir":   _compute_mtf_ema_dir(df, "5min",  self.mtf_5m_period),
            "mtf_15m_dir":  _compute_mtf_ema_dir(df, "15min", self.mtf_15m_period),
        }

        # ── Step 2: 特征提取 ─────────────────────────────────────────
        df, info = self._feature_engine(df, info)
        ft    = info["features"]
        price = ft["price"]
        atr   = ft["atr"]
        chop  = ft.get("chop", float("nan"))

        # ── Regime: trend / range / transition ───────────────────────
        # 使用 CHOP 做制度切换：
        # - chop <= chop_trend_max   → 趋势（启用 trend-follow gate+score）
        # - chop >= chop_range_min   → 震荡（启用 mean-reversion 子策略）
        # - 中间区间                 → transition（不交易）
        regime = "n/a"
        try:
            if not np.isnan(float(chop)):
                if float(chop) <= float(self.chop_trend_max):
                    regime = "trend"
                elif float(chop) >= float(self.chop_range_min):
                    regime = "range"
                else:
                    regime = "transition"
        except Exception:
            regime = "n/a"
        info["regime"] = regime

        # ── Step 3: 更新 hold_bars / last_side ──────────────────────
        # _tpsl_closed: 检测到仓位被交易所 TP/SL 外部平掉（上一 bar 有仓，本 bar 无仓，
        # 但策略自身没有发出 CLOSE 信号），用于在 Step 6 补标 CLOSE 信号到日志/画图。
        _tpsl_closed = False
        if curr_side != 0:
            if ts.last_side == curr_side:
                ts.hold_bars += 1
            else:
                # 新持仓刚开（第一个有持仓的 bar）
                ts.hold_bars = 1
                ts.last_side = curr_side
                self._exit_controller.reset()
        else:
            if ts.last_side != 0:
                _tpsl_closed = True
                # 刚平仓 → 触发冷却，重置 exit 追踪
                self._enter_controller.set_cooldown(self.cooldown_bars)
                self._exit_controller.reset()
                ts.entry_alpha_type = "none"
                ts.entry_regime = "n/a"
                ts.entry_atr = 0.0
            ts.hold_bars = 0
            ts.last_side = 0

        # ── Step 4: 出场检查（持仓中）───────────────────────────────
        df, info = self._exit_controller(df, info)
        exit_sig = info.get("exit_signal")

        # ── Step 5: 入场信号（无持仓时）─────────────────────────────
        df, info = self._enter_controller(df, info)
        new_sig  = int(info.get("signal_direction", 0))

        # ── Step 6: 信号合并 + 构建订单 ─────────────────────────────
        signal_name  = "CLOSE" if _tpsl_closed else "HOLD"
        order: Optional[Dict] = None
        meta:  Dict[str, Any] = {}
        final_signal = 0

        if curr_side != 0 and exit_sig is not None:
            # 平仓
            close_side = "SELL" if curr_side == 1 else "BUY"
            close_qty  = round(abs(current_pos), self.quantity_precision)
            if close_qty > 0:
                order = {
                    "symbol":        self.symbol,
                    "side":          close_side,
                    "type":          "MARKET",
                    "quantity":      close_qty,
                    "position_side": "BOTH",
                    "reduce_only":   True,
                }
            signal_name  = "CLOSE"
            final_signal = 0

        elif curr_side == 0 and new_sig != 0:
            # 开仓
            df, info = self._position_sizer(df, info)
            sizing   = info["sizing"]
            qty      = sizing["quantity"]
            alpha_type = str(info.get("alpha_type") or "trend")
            ts.entry_alpha_type = alpha_type
            ts.entry_regime = str(info.get("regime") or "n/a")
            ts.entry_atr = float(ft["atr"]) if not np.isnan(ft["atr"]) else 0.0

            order = {
                "symbol":        self.symbol,
                "side":          "BUY" if new_sig == 1 else "SELL",
                "type":          "MARKET",
                "quantity":      qty,
                "position_side": "BOTH",
            }

            # 附加参考 TP/SL 价格（记录用，策略层出场以 ExitController 为准）
            # range/chase/trend 使用不同 ROI 参数
            _lev = max(1, int(self.leverage))
            if alpha_type == "range":
                _sl = float(self.range_sl_roi_pct)
                _tp = float(self.range_tp_roi_pct)
            elif alpha_type == "chase":
                _sl = float(self.chase_sl_roi_pct)
                _tp = float(self.chase_tp_roi_pct)
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
            meta.update({"sl_price": sl_price, "tp_price": tp_price,
                         "sl_distance": round(sl_dist, 2),
                         "tp_distance": round(tp_dist, 2)})
            meta["alpha_type"] = alpha_type

            meta["fee"]    = compute_fee_info(price, qty, 0.0, 0.0, self.fee_rate)
            ts.pending_fill   = True
            ts.entry_leverage = int(self.leverage)
            signal_name  = "LONG" if new_sig == 1 else "SHORT"
            final_signal = new_sig

        # 确保 sizing 存在（HOLD / CLOSE 路径）
        if "sizing" not in info:
            df, info = self._position_sizer(df, info)
        sizing = info["sizing"]

        # ── Step 7: 日志字段 ─────────────────────────────────────────
        st_tag = f"{self.supertrend_period}_{int(self.supertrend_mult * 10)}"
        factor_cols = [
            f"ema_{self.ema_entry_period}",
            f"ema_{self.ema_slow_period}",
            f"rsi_{self.rsi_period}",
            f"adx_{self.adx_period}",
            f"atr_{self.atr_period}",
            f"chop_{self.chop_period}",
            f"roc_{self.chase_mom_window}",
            f"supertrend_dir_{st_tag}",
        ]
        factors_json = build_factors_json(df, factor_cols, extra={
            "hold_bars":          ts.hold_bars,
            "cooldown_remaining": self._enter_controller._cooldown,
            "signal_dir":         final_signal,
            "entry_score":        info.get("entry_score"),
            "dist_atr":           info.get("dist_atr"),
            "momentum":           info.get("momentum"),
            "ret_1m":             info.get("ret_1m"),
            "down_1m":            info.get("down_1m"),
            "up_1m":              info.get("up_1m"),
            "regime":             info.get("regime"),
            "alpha_type":         info.get("alpha_type"),
            "mtf_5m_dir":         info.get("mtf_5m_dir",  0),
            "mtf_15m_dir":        info.get("mtf_15m_dir", 0),
        })
        meta["factors_json"] = factors_json

        current_atr = float(atr) if not np.isnan(atr) else 0.0

        return {
            "signal":       signal_name,
            "order":        order,
            "leverage":     int(self.leverage),
            "sizing":       sizing,
            "atr":          current_atr,
            "meta":         meta,
            "factors_json": factors_json,
            "risk": {
                "cooldown_remaining": self._enter_controller._cooldown,
                "consecutive_losses": 0,
                "loss_pause_bars":    0,
                "hold_bars":          ts.hold_bars,
                "regime":             str(info.get("regime") or "n/a"),
                "alpha_score":        float(info.get("entry_score") or 0.0),
                "alpha_type":         str(info.get("alpha_type") or ("trend" if final_signal != 0 else "none")),
            },
            "trailing_stop": {
                "active":        False,
                "should_update": False,
                "sl_price":      None,
                "best_price":    None,
                "entry_price":   ts.entry_price,
            },
            "decision_trace": [],
        }

    # ==================================================================
    # _empty_result — warmup / 异常路径返回
    # ==================================================================

    def _empty_result(
        self,
        info: Optional[Dict] = None,
        *,
        factors_json: Optional[str] = None,
    ) -> Dict[str, Any]:
        ts = self._trade_state
        return {
            "signal":   "HOLD",
            "order":    None,
            "leverage": self.min_leverage,
            "sizing":   {"quantity": 0.0, "leverage": self.min_leverage, "notional": 0.0},
            "atr":      0.0,
            "meta":     {},
            "factors_json": factors_json if factors_json is not None else "{}",
            "risk": {
                "cooldown_remaining": self._enter_controller._cooldown,
                "consecutive_losses": 0,
                "loss_pause_bars":    0,
                "hold_bars":          ts.hold_bars,
                "regime":             "n/a",
                "alpha_score":        0.0,
                "alpha_type":         "none",
            },
            "trailing_stop": {
                "active":        False,
                "should_update": False,
                "sl_price":      None,
                "best_price":    None,
                "entry_price":   ts.entry_price,
            },
            "decision_trace": [],
        }
