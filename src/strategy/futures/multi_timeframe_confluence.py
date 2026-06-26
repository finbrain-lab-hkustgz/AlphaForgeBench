"""
多时间框架共振策略 — Binance USDT-M 永续合约 (BTCUSDT)

一个平衡型高质量信号策略，通过多个时间框架的方向一致性过滤噪音。

设计理念: **高时间框架定方向，低时间框架精入场，策略层管理退出**
  - 将 1 分钟数据重采样到 5 分钟、15 分钟
  - 高时间框架（15 分钟）: 确定大趋势方向（EMA 排列 + ADX）
  - 中时间框架（5 分钟）: MACD 确认动量方向 + 动量递增 + 成交量过滤
  - 低时间框架（1 分钟）: RSI 回调入场 + EMA 交叉触发
  - 三个时间框架方向一致时才入场 → 极高的信号质量
  - 入场后策略层两阶段追踪止损管理退出（不截断趋势利润）
  - 每 bar 更新 EMA 交叉状态（确保状态不冻结）

关键优势:
  - 胜率远高于单时间框架策略（50-60% vs 40-50%）
  - 假突破过滤效果极佳
  - 信号稀少但精准（质量 > 数量）

信号生成:
  1. 15 分钟框架（方向过滤）:
     - EMA(20) > EMA(50) → 看多
     - EMA(20) < EMA(50) → 看空
     - ADX > 20 确认有趋势（否则不交易）
  2. 5 分钟框架（动量确认）:
     - MACD 柱状图方向与 15 分钟方向一致 + 动量递增
     - RSI 不在极端位置（避免追顶/追底）
     - 成交量 > 最近 20 bar 均值（缩量不入场）
  3. 1 分钟框架（精准入场）:
     - RSI 回调到顺势区域（做多: RSI 35-45; 做空: RSI 55-65）
     - 或 EMA(20) 与 EMA(50) 交叉

风控机制:
  - 杠杆: 5-7 倍
  - 名义价值: 10-20% 权益
  - 追踪止损: 盈利 >= 2×ATR 激活, 回撤 1.5×ATR 止盈（策略层管理）
  - 保底 TP: 4×ATR（追踪未激活时的极端跳空保护）
  - 高时间框架趋势反转 → 强制平仓
  - 最大持仓: 180 bar（3 小时）
  - 冷却: 30 bar（亏损）/ 15 bar（盈利）; 连续 3 亏损 → 暂停 300 bar
"""

from typing import List, Dict, Any, Optional
from pydantic import Field
import numpy as np
import pandas as pd

from src.strategy.types import Strategy
from src.factor import factor_manager
from src.strategy.futures._utils import (
    PRICE_COL, VOL_COL, DEFAULT_TAKER_FEE, safe_float,
    calc_position_size, build_order, compute_trailing_stop, build_factors_json,
)


class MultiTimeframeConfluenceStrategy(Strategy):
    """多时间框架共振策略 — 多 TF 方向一致 + 策略层追踪止损。"""

    # ── 策略元信息 ────────────────────────────────────────────
    name: str = Field(default="multi_timeframe_confluence", description="策略名称")
    description: str = Field(
        default="多时间框架共振策略 — 15m/5m/1m 三重确认 + 策略层追踪止损",
        description="策略描述",
    )
    factor_names: List[str] = Field(
        default=["ema", "rsi", "macd", "atr", "adx"],
        description="使用的因子",
    )

    # ── 交易参数 ──────────────────────────────────────────────
    symbol: str = Field(default="BTCUSDT", description="交易对")
    quantity_precision: int = Field(default=3, description="数量精度")
    min_quantity: float = Field(default=0.02, description="最小下单数量")

    # ── 仓位管理 ──────────────────────────────────────────────
    risk_per_trade: float = Field(default=0.02, description="每笔风险 2% 权益")
    atr_sl_multiplier: float = Field(default=2.0, description="交易所止损 = 2× ATR")
    atr_tp_multiplier: float = Field(
        default=4.0,
        description=(
            "保底 TP 阈值 = 4× ATR — 仅在追踪未激活时作为保底止盈"
            "（正常路径: 追踪在 2×ATR 激活，此值不触发）"
        ),
    )
    trailing_activation: float = Field(
        default=2.0,
        description="盈利 >= 2× ATR 后激活追踪止损",
    )
    trailing_atr_mult: float = Field(default=1.5, description="追踪止损间距 = 1.5× ATR")
    min_leverage: int = Field(default=5, description="最小杠杆")
    max_leverage: int = Field(default=10, description="最大杠杆 10 倍")
    min_notional_pct: float = Field(default=1.0, description="最小名义价值 100%（≈20%保证金@5x，统一仓位风格）")
    max_notional_pct: float = Field(default=1.0, description="最大名义价值 100%（固定）")
    fee_rate: float = Field(default=DEFAULT_TAKER_FEE, description="单边手续费率")

    # ── 多时间框架参数 ────────────────────────────────────────
    # 15 分钟框架
    htf_ema_fast: int = Field(default=20, description="15 分钟 EMA 快线")
    htf_ema_slow: int = Field(default=50, description="15 分钟 EMA 慢线")
    htf_adx_threshold: float = Field(default=20.0, description="15 分钟 ADX 趋势阈值")
    # 5 分钟框架
    mtf_rsi_period: int = Field(default=14, description="5 分钟 RSI 周期")
    mtf_rsi_extreme_low: float = Field(default=30.0, description="5 分钟 RSI 极端低位（避免追底）")
    mtf_rsi_extreme_high: float = Field(default=70.0, description="5 分钟 RSI 极端高位（避免追顶）")
    mtf_macd_rising: bool = Field(
        default=True,
        description="要求 MACD histogram 动量递增（做多递增/做空递减）",
    )
    mtf_vol_filter: bool = Field(
        default=True,
        description="5 分钟成交量过滤 — 要求 volume > 最近 20 bar 均值",
    )
    mtf_vol_lookback: int = Field(default=20, description="5 分钟成交量均值回看")
    # 1 分钟框架
    ltf_ema_fast: int = Field(default=20, description="1 分钟 EMA 快线")
    ltf_ema_slow: int = Field(default=50, description="1 分钟 EMA 慢线")
    ltf_rsi_period: int = Field(default=14, description="1 分钟 RSI 周期")
    ltf_rsi_pullback_long: tuple = Field(default=(35, 45), description="做多 RSI 回调区间")
    ltf_rsi_pullback_short: tuple = Field(default=(55, 65), description="做空 RSI 回调区间")
    atr_period: int = Field(default=14, description="ATR 周期（1 分钟）")
    adx_period: int = Field(default=14, description="ADX 周期")

    # ── 风控 ──────────────────────────────────────────────────
    min_hold_bars: int = Field(default=10, description="最小持仓 10 bar")
    max_hold_bars: int = Field(default=180, description="最大持仓 180 bar（3 小时）")
    cooldown_bars: int = Field(default=30, description="止损冷却 30 bar")
    win_cooldown_bars: int = Field(
        default=15,
        description="盈利冷却 15 bar",
    )
    max_consecutive_losses: int = Field(default=3, description="最多连续亏损")
    loss_pause_timeout: int = Field(default=300, description="暂停超时 300 bar")

    # ── 反向处理（从 adaptive 迁移）───────────────────────────
    flip_mode: str = Field(
        default="close_then_wait",
        description=(
            "反向信号处理模式: "
            "'flip' = 直接反手；"
            "'close_only' = 只平仓，不反手；"
            "'close_then_wait' = 只平仓，并等待 reverse_cooldown_bars 后才允许再次开仓（默认）。"
        ),
    )
    reverse_cooldown_bars: int = Field(
        default=60,
        description="close_then_wait 触发后，平仓后等待的 bar 数（1m 上默认 60 分钟）。",
    )

    # ── 轻量 no-chase（仅入场时）──────────────────────────────
    use_no_chase_filter: bool = Field(default=True, description="是否启用不追价过滤（仅入场时）")
    max_chase_pct: float = Field(
        default=0.005,
        description="相对 1m EMA20 最大追价偏离（0.005=0.5%）。超过则拒绝入场，等待回调。",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 状态
        self._hold_bars: int = 0
        self._last_side: int = 0
        self._cooldown_remaining: int = 0
        self._consecutive_losses: int = 0
        self._loss_pause_bars: int = 0
        self._last_position: float = 0.0
        self._entry_price: Optional[float] = None
        self._best_price: Optional[float] = None
        self._trailing_active: bool = False  # 追踪止损是否已激活
        self._last_htf_direction: int = 0  # 上一次 15 分钟方向
        self._last_ema_fast_1m: Optional[float] = None
        self._last_ema_slow_1m: Optional[float] = None
        self._reverse_cooldown_remaining: int = 0

    # ==================================================================
    # 数据重采样
    # ==================================================================

    @staticmethod
    def _resample(df: pd.DataFrame, rule: str) -> Optional[pd.DataFrame]:
        """将 1 分钟数据重采样到更高时间框架。"""
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df_copy = df.copy()
                    df_copy.index = pd.to_datetime(df_copy["timestamp"])
                elif "open_time" in df.columns:
                    df_copy = df.copy()
                    df_copy.index = pd.to_datetime(df_copy["open_time"])
                else:
                    # 尝试假设索引为时间戳
                    df_copy = df.copy()
                    try:
                        df_copy.index = pd.to_datetime(df_copy.index)
                    except Exception:
                        return None
            else:
                df_copy = df.copy()

            agg_map = {}
            if "open" in df_copy.columns:
                agg_map["open"] = "first"
            if "high" in df_copy.columns:
                agg_map["high"] = "max"
            if "low" in df_copy.columns:
                agg_map["low"] = "min"
            if "close" in df_copy.columns:
                agg_map["close"] = "last"
            if "volume" in df_copy.columns:
                agg_map["volume"] = "sum"

            if not agg_map:
                return None

            resampled = df_copy.resample(rule).agg(agg_map).dropna()
            return resampled if len(resampled) > 0 else None
        except Exception:
            return None

    # ==================================================================
    # 风控
    # ==================================================================

    def _update_risk_state(self, current_pos: float, price: float):
        """风控状态更新。"""
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
                self._loss_pause_bars = 0
            else:
                self._consecutive_losses = 0
                self._loss_pause_bars = 0  # 盈利时重置
                self._cooldown_remaining = self.win_cooldown_bars  # 盈利冷却

        if current_pos != 0 and self._last_position == 0:
            self._best_price = None
            self._trailing_active = False  # 新开仓重置
        elif old_exited:
            self._best_price = None
            self._entry_price = None
            self._trailing_active = False  # 平仓重置

        self._last_position = current_pos

    # ==================================================================
    # EMA 交叉状态管理
    # ==================================================================

    def _update_ema_state(self, df: pd.DataFrame) -> None:
        """每 bar 更新 1m EMA 交叉检测状态。

        确保 _last_ema_* 每 bar 更新（而非仅在 _analyze_ltf 调用时），
        当 HTF/MTF 不满足条件导致 _analyze_ltf 长期不被调用时，
        状态冻结 → 下次调用时产生虚假交叉或漏检。
        """
        ema_fast_col = f"ema_{self.ltf_ema_fast}"
        ema_slow_col = f"ema_{self.ltf_ema_slow}"
        if ema_fast_col in df.columns and ema_slow_col in df.columns:
            self._last_ema_fast_1m = safe_float(df[ema_fast_col].iloc[-1])
            self._last_ema_slow_1m = safe_float(df[ema_slow_col].iloc[-1])

    # ==================================================================
    # 多时间框架分析
    # ==================================================================

    async def _analyze_htf(self, df_15m: pd.DataFrame) -> int:
        """15 分钟框架分析 — 确定大方向。返回 +1/-1/0。"""
        if df_15m is None or len(df_15m) < 60:
            return 0

        # 通过 factor_manager 获取因子实例并应用于重采样数据（动态传入周期）
        ema_factor = await factor_manager.get("ema")
        adx_factor = await factor_manager.get("adx")
        if ema_factor:
            await ema_factor(df_15m, periods=[self.htf_ema_fast, self.htf_ema_slow])
        if adx_factor:
            await adx_factor(df_15m, periods=[self.adx_period])

        ema_fast = safe_float(df_15m[f"ema_{self.htf_ema_fast}"].iloc[-1])
        ema_slow = safe_float(df_15m[f"ema_{self.htf_ema_slow}"].iloc[-1])
        # 使用参数化列名
        adx = safe_float(df_15m[f"adx_{self.adx_period}"].iloc[-1], 0.0)

        if adx < self.htf_adx_threshold:
            return 0  # 无趋势

        if ema_fast > ema_slow * 1.001:
            return 1  # 看多
        elif ema_fast < ema_slow * 0.999:
            return -1  # 看空
        return 0

    async def _analyze_mtf(self, df_5m: pd.DataFrame, htf_direction: int) -> bool:
        """5 分钟框架分析 — 确认动量方向。返回是否确认。"""
        if df_5m is None or len(df_5m) < 30:
            return False

        # 通过 factor_manager 获取因子实例并应用于重采样数据（动态传入周期）
        macd_factor = await factor_manager.get("macd")
        rsi_factor = await factor_manager.get("rsi")
        if macd_factor:
            await macd_factor(df_5m)
        if rsi_factor:
            await rsi_factor(df_5m, periods=[self.mtf_rsi_period])

        macd_hist = safe_float(df_5m["macd_hist"].iloc[-1], 0.0)
        macd_hist_prev = (
            safe_float(df_5m["macd_hist"].iloc[-2], 0.0) if len(df_5m) > 1 else 0.0
        )
        # 使用参数化列名
        rsi_col = f"rsi_{self.mtf_rsi_period}"
        rsi = safe_float(df_5m[rsi_col].iloc[-1], 50.0) if rsi_col in df_5m.columns else 50.0

        # MACD 方向一致
        if htf_direction == 1 and macd_hist <= 0:
            return False
        if htf_direction == -1 and macd_hist >= 0:
            return False

        # MACD 动量递增 — 要求 histogram 朝趋势方向加强
        if self.mtf_macd_rising:
            if htf_direction == 1 and macd_hist < macd_hist_prev:
                return False  # 做多但 histogram 递减 → 动量衰减
            if htf_direction == -1 and macd_hist > macd_hist_prev:
                return False  # 做空但 histogram 递增 → 动量衰减

        # RSI 不在极端位置
        if htf_direction == 1 and rsi > self.mtf_rsi_extreme_high:
            return False  # 已经超买，不追多
        if htf_direction == -1 and rsi < self.mtf_rsi_extreme_low:
            return False  # 已经超卖，不追空

        # 成交量过滤 — 缩量趋势不可靠
        if self.mtf_vol_filter and "volume" in df_5m.columns:
            volumes = df_5m["volume"].astype(float)
            if len(volumes) > self.mtf_vol_lookback:
                vol_avg = float(volumes.iloc[-self.mtf_vol_lookback - 1:-1].mean())
                cur_vol = safe_float(volumes.iloc[-1])
                if vol_avg > 0 and cur_vol < vol_avg:
                    return False  # 当前 5m 成交量低于均值 → 趋势不可靠

        return True

    def _analyze_ltf(self, df: pd.DataFrame, htf_direction: int) -> bool:
        """1 分钟框架分析 — 精准入场触发。

        EMA 状态由 _update_ema_state() 每 bar 更新，
        此处仅读取最新值并检测交叉，不再更新 _last_ema_*。
        """
        rsi = safe_float(df[f"rsi_{self.ltf_rsi_period}"].iloc[-1], 50.0)
        ema_fast = safe_float(df[f"ema_{self.ltf_ema_fast}"].iloc[-1])
        ema_slow = safe_float(df[f"ema_{self.ltf_ema_slow}"].iloc[-1])

        # 条件 1: RSI 回调到顺势区域
        rsi_pullback = False
        if htf_direction == 1:
            low, high = self.ltf_rsi_pullback_long
            rsi_pullback = low <= rsi <= high
        elif htf_direction == -1:
            low, high = self.ltf_rsi_pullback_short
            rsi_pullback = low <= rsi <= high

        # 条件 2: EMA 交叉（基于 _last_ema_* 状态，每 bar 由 _update_ema_state 更新）
        ema_cross = False
        if self._last_ema_fast_1m is not None and self._last_ema_slow_1m is not None:
            prev_fast_above = self._last_ema_fast_1m > self._last_ema_slow_1m
            curr_fast_above = ema_fast > ema_slow
            if htf_direction == 1 and not prev_fast_above and curr_fast_above:
                ema_cross = True
            if htf_direction == -1 and prev_fast_above and not curr_fast_above:
                ema_cross = True

        # 不在此处更新 _last_ema_*，由 _update_ema_state() 统一管理

        return rsi_pullback or ema_cross

    # ==================================================================
    # 信号
    # ==================================================================

    async def _generate_signal(
        self, df: pd.DataFrame, current_pos: float,
    ) -> float:
        """多时间框架共振信号。

        核心机制:
          - 策略层两阶段追踪止损退出（Phase 1 激活 → Phase 2 追踪回撤）
          - 保底 TP 防跳空
        """
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)

        # ── 反向平仓后的等待期（close_then_wait）──────────────────
        if curr_side == 0 and self._reverse_cooldown_remaining > 0:
            self._reverse_cooldown_remaining -= 1
            return 0.0

        # ── 冷却/暂停 ────────────────────────────────────────
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return 0.0

        if self._consecutive_losses >= self.max_consecutive_losses:
            self._loss_pause_bars += 1
            if self._loss_pause_bars >= self.loss_pause_timeout:
                self._consecutive_losses = 0
                self._loss_pause_bars = 0
            else:
                return 0.0

        # ── 读取 1m 价格/ATR ─────────────────────────────────
        price = safe_float(df[PRICE_COL].iloc[-1])
        atr = safe_float(df[f"atr_{self.atr_period}"].iloc[-1])

        # ── 高时间框架分析（持仓退出 + 入场都需要）────────────
        df_15m = self._resample(df, "15min")
        htf_direction = await self._analyze_htf(df_15m)

        # ── 持仓管理（策略层追踪止损）────────────────────
        if curr_side != 0:
            if curr_side == self._last_side:
                self._hold_bars += 1
            else:
                self._hold_bars = 1
                self._last_side = curr_side

            # (1) 最大持仓时间
            if self._hold_bars >= self.max_hold_bars:
                return float(-curr_side)

            # (2) 高时间框架趋势反转 → 平仓（需满足最小持仓）
            if htf_direction != 0 and htf_direction != curr_side:
                if self._hold_bars >= self.min_hold_bars:
                    return float(-curr_side)

            # (3) 两阶段追踪止损 — 策略层自主管理
            if self._entry_price is not None and atr > 0 and price > 0:
                if curr_side == 1:
                    unrealized = price - self._entry_price
                else:
                    unrealized = self._entry_price - price

                if not self._trailing_active:
                    # Phase 1: 等待追踪激活
                    if unrealized >= self.trailing_activation * atr:
                        self._trailing_active = True
                        # 保留 compute_trailing_stop 已维护的历史最优价
                        # 不覆盖 — best_price 可能已高于当前价（冲高回落到激活线）
                        if self._best_price is None:
                            self._best_price = price
                    elif unrealized >= self.atr_tp_multiplier * atr:
                        # 保底 TP: 追踪未激活但利润已极高（跳空等）
                        if self._hold_bars >= self.min_hold_bars:
                            return float(-curr_side)

                if self._trailing_active:
                    # Phase 2: 追踪止损管理
                    if curr_side == 1:
                        if self._best_price is None or price > self._best_price:
                            self._best_price = price
                        trail_sl = self._best_price - self.trailing_atr_mult * atr
                        if price <= trail_sl and self._hold_bars >= self.min_hold_bars:
                            return float(-curr_side)  # 追踪止损触发
                    else:
                        if self._best_price is None or price < self._best_price:
                            self._best_price = price
                        trail_sl = self._best_price + self.trailing_atr_mult * atr
                        if price >= trail_sl and self._hold_bars >= self.min_hold_bars:
                            return float(-curr_side)  # 追踪止损触发

            return 0.0

        # ── 无趋势不交易 ─────────────────────────────────────
        if htf_direction == 0:
            return 0.0

        # ── 5 分钟确认（延迟重采样 — 持仓期间不需要 5m 数据）──
        df_5m = self._resample(df, "5min")
        mtf_confirmed = await self._analyze_mtf(df_5m, htf_direction)
        if not mtf_confirmed:
            return 0.0

        # ── 1 分钟入场触发 ───────────────────────────────────
        ltf_triggered = self._analyze_ltf(df, htf_direction)
        if not ltf_triggered:
            return 0.0

        # ── 轻量不追价过滤（只对空仓入场）────────────────────────
        if self.use_no_chase_filter:
            ema_col = f"ema_{self.ltf_ema_fast}"
            ema_val = safe_float(df[ema_col].iloc[-1]) if ema_col in df.columns else 0.0
            if ema_val > 0 and price > 0:
                if htf_direction == 1:
                    chase = (price - ema_val) / ema_val
                else:
                    chase = (ema_val - price) / ema_val
                if chase > self.max_chase_pct:
                    return 0.0

        # ── 三重确认通过 → 入场 ──────────────────────────────
        self._hold_bars = 0
        self._last_side = htf_direction
        self._last_htf_direction = htf_direction
        self._trailing_active = False
        return float(htf_direction)

    # ==================================================================
    # __call__
    # ==================================================================

    async def __call__(
        self,
        df: pd.DataFrame,
        current_pos: float = 0.0,
        equity: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """多时间框架共振策略主入口。"""
        # 0. 通过 factor_manager 计算 1 分钟因子（动态传入周期）
        await factor_manager("ema", df, periods=[self.ltf_ema_fast, self.ltf_ema_slow])
        await factor_manager("rsi", df, periods=[self.ltf_rsi_period])
        await factor_manager("atr", df, periods=[self.atr_period])

        price = safe_float(df[PRICE_COL].iloc[-1])
        atr = safe_float(df[f"atr_{self.atr_period}"].iloc[-1])

        # 1. 风控
        self._update_risk_state(current_pos, price)

        # 2. 信号（包含多时间框架分析 + 追踪止损管理）
        signal = await self._generate_signal(df, current_pos)
        sig_int = int(np.sign(signal)) if signal != 0 else 0

        # 3. 每 bar 更新 EMA 交叉检测状态（必须在 _generate_signal 之后！）
        # 注意: 此步必须在信号之后执行，确保 _last_ema_* 保存的是当前 bar 值，
        # 供下一 bar 的 _analyze_ltf 做交叉比较（当前 bar 与前一 bar 比较）。
        self._update_ema_state(df)

        # 4. 仓位
        sizing = calc_position_size(
            equity=equity, price=price, atr=atr,
            risk_pct=self.risk_per_trade, atr_sl_mult=self.atr_sl_multiplier,
            min_leverage=self.min_leverage, max_leverage=self.max_leverage,
            min_notional_pct=self.min_notional_pct, max_notional_pct=self.max_notional_pct,
            min_quantity=self.min_quantity, quantity_precision=self.quantity_precision,
        )

        # 5. 订单（策略层管理追踪止损 — 不设交易所 TP）
        result = build_order(
            symbol=self.symbol, signal=sig_int, current_pos=current_pos,
            sizing=sizing, price=price, atr=atr,
            atr_sl_multiplier=self.atr_sl_multiplier,
            atr_tp_multiplier=self.atr_tp_multiplier,
            use_trailing_stop=True,  # 不设交易所 TP, 策略管理退出
            flip_mode=self.flip_mode,
            fee_rate=self.fee_rate,
            quantity_precision=self.quantity_precision,
        )

        if result["signal"] == "CLOSE" and self.flip_mode == "close_then_wait":
            self._reverse_cooldown_remaining = int(self.reverse_cooldown_bars)

        if result["signal"] in ("LONG", "SHORT"):
            self._entry_price = price
            self._best_price = price

        # 6. 追踪止损信息（供调用方监控）
        effective_pos = current_pos
        if result["signal"] == "LONG":
            effective_pos = sizing["quantity"]
        elif result["signal"] == "SHORT":
            effective_pos = -sizing["quantity"]

        trailing_info, self._best_price = compute_trailing_stop(
            effective_pos, price, atr, self.trailing_atr_mult,
            self._entry_price, self._best_price,
        )
        trailing_info["trailing_active"] = self._trailing_active

        factors_json = build_factors_json(df, [
            f"ema_{self.ltf_ema_fast}", f"ema_{self.ltf_ema_slow}",
            f"rsi_{self.ltf_rsi_period}",
            f"atr_{self.atr_period}",
        ], extra={
            "htf_direction": self._last_htf_direction,
            "hold_bars": self._hold_bars,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_remaining": self._cooldown_remaining,
        })

        meta = {
            **result["meta"],
            "htf_direction": self._last_htf_direction,
            "factors_json": factors_json,
        }

        return {
            "signal": result["signal"],
            "order": result["order"],
            "leverage": sizing["leverage"],
            "sizing": sizing,
            "atr": atr,
            "meta": meta,
            "factors_json": factors_json,
            "risk": {
                "cooldown_remaining": self._cooldown_remaining,
                "consecutive_losses": self._consecutive_losses,
                "loss_pause_bars": self._loss_pause_bars,
                "hold_bars": self._hold_bars,
            },
            "trailing_stop": trailing_info,
        }
