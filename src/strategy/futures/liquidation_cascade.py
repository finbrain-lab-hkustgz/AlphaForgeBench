"""
清算猎杀策略 — Binance USDT-M 永续合约 (BTCUSDT)

一个适度激进的事件驱动策略，检测清算瀑布后等待回调顺势入场。

设计理念: **检测清算瀑布，回调入场顺势交易**

  Phase 1 — 瀑布检测（标记事件，不入场）:
    - 成交量爆发（3×+ 均值）
    - 当前 bar 波幅 > 2× ATR（价格剧烈运动）
    - 价格加速: |ROC| > 阈值
    - Pin Bar 检测（长影线是清算的标志性形态）
    - 至少 1 项确认: Pin bar / OBV 方向 / 连续同向 K 线

  Phase 2 — 回调入场:
    - 等待价格回调至瀑布幅度的 25%-70%
    - 在回调处顺势入场（买在回调低点 / 卖在反弹高点）
    - 超时 20 bar 或回调过深（>70%）→ 取消信号

  优势: 入场在回调处而非瀑布尾部
    - 更好的价格 → 更高的风险回报比
    - 瀑布方向已确认 → 更高的胜率
    - 自然的止损位（瀑布起点）→ 更合理的风控

风控:
  - 杠杆: 5-8 倍，名义价值 10-15% 权益
  - SL: 2.5× ATR（宽止损适应瀑布后高波动）
  - TP: 3× ATR
  - 最大持仓: 20 bar
  - 盈利冷却: 15 bar / 亏损冷却: 60 bar
  - 连续 2 次亏损 → 暂停 300 bar
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


class LiquidationCascadeStrategy(Strategy):
    """清算猎杀策略 — 两阶段入场: 瀑布检测 → 回调入场。"""

    # ── 策略元信息 ────────────────────────────────────────────
    name: str = Field(default="liquidation_cascade", description="策略名称")
    description: str = Field(
        default="清算猎杀策略 — 检测级联清算后等待回调顺势入场",
        description="策略描述",
    )
    factor_names: List[str] = Field(
        default=["roc", "atr", "obv"],
        description="使用的因子",
    )

    # ── 交易参数 ──────────────────────────────────────────────
    symbol: str = Field(default="BTCUSDT", description="交易对")
    quantity_precision: int = Field(default=3, description="数量精度")
    min_quantity: float = Field(default=0.02, description="最小下单数量")

    # ── 仓位管理 ──────────────────────────────────────────────
    risk_per_trade: float = Field(default=0.015, description="每笔风险 1.5% 权益")
    atr_sl_multiplier: float = Field(
        default=2.5,
        description="止损 = 2.5× ATR（宽止损 — 瀑布后波动率远超正常，1× ATR 必被扫）",
    )
    atr_tp_multiplier: float = Field(default=3.0, description="止盈 = 3× ATR")
    min_leverage: int = Field(default=5, description="最小杠杆")
    max_leverage: int = Field(default=10, description="最大杠杆 10 倍")
    min_notional_pct: float = Field(default=1.0, description="最小名义价值 100%（≈20%保证金@5x，统一仓位风格）")
    max_notional_pct: float = Field(default=1.0, description="最大名义价值 100%（固定）")
    fee_rate: float = Field(default=DEFAULT_TAKER_FEE, description="单边手续费率")

    # ── 瀑布检测参数 ──────────────────────────────────────────
    vol_lookback: int = Field(default=20, description="成交量均值回看")
    vol_explosion_mult: float = Field(
        default=3.0,
        description="成交量爆发倍数（3× = 非常严格，避免误判）",
    )
    roc_period: int = Field(
        default=5,
        description="ROC 周期（使用 factor_manager 的 roc_5）",
    )
    roc_threshold: float = Field(
        default=0.15,
        description="价格变动阈值（%）— 5 bar 内变动需超此值",
    )
    bar_range_atr_mult: float = Field(
        default=2.0,
        description="Bar 波幅需 > 2× ATR — 确认价格剧烈运动（新增）",
    )
    consecutive_bars: int = Field(
        default=3,
        description="连续同方向 K 线数量（方向确认条件之一）",
    )
    atr_period: int = Field(default=14, description="ATR 周期")
    obv_trend_period: int = Field(default=10, description="OBV 趋势比较周期")
    pin_bar_range_pct: float = Field(
        default=0.6,
        description="Pin bar 备用检测: 影线 >= 总波幅 × 此比例（处理十字星等小 body 情况）",
    )
    pin_bar_wick_ratio: float = Field(
        default=2.0,
        description="Pin bar: 影线 >= body × 此倍数（新增: 清算标志性形态）",
    )

    # ── 回调入场参数（新增）────────────────────────────────────
    pullback_min_pct: float = Field(
        default=0.25,
        description="最小回调比例 25% — 至少回调瀑布幅度的 25% 才入场",
    )
    pullback_max_pct: float = Field(
        default=0.70,
        description="最大回调比例 70% — 超过视为瀑布失败，取消信号",
    )
    pullback_timeout_bars: int = Field(
        default=20,
        description="回调等待超时 20 bar — 超时未回调则取消信号",
    )

    # ── 风控 ──────────────────────────────────────────────────
    max_hold_bars: int = Field(default=20, description="最大持仓 20 bar")
    cooldown_bars: int = Field(default=60, description="亏损冷却 60 bar")
    win_cooldown_bars: int = Field(
        default=15,
        description="盈利冷却 15 bar（新增: 瀑布后市场不稳定，避免立刻重入）",
    )
    max_consecutive_losses: int = Field(default=2, description="最多连续 2 次亏损")
    loss_pause_timeout: int = Field(default=300, description="暂停 300 bar")

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

    # ── 瀑布后稳定性过滤（减少二次扫损）──────────────────────────
    stabilize_min_wait_bars: int = Field(
        default=2,
        description="瀑布检测后至少等待 N 根 bar 才允许回调入场（让市场先冷却）。",
    )
    stabilize_vol_max_mult: float = Field(
        default=2.2,
        description="回调入场时要求当前成交量不超过最近均值×此倍数（过滤仍在剧烈波动期）。",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 持仓状态
        self._hold_bars: int = 0
        self._last_side: int = 0
        self._cooldown_remaining: int = 0
        self._consecutive_losses: int = 0
        self._loss_pause_bars: int = 0
        self._last_position: float = 0.0
        self._entry_price: Optional[float] = None
        self._best_price: Optional[float] = None
        # 瀑布状态（两阶段入场）
        self._cascade_direction: int = 0       # 0=未检测, +1=看涨, -1=看跌
        self._cascade_peak: float = 0.0        # 瀑布极值（涨=high, 跌=low）
        self._cascade_origin: float = 0.0      # 瀑布起始价格
        self._cascade_range: float = 0.0       # 瀑布幅度 |peak - origin|
        self._pullback_wait_bars: int = 0      # 已等待回调 bar 数
        self._reverse_cooldown_remaining: int = 0

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
                self._loss_pause_bars = 0          # 修复: 盈利时也重置
                self._cooldown_remaining = self.win_cooldown_bars  # 盈利后短冷却

        if current_pos != 0 and self._last_position == 0:
            self._best_price = None
        elif old_exited:
            self._best_price = None
            self._entry_price = None
            # 平仓后清除瀑布等待状态
            self._cascade_direction = 0

        self._last_position = current_pos

    # ==================================================================
    # Phase 1: 瀑布检测
    # ==================================================================

    def _detect_cascade(self, df: pd.DataFrame) -> int:
        """检测级联清算瀑布。返回方向 (+1/-1) 或 0。

        仅标记瀑布事件，不直接入场。

        检测条件（全部必须满足）:
          1. 成交量爆发: volume >= vol_explosion_mult × avg
          2. Bar 波幅爆发: (high - low) >= bar_range_atr_mult × ATR
          3. 价格加速: 利用 ROC 因子, |price_change| > roc_threshold

        方向确认（至少 1 项通过）:
          A. Pin bar（长影线 — 清算标志性形态）
          B. OBV 方向一致（资金流确认）
          C. 连续同方向 K 线
        """
        min_len = max(self.vol_lookback + 5, self.consecutive_bars + 1, 20)
        if len(df) < min_len:
            return 0

        price = safe_float(df[PRICE_COL].iloc[-1])
        if price <= 0:
            return 0

        # ── 条件 1: 成交量爆发 ────────────────────────────────
        if VOL_COL not in df.columns:
            return 0

        volumes = df[VOL_COL].astype(float)
        current_vol = safe_float(volumes.iloc[-1])
        vol_avg = volumes.iloc[-self.vol_lookback - 1:-1].mean()

        if vol_avg <= 0 or current_vol < vol_avg * self.vol_explosion_mult:
            return 0

        # ── 条件 2: Bar 波幅爆发 ──────────────────────────────
        atr = safe_float(df[f"atr_{self.atr_period}"].iloc[-1])
        if atr <= 0:
            return 0

        high = safe_float(df["high"].iloc[-1]) if "high" in df.columns else price
        low = safe_float(df["low"].iloc[-1]) if "low" in df.columns else price
        bar_range = high - low

        if bar_range < atr * self.bar_range_atr_mult:
            return 0

        # ── 条件 3: 价格加速（使用 ROC 因子）──────────────────
        # ROC = past_price / current_price
        # price_change_pct = (1 - roc) × 100 (正=涨, 负=跌)
        roc_col = f"roc_{self.roc_period}"
        if roc_col in df.columns:
            roc_val = safe_float(df[roc_col].iloc[-1], 1.0)
            price_change_pct = (1.0 - roc_val) * 100
        else:
            closes = df[PRICE_COL].astype(float)
            if len(closes) > self.roc_period:
                prev = safe_float(closes.iloc[-1 - self.roc_period])
                price_change_pct = (price - prev) / prev * 100 if prev > 0 else 0.0
            else:
                price_change_pct = 0.0

        if abs(price_change_pct) < self.roc_threshold:
            return 0

        direction = 1 if price_change_pct > 0 else -1

        # ── 方向确认（至少 1 项通过）──────────────────────────
        confirmations = 0

        # 确认 A: Pin bar（清算标志性形态）
        if "open" in df.columns:
            open_price = safe_float(df["open"].iloc[-1])
            body = abs(price - open_price)
            upper_wick = high - max(open_price, price)
            lower_wick = min(open_price, price) - low
            total_range = high - low
            has_pin = False

            if body > 0:
                # 标准 pin bar: 影线 >= body × wick_ratio
                if direction == 1 and lower_wick >= body * self.pin_bar_wick_ratio:
                    has_pin = True
                if direction == -1 and upper_wick >= body * self.pin_bar_wick_ratio:
                    has_pin = True

            # 备用: 十字星/极小 body — 影线 >= 总波幅 × pin_bar_range_pct
            # 蜻蜓十字星/墓碑十字星是最强的清算信号之一
            if not has_pin and total_range > 0:
                if direction == 1 and lower_wick >= total_range * self.pin_bar_range_pct:
                    has_pin = True
                if direction == -1 and upper_wick >= total_range * self.pin_bar_range_pct:
                    has_pin = True

            if has_pin:
                confirmations += 1

        # 确认 B: OBV 方向一致
        if "obv" in df.columns:
            obv = df["obv"].astype(float)
            if len(obv) > self.obv_trend_period:
                obv_change = obv.iloc[-1] - obv.iloc[-self.obv_trend_period]
                if direction == 1 and obv_change > 0:
                    confirmations += 1
                elif direction == -1 and obv_change < 0:
                    confirmations += 1

        # 确认 C: 连续同方向 K 线
        closes = df[PRICE_COL].astype(float)
        if len(closes) >= self.consecutive_bars + 1:
            consecutive = True
            for i in range(-self.consecutive_bars, 0):
                bar_change = closes.iloc[i] - closes.iloc[i - 1]
                if direction == 1 and bar_change <= 0:
                    consecutive = False
                    break
                if direction == -1 and bar_change >= 0:
                    consecutive = False
                    break
            if consecutive:
                confirmations += 1

        if confirmations < 1:
            return 0

        return direction

    # ==================================================================
    # Phase 2: 回调入场
    # ==================================================================

    def _check_pullback_entry(self, df: pd.DataFrame) -> int:
        """检查已检测的瀑布是否产生了合适的回调入场机会。

        看涨瀑布（direction=+1）:
          瀑布把价格从 origin 推到 peak（high）。
          回调 = 价格从 peak 回落。
          当回落幅度达到瀑布幅度的 25%-70% 时入场做多。

        看跌瀑布（direction=-1）:
          瀑布把价格从 origin 砸到 peak（low）。
          回调 = 价格从 peak 反弹。
          当反弹幅度达到瀑布幅度的 25%-70% 时入场做空。

        返回: 入场方向 (+1/-1) 或 0（继续等待/已取消）。
        """
        if self._cascade_direction == 0:
            return 0

        self._pullback_wait_bars += 1
        price = safe_float(df[PRICE_COL].iloc[-1])

        # 稳定性过滤：至少等待 N 根 bar，且成交量不再处于极端爆发
        if self._pullback_wait_bars < self.stabilize_min_wait_bars:
            return 0
        if VOL_COL in df.columns and len(df) > self.vol_lookback + 1:
            volumes = df[VOL_COL].astype(float)
            curr_vol = safe_float(volumes.iloc[-1])
            vol_avg = float(volumes.iloc[-self.vol_lookback - 1:-1].mean())
            if vol_avg > 0 and curr_vol > vol_avg * self.stabilize_vol_max_mult:
                return 0

        # 超时 → 取消
        if self._pullback_wait_bars >= self.pullback_timeout_bars:
            self._cascade_direction = 0
            return 0

        if self._cascade_range <= 0:
            self._cascade_direction = 0
            return 0

        # 更新 peak: 如果瀑布在检测后继续延伸，追踪真实极值
        # 否则回调深度会基于过时的 peak 计算，导致错过入场
        if self._cascade_direction == 1:
            current_high = (
                safe_float(df["high"].iloc[-1])
                if "high" in df.columns else price
            )
            if current_high > self._cascade_peak:
                self._cascade_peak = current_high
                self._cascade_range = self._cascade_peak - self._cascade_origin
        else:
            current_low = (
                safe_float(df["low"].iloc[-1])
                if "low" in df.columns else price
            )
            if current_low < self._cascade_peak:
                self._cascade_peak = current_low
                self._cascade_range = self._cascade_origin - self._cascade_peak

        if self._cascade_range <= 0:
            self._cascade_direction = 0
            return 0

        # 计算回调深度（基于最新 peak）
        if self._cascade_direction == 1:
            # 看涨瀑布后: 价格从 peak 回落
            pullback_depth = (self._cascade_peak - price) / self._cascade_range
        else:
            # 看跌瀑布后: 价格从 peak（低点）反弹
            pullback_depth = (price - self._cascade_peak) / self._cascade_range

        # 回调过深（> 70%）→ 瀑布失败
        if pullback_depth > self.pullback_max_pct:
            self._cascade_direction = 0
            return 0

        # 回调足够（25%-70%）→ 入场!
        if pullback_depth >= self.pullback_min_pct:
            direction = self._cascade_direction
            self._cascade_direction = 0  # 消费信号
            return direction

        # 还没回调够 → 继续等待
        return 0

    def _store_cascade(self, df: pd.DataFrame, direction: int):
        """记录瀑布事件参数，用于后续回调检测。"""
        closes = df[PRICE_COL].astype(float)
        price = safe_float(closes.iloc[-1])

        self._cascade_direction = direction
        self._pullback_wait_bars = 0

        if direction == 1:
            # 看涨瀑布: origin=瀑布前价格, peak=当前 high
            origin = (
                safe_float(closes.iloc[-1 - self.roc_period])
                if len(closes) > self.roc_period else price
            )
            peak = (
                safe_float(df["high"].iloc[-1])
                if "high" in df.columns else price
            )
            self._cascade_origin = origin
            self._cascade_peak = peak
            self._cascade_range = max(peak - origin, 0.0)
        else:
            # 看跌瀑布: origin=瀑布前价格, peak=当前 low
            origin = (
                safe_float(closes.iloc[-1 - self.roc_period])
                if len(closes) > self.roc_period else price
            )
            peak = (
                safe_float(df["low"].iloc[-1])
                if "low" in df.columns else price
            )
            self._cascade_origin = origin
            self._cascade_peak = peak
            self._cascade_range = max(origin - peak, 0.0)

    # ==================================================================
    # 信号
    # ==================================================================

    def _generate_signal(self, df: pd.DataFrame, current_pos: float) -> float:
        """清算猎杀信号（两阶段: 瀑布检测 → 回调入场）。

        状态流转:
          IDLE → [瀑布检测通过] → WAITING_PULLBACK
          WAITING_PULLBACK → [回调到位] → 返回入场信号
          WAITING_PULLBACK → [超时/回调过深] → IDLE
          HOLDING → [max_hold] → 平仓 → IDLE
        """
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)

        # ── 反向平仓后的等待期（close_then_wait）──────────────────
        if curr_side == 0 and self._reverse_cooldown_remaining > 0:
            self._reverse_cooldown_remaining -= 1
            return 0.0

        # ── 冷却/暂停 ────────────────────────────────────────
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            # 冷却期间清除未处理的瀑布信号
            if self._cascade_direction != 0:
                self._cascade_direction = 0
            return 0.0

        if self._consecutive_losses >= self.max_consecutive_losses:
            self._loss_pause_bars += 1
            if self._loss_pause_bars >= self.loss_pause_timeout:
                self._consecutive_losses = 0
                self._loss_pause_bars = 0
            else:
                if self._cascade_direction != 0:
                    self._cascade_direction = 0
                return 0.0

        # ── 持仓管理 ──────────────────────────────────────────
        if curr_side != 0:
            if curr_side == self._last_side:
                self._hold_bars += 1
            else:
                self._hold_bars = 1
                self._last_side = curr_side

            if self._hold_bars >= self.max_hold_bars:
                return float(-curr_side)

            return 0.0

        # ── Phase 2: 回调入场（如果已有瀑布信号）─────────────
        if self._cascade_direction != 0:
            entry = self._check_pullback_entry(df)
            if entry != 0:
                self._hold_bars = 0
                self._last_side = entry
                return float(entry)
            # 继续等待回调（_check_pullback_entry 内部处理超时/取消）
            return 0.0

        # ── Phase 1: 瀑布检测 ─────────────────────────────────
        direction = self._detect_cascade(df)
        if direction != 0:
            self._store_cascade(df, direction)
            # 不立刻入场 — 等待回调
        return 0.0

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
        """清算猎杀策略主入口。"""
        # 0. 通过 factor_manager 计算因子
        await factor_manager("roc", df)
        await factor_manager("atr", df)
        await factor_manager("obv", df)

        price = safe_float(df[PRICE_COL].iloc[-1])
        atr = safe_float(df[f"atr_{self.atr_period}"].iloc[-1])

        # 1. 风控
        self._update_risk_state(current_pos, price)

        # 2. 信号
        signal = self._generate_signal(df, current_pos)
        sig_int = int(np.sign(signal)) if signal != 0 else 0

        # 3. 仓位
        sizing = calc_position_size(
            equity=equity, price=price, atr=atr,
            risk_pct=self.risk_per_trade, atr_sl_mult=self.atr_sl_multiplier,
            min_leverage=self.min_leverage, max_leverage=self.max_leverage,
            min_notional_pct=self.min_notional_pct, max_notional_pct=self.max_notional_pct,
            min_quantity=self.min_quantity, quantity_precision=self.quantity_precision,
        )

        # 4. 订单（固定 TP/SL — 快进快出）
        result = build_order(
            symbol=self.symbol, signal=sig_int, current_pos=current_pos,
            sizing=sizing, price=price, atr=atr,
            atr_sl_multiplier=self.atr_sl_multiplier,
            atr_tp_multiplier=self.atr_tp_multiplier,
            use_trailing_stop=False, fee_rate=self.fee_rate,
            flip_mode=self.flip_mode,
            tp_sl_mode="ROI",
            quantity_precision=self.quantity_precision,
        )

        if result["signal"] == "CLOSE" and self.flip_mode == "close_then_wait":
            self._reverse_cooldown_remaining = int(self.reverse_cooldown_bars)

        if result["signal"] in ("LONG", "SHORT"):
            self._entry_price = price
            self._best_price = price

        # 5. 追踪止损信息（保持接口一致）
        effective_pos = current_pos
        if result["signal"] == "LONG":
            effective_pos = sizing["quantity"]
        elif result["signal"] == "SHORT":
            effective_pos = -sizing["quantity"]

        trailing_info, self._best_price = compute_trailing_stop(
            effective_pos, price, atr, self.atr_sl_multiplier,
            self._entry_price, self._best_price,
        )

        # 6. 瀑布状态信息
        if self._cascade_direction != 0:
            phase = "waiting_pullback"
        elif current_pos != 0:
            phase = "holding"
        else:
            phase = "idle"

        cascade_info = {
            "phase": phase,
            "cascade_direction": self._cascade_direction,
            "cascade_origin": round(self._cascade_origin, 2),
            "cascade_peak": round(self._cascade_peak, 2),
            "cascade_range": round(self._cascade_range, 2),
            "pullback_wait_bars": self._pullback_wait_bars,
        }

        factors_json = build_factors_json(df, [
            f"roc_{self.roc_period}",
            f"atr_{self.atr_period}",
            "obv",
        ], extra={
            **cascade_info,
            "hold_bars": self._hold_bars,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_remaining": self._cooldown_remaining,
        })

        meta = {**result["meta"], "cascade": cascade_info, "factors_json": factors_json}

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
