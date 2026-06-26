"""
波动率自适应网格策略 — Binance USDT-M 永续合约 (BTCUSDT)

一个平衡型网格策略，在震荡市中通过价格在网格间波动持续盈利。

设计理念: **动态网格 + 趋势感知 = 震荡市"印钞机"**
  - 以 EMA(50) 为网格中心，ATR 为网格间距（自适应波动率）
  - 上下各 N 层网格，价格穿越网格线 → 做多/做空
  - 每笔网格交易目标利润 = 1 个网格间距
  - 入场时锁定 center/spacing，持仓期间以锁定值判断 TP/SL
  - 空仓时 center/spacing 实时更新（趋势感知）
  - ADX > 25 时暂停网格（趋势来临，网格策略不适合）
  - 价格突破网格范围 N 层 → 止损关闭所有仓位
  - 入场前检查 spacing 是否覆盖手续费，避免低波动期亏损

简化设计:
  由于 __call__ 接口每次只返回一个订单，本策略:
  - 跟踪当前最近的网格层级
  - 每 bar 只处理一个网格触发事件
  - 用内部状态追踪已触发的网格层级

网格结构（以做多为例）:
    Grid +3: 卖出止盈     (center + 3 × spacing)
    Grid +2: 卖出止盈     (center + 2 × spacing)
    Grid +1: 卖出止盈     (center + 1 × spacing)
    Center:  EMA(50)      (中心锚点)
    Grid -1: 买入入场     (center - 1 × spacing)
    Grid -2: 买入入场     (center - 2 × spacing)
    Grid -3: 买入入场     (center - 3 × spacing)

风控机制:
  - 杠杆: 2-3 倍（低杠杆，多笔小仓位）
  - 单笔名义价值: 3-5% 权益（每层网格仓位很小）
  - 总名义价值上限: 15% 权益
  - ADX > 25 → 暂停新网格交易（趋势保护）
  - 价格超出网格范围 → 止损平仓（含无仓位时的入场屏蔽）
  - 入场 spacing 必须 > 往返手续费 × 安全倍数（fee profitability guard）
  - 每笔止损 = 2× 网格间距
"""

from typing import List, Dict, Any, Optional
from pydantic import Field
import numpy as np
import pandas as pd

from src.strategy.types import Strategy
from src.factor import factor_manager
from src.strategy.futures._utils import (
    PRICE_COL, DEFAULT_TAKER_FEE, safe_float,
    calc_position_size, build_order, build_factors_json,
)


class VolatilityGridStrategy(Strategy):
    """波动率自适应网格策略 — 震荡市中的网格交易。"""

    # ── 策略元信息 ────────────────────────────────────────────
    name: str = Field(default="volatility_grid", description="策略名称")
    description: str = Field(
        default="波动率自适应网格策略 — 动态网格间距，震荡市持续盈利",
        description="策略描述",
    )
    factor_names: List[str] = Field(
        default=["ema", "atr", "adx"],
        description="使用的因子",
    )

    # ── 交易参数 ──────────────────────────────────────────────
    symbol: str = Field(default="BTCUSDT", description="交易对")
    quantity_precision: int = Field(default=3, description="数量精度")
    min_quantity: float = Field(default=0.02, description="最小下单数量")

    # ── 网格参数 ──────────────────────────────────────────────
    grid_levels: int = Field(default=3, description="单侧网格层数（总 6 层）")
    spacing_atr_mult: float = Field(
        default=1.0,
        description="网格间距 = spacing_atr_mult × ATR（自适应波动率）",
    )
    ema_center_period: int = Field(default=50, description="网格中心 EMA 周期")
    atr_period: int = Field(default=14, description="ATR 周期")
    adx_period: int = Field(default=14, description="ADX 周期")

    # ── 仓位管理（单笔保守，总量可控）─────────────────────────
    risk_per_grid: float = Field(default=0.01, description="每层网格风险 1% 权益")
    min_leverage: int = Field(default=2, description="最小杠杆 2 倍")
    max_leverage: int = Field(default=3, description="最大杠杆 3 倍")
    per_grid_notional_pct: float = Field(default=0.15, description="每层名义价值 15% 权益")
    max_total_notional_pct: float = Field(default=1.0, description="总名义价值 100% 权益上限")
    fee_rate: float = Field(default=DEFAULT_TAKER_FEE, description="单边手续费率")

    # ── 风控 ──────────────────────────────────────────────────
    adx_pause_threshold: float = Field(default=25.0, description="ADX > 25 暂停网格")
    adx_resume_threshold: float = Field(default=20.0, description="ADX < 20 恢复网格")
    max_breach_levels: int = Field(default=2, description="价格超出网格 N 层 → 止损")
    cooldown_bars: int = Field(default=40, description="止损冷却 40 bar")
    max_consecutive_losses: int = Field(default=3, description="最多连续亏损")
    loss_pause_timeout: int = Field(default=300, description="暂停超时")
    win_cooldown_bars: int = Field(default=10, description="盈利后冷却 bar 数")
    fee_safety_mult: float = Field(
        default=2.0,
        description="手续费安全倍数: spacing 必须 > round_trip_fee × fee_safety_mult 才入场",
    )

    # ── 反向处理（网格策略只允许平仓，不反手）────────────────────
    flip_mode: str = Field(
        default="close_only",
        description="反向信号处理模式（网格策略默认 close_only，避免反手追单）。",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 网格状态
        self._grid_paused: bool = False
        self._current_grid_level: int = 0  # 当前网格层（>0 多仓层, <0 空仓层, 0 无仓）
        self._last_center: Optional[float] = None
        self._last_spacing: Optional[float] = None
        # 持仓期间锁定的 center/spacing（避免 EMA/ATR 漂移导致误判 TP/SL）
        self._entry_center: Optional[float] = None
        self._entry_spacing: Optional[float] = None
        # 风控
        self._cooldown_remaining: int = 0
        self._consecutive_losses: int = 0
        self._loss_pause_bars: int = 0
        self._win_cooldown_remaining: int = 0
        self._hold_bars: int = 0
        self._last_position: float = 0.0
        self._entry_price: Optional[float] = None

    # ==================================================================
    # 网格计算
    # ==================================================================

    def _compute_grid(self, center: float, spacing: float) -> Dict[str, Any]:
        """计算网格层级价格。"""
        buy_levels = []   # 下方买入价
        sell_levels = []  # 上方卖出价
        for i in range(1, self.grid_levels + 1):
            buy_levels.append(round(center - i * spacing, 2))
            sell_levels.append(round(center + i * spacing, 2))
        return {
            "center": center,
            "spacing": spacing,
            "buy_levels": buy_levels,   # 从高到低 [-1, -2, -3]
            "sell_levels": sell_levels,  # 从低到高 [+1, +2, +3]
        }

    def _find_grid_level(self, price: float, center: float, spacing: float) -> int:
        """确定价格所在的网格层级。

        使用截断（向零取整），确保价格必须完全穿越一条网格线才触发层级变化。
        例如: offset=0.9 → level=0, offset=1.0 → level=1

        返回: >0 = 在中心以上第 N 层, <0 = 在中心以下第 N 层, 0 = 在中心附近
        """
        if spacing <= 0:
            return 0
        offset = (price - center) / spacing
        return int(offset)  # 截断向零取整：必须完整跨越网格线

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
                # 盈利: 重置连续亏损和亏损暂停状态，启动盈利冷却
                self._consecutive_losses = 0
                self._loss_pause_bars = 0
                self._win_cooldown_remaining = self.win_cooldown_bars

        if current_pos == 0:
            self._current_grid_level = 0
            # 清仓时释放锁定的入场网格参数
            self._entry_center = None
            self._entry_spacing = None
            self._hold_bars = 0
        elif current_pos != 0:
            self._hold_bars += 1

        if old_exited:
            self._entry_price = None

        self._last_position = current_pos

    # ==================================================================
    # 信号
    # ==================================================================

    def _generate_signal(self, df: pd.DataFrame, current_pos: float) -> float:
        """网格信号生成。"""
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)

        # ── 冷却/暂停 ────────────────────────────────────────
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return 0.0

        if self._win_cooldown_remaining > 0:
            self._win_cooldown_remaining -= 1
            return 0.0

        if self._consecutive_losses >= self.max_consecutive_losses:
            self._loss_pause_bars += 1
            if self._loss_pause_bars >= self.loss_pause_timeout:
                self._consecutive_losses = 0
                self._loss_pause_bars = 0
            else:
                return 0.0

        price = safe_float(df[PRICE_COL].iloc[-1])
        ema_center = safe_float(df[f"ema_{self.ema_center_period}"].iloc[-1])
        atr = safe_float(df[f"atr_{self.atr_period}"].iloc[-1])
        adx = safe_float(df[f"adx_{self.adx_period}"].iloc[-1], 15.0)

        if price <= 0 or ema_center <= 0 or atr <= 0:
            return 0.0

        spacing = atr * self.spacing_atr_mult
        if spacing <= 0:
            return 0.0

        # 实时网格参数（仅空仓时用于入场判断）
        self._last_center = ema_center
        self._last_spacing = spacing

        # ── ADX 趋势保护 ─────────────────────────────────────
        if adx > self.adx_pause_threshold:
            self._grid_paused = True
        elif adx < self.adx_resume_threshold:
            self._grid_paused = False

        # ── 持仓期间: 使用入场时锁定的 center/spacing ──────────
        if curr_side != 0:
            # 冷启动容错: 如果已有持仓但 _entry_center 未初始化
            # （策略重启、外部开仓等场景），从当前实时值初始化
            if self._entry_center is None or self._entry_spacing is None:
                self._entry_center = ema_center
                self._entry_spacing = spacing
                if self._entry_price is None:
                    self._entry_price = price

            hold_center = self._entry_center
            hold_spacing = self._entry_spacing
            hold_level = self._find_grid_level(price, hold_center, hold_spacing)

            # 价格超出网格范围（基于入场网格）→ 止损
            if abs(hold_level) > self.grid_levels + self.max_breach_levels:
                return float(-curr_side)  # 平仓止损

            # 止盈: 价格回到入场网格中心附近
            if curr_side == 1 and hold_level >= 0:
                return float(-curr_side)
            if curr_side == -1 and hold_level <= 0:
                return float(-curr_side)

            return 0.0

        # ── 空仓逻辑: 使用实时 center/spacing ─────────────────
        current_level = self._find_grid_level(price, ema_center, spacing)

        # 价格超出网格范围时，即使无仓位也不入场（避免在极端价格开仓）
        if abs(current_level) > self.grid_levels:
            return 0.0

        # 网格暂停时不开新仓
        if self._grid_paused:
            return 0.0

        # ── 手续费盈利性检查 ─────────────────────────────────
        # spacing 必须 > 往返手续费 × 安全倍数，否则每笔交易注定亏损
        round_trip_fee_cost = 2.0 * self.fee_rate * price
        if spacing < round_trip_fee_cost * self.fee_safety_mult:
            return 0.0  # spacing 太小，无利可图

        # ── 网格入场 ─────────────────────────────────────────
        if current_level <= -1:
            # 价格在中心以下 → 做多（均值回归）
            self._current_grid_level = current_level
            return 1.0
        elif current_level >= 1:
            # 价格在中心以上 → 做空（均值回归）
            self._current_grid_level = current_level
            return -1.0

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
        """波动率网格策略主入口。"""
        # 0. 通过 factor_manager 计算因子（动态传参确保 period 匹配）
        await factor_manager("ema", df, periods=[self.ema_center_period])
        await factor_manager("atr", df, periods=[self.atr_period])
        await factor_manager("adx", df, periods=[self.adx_period])

        price = safe_float(df[PRICE_COL].iloc[-1])
        atr = safe_float(df[f"atr_{self.atr_period}"].iloc[-1])

        # 1. 风控
        self._update_risk_state(current_pos, price)

        # 2. 信号
        signal = self._generate_signal(df, current_pos)
        sig_int = int(np.sign(signal)) if signal != 0 else 0

        # 3. 仓位（单层网格仓位）
        sizing = calc_position_size(
            equity=equity, price=price, atr=atr,
            risk_pct=self.risk_per_grid, atr_sl_mult=2.0 * self.spacing_atr_mult,
            min_leverage=self.min_leverage, max_leverage=self.max_leverage,
            min_notional_pct=0.03, max_notional_pct=self.per_grid_notional_pct,
            min_quantity=self.min_quantity, quantity_precision=self.quantity_precision,
        )

        # 4. 订单（固定止损 = 2× 网格间距，止盈由信号逻辑处理）
        sl_mult = 2.0 * self.spacing_atr_mult  # 止损 = 2× 网格间距
        result = build_order(
            symbol=self.symbol, signal=sig_int, current_pos=current_pos,
            sizing=sizing, price=price, atr=atr,
            atr_sl_multiplier=sl_mult, atr_tp_multiplier=0.0,
            use_trailing_stop=False,
            flip_mode=self.flip_mode,
            fee_rate=self.fee_rate,
            quantity_precision=self.quantity_precision,
        )

        # 5. 入场时锁定 center/spacing，持仓期间保持不变
        if result["signal"] in ("LONG", "SHORT"):
            self._entry_price = price
            self._hold_bars = 0
            self._entry_center = self._last_center
            self._entry_spacing = self._last_spacing

        # 6. 网格信息
        grid_info = {}
        # 展示网格: 持仓时用入场网格，空仓时用实时网格
        show_center = self._entry_center if self._entry_center else self._last_center
        show_spacing = self._entry_spacing if self._entry_spacing else self._last_spacing
        if show_center and show_spacing:
            grid_info = self._compute_grid(show_center, show_spacing)
            grid_info["current_level"] = self._current_grid_level
            grid_info["paused"] = self._grid_paused
            grid_info["locked"] = self._entry_center is not None

        factors_json = build_factors_json(df, [
            f"ema_{self.ema_center_period}",
            f"atr_{self.atr_period}",
            f"adx_{self.adx_period}",
        ], extra={
            "hold_bars": self._hold_bars,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_remaining": self._cooldown_remaining,
            "grid_level": self._current_grid_level,
        })

        meta = {**result["meta"], "grid": grid_info, "factors_json": factors_json}

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
                "win_cooldown_remaining": self._win_cooldown_remaining,
                "hold_bars": self._hold_bars,
            },
            "trailing_stop": {
                "active": False, "should_update": False, "sl_price": None,
                "entry_price": self._entry_price,
            },
        }
