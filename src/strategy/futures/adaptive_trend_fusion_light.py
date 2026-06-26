"""
Adaptive Trend Fusion Light — 轻量版趋势融合（1m 永续友好）

结构：因子计算走 factor_manager → __call__ 按阶段编排
信号源：EMA 交叉（一次性事件）+ 可选动量突破 / 回调
过滤层：ADX 震荡过滤 → ATR 低波动过滤 → SMA 中性区过滤 → RSI 极端区过滤
风控层：min_hold + 冷却期 + 信号去重 + ROI TP/SL + 动态仓位
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from pydantic import Field

from src.strategy.types import Strategy
from src.factor import factor_manager
from src.strategy.futures._utils import (
    PRICE_COL,
    VOL_COL,
    DEFAULT_TAKER_FEE,
    build_factors_json,
)


# ============================================================
#         轻量策略工具
# ============================================================

def find_swing_points(high: pd.Series, low: pd.Series, lookback: int = 20) -> Tuple[Optional[float], Optional[float]]:
    lb = int(lookback)
    if len(high) < lb + 2:
        return None, None
    hh = high.iloc[-(lb + 1):-1].max()
    ll = low.iloc[-(lb + 1):-1].min()
    return (float(hh) if not pd.isna(hh) else None), (float(ll) if not pd.isna(ll) else None)


def check_volume_surge(volumes: pd.Series, current_vol: float, lookback: int = 20, multiplier: float = 1.5) -> bool:
    lb = int(lookback)
    if len(volumes) < lb:
        return False
    avg_vol = float(volumes.iloc[-lb:].mean())
    if avg_vol <= 0:
        return False
    return float(current_vol) >= avg_vol * float(multiplier)


# ============================================================
#         Adaptive Trend Fusion Light（策略类）
# ============================================================

class AdaptiveTrendFusionLightStrategy(Strategy):
    name: str = Field(default="adaptive_trend_fusion_light", description="策略名称")
    description: str = Field(
        default="轻量趋势融合：SMA 主趋势 + EMA 交叉/放量突破 + ADX/ATR/RSI 多层过滤",
        description="策略描述",
    )
    factor_names: List[str] = Field(
        default=["sma", "ema", "rsi", "atr", "adx"],
        description="light 策略用到的因子",
    )

    # ── 交易参数 ─────────────────────────────────────────────
    symbol: str = Field(default="BTCUSDT", description="交易对")
    quantity: float = Field(default=0.04, description="固定下单数量（use_dynamic_sizing=False 时生效）")
    leverage: int = Field(default=5, description="固定杠杆")
    quantity_precision: int = Field(default=3, description="数量精度")
    min_quantity: float = Field(default=0.04, description="最小下单数量")

    # ── 持仓 & 冷却 ─────────────────────────────────────────
    min_hold_bars: int = Field(default=120, description="最小持仓 bar 数（2h），防噪声反复开平")
    cooldown_bars: int = Field(default=120, description="平仓后冷却期（2h），等待新趋势确认")

    # ── 趋势主干 ─────────────────────────────────────────────
    sma_trend_period: int = Field(default=200, description="主趋势 SMA 周期")
    sma_trend_buf: float = Field(default=0.006, description="SMA 趋势缓冲 0.6%；中性区内完全禁止开仓")
    ema_fast: int = Field(default=20, description="EMA 快线周期")
    ema_slow: int = Field(default=50, description="EMA 慢线周期")

    # ── ADX 过滤 ─────────────────────────────────────────────
    adx_period: int = Field(default=14, description="ADX 周期")
    adx_threshold: float = Field(default=25.0, description="ADX <= 阈值视为震荡，禁止新开仓")

    # ── ATR 低波动过滤 ───────────────────────────────────────
    atr_min_threshold: float = Field(
        default=35.0,
        description="ATR 下限（USD）；低于此值视为低波动盘整，禁止开仓（BTC 1m 参考值 35-40）",
    )

    # ── RSI ───────────────────────────────────────────────────
    rsi_period: int = Field(default=14, description="RSI 周期")
    rsi_oversold: float = Field(default=30.0, description="多头回调入场 RSI 阈值")
    rsi_overbought: float = Field(default=70.0, description="空头回调入场 RSI 阈值")
    rsi_cross_long_max: float = Field(default=65.0, description="EMA/动量做多 RSI 上限（从70收严至65）")
    rsi_cross_short_min: float = Field(default=35.0, description="EMA/动量做空 RSI 下限（从30收严至35）")

    # ── 动量突破 ─────────────────────────────────────────────
    swing_lookback: int = Field(default=20, description="结构高低点回看")
    breakout_buf: float = Field(default=0.0005, description="突破确认缓冲（0.05%）")
    vol_surge_lookback: int = Field(default=20, description="成交量回看")
    vol_surge_multiplier: float = Field(default=1.5, description="成交量放大倍数")

    # ── TP/SL ────────────────────────────────────────────────
    use_roi_tpsl: bool = Field(default=True, description="是否使用 ROI 模式 TP/SL")
    tp_roi_pct: float = Field(default=10.0, description="止盈 10% ROI = +2% 价格（5x 杠杆）")
    sl_roi_pct: float = Field(default=5.0, description="止损 5% ROI = -1% 价格（5x 杠杆）")

    # ── ATR 策略层止损（默认关闭）───────────────────────────
    use_atr_stop: bool = Field(default=False, description="策略层 ATR 止损开关")
    atr_sl_mult: float = Field(default=6.0, description="ATR 止损倍数")

    # ── 入场信号开关 ─────────────────────────────────────────
    use_pullback_entry: bool = Field(default=False, description="回调入场（默认关闭）")
    use_momentum_entry: bool = Field(default=True, description="动量突破入场")

    # ── 动态仓位 ─────────────────────────────────────────────
    use_dynamic_sizing: bool = Field(default=True, description="按权益风险比例动态计算仓位")
    risk_pct: float = Field(default=0.01, description="每笔风险 = 权益 × 1%")

    # ── 执行成本 ─────────────────────────────────────────────
    fee_rate: float = Field(default=DEFAULT_TAKER_FEE, description="每侧手续费率")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_side: int = 0
        self._hold_bars: int = 0
        self._last_ema_fast: Optional[float] = None
        self._last_ema_slow: Optional[float] = None
        self._entry_price: Optional[float] = None
        self._cooldown_remaining: int = 0
        self._last_signal_dir: int = 0

    # ──────────────────────────────────────────────────────────
    #  内部方法
    # ──────────────────────────────────────────────────────────

    def _update_hold_state(self, curr_side: int, prev_side: int) -> None:
        if curr_side == 0:
            if prev_side != 0:
                self._cooldown_remaining = int(self.cooldown_bars)
            self._hold_bars = 0
            self._last_side = 0
            return
        if curr_side == self._last_side:
            self._hold_bars += 1
        else:
            self._hold_bars = 1
            self._last_side = curr_side

    def _read_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        closes = df[PRICE_COL].astype(float)
        highs = df["high"].astype(float) if "high" in df.columns else closes
        lows = df["low"].astype(float) if "low" in df.columns else closes
        vols = df[VOL_COL].astype(float)

        def _last(col: str) -> float:
            return float(df[col].iloc[-1]) if col in df.columns else float("nan")

        sma_col = f"sma_{int(self.sma_trend_period)}"
        ema_fast_col = f"ema_{int(self.ema_fast)}"
        ema_slow_col = f"ema_{int(self.ema_slow)}"
        rsi_col = f"rsi_{int(self.rsi_period)}"
        atr_col = f"atr_{int(self.adx_period)}"
        adx_col = f"adx_{int(self.adx_period)}"

        swing_high, swing_low = find_swing_points(highs, lows, self.swing_lookback)
        vol_surge = check_volume_surge(
            vols, float(vols.iloc[-1]), self.vol_surge_lookback, self.vol_surge_multiplier,
        )

        return {
            "price": float(closes.iloc[-1]),
            "volume": float(vols.iloc[-1]),
            "sma_trend": _last(sma_col),
            "ema_fast": _last(ema_fast_col),
            "ema_slow": _last(ema_slow_col),
            "rsi": _last(rsi_col),
            "adx": _last(adx_col),
            "atr": _last(atr_col),
            "swing_high": swing_high,
            "swing_low": swing_low,
            "vol_surge": bool(vol_surge),
        }

    def _generate_raw_side(self, ind: Dict[str, float]) -> int:
        price = ind["price"]
        sma_trend = ind["sma_trend"]
        ema_f = ind["ema_fast"]
        ema_s = ind["ema_slow"]
        rsi = ind["rsi"]
        adx = ind["adx"]
        atr = ind.get("atr", 0.0) or 0.0

        if any(np.isnan(v) for v in [price, sma_trend, ema_f, ema_s, rsi, adx]):
            return 0

        # ── 过滤层 1：ADX 震荡过滤 ──
        if adx <= float(self.adx_threshold):
            return 0

        # ── 过滤层 2：ATR 低波动过滤 ──
        if atr < float(self.atr_min_threshold):
            return 0

        # ── 过滤层 3：SMA 趋势方向 + 中性区禁入 ──
        trend_dir = 0
        if price > sma_trend * (1.0 + float(self.sma_trend_buf)):
            trend_dir = 1
        elif price < sma_trend * (1.0 - float(self.sma_trend_buf)):
            trend_dir = -1
        # 中性区（price 在 SMA ± 0.6% 内）：完全禁止开仓
        if trend_dir == 0:
            return 0

        # ── 信号源 1：EMA 交叉 ──
        ema_cross = 0
        if self._last_ema_fast is not None and self._last_ema_slow is not None:
            prev_fast_above = float(self._last_ema_fast) > float(self._last_ema_slow)
            curr_fast_above = float(ema_f) > float(ema_s)
            if (not prev_fast_above) and curr_fast_above:
                ema_cross = 1
            elif prev_fast_above and (not curr_fast_above):
                ema_cross = -1
        else:
            ema_cross = 1 if ema_f > ema_s else (-1 if ema_f < ema_s else 0)

        self._last_ema_fast = float(ema_f)
        self._last_ema_slow = float(ema_s)

        # RSI 过滤 EMA 交叉（避免超买追多 / 超卖追空）
        if ema_cross == 1 and rsi > float(self.rsi_cross_long_max):
            ema_cross = 0
        if ema_cross == -1 and rsi < float(self.rsi_cross_short_min):
            ema_cross = 0

        # ── 信号源 2：动量突破 ──
        momentum = 0
        if self.use_momentum_entry:
            sh = ind.get("swing_high", float("nan"))
            sl = ind.get("swing_low", float("nan"))
            if (not np.isnan(sh)) and (not np.isnan(sl)) and bool(ind.get("vol_surge", False)):
                buf = float(self.breakout_buf)
                if price > sh * (1.0 + buf):
                    momentum = 1
                elif price < sl * (1.0 - buf):
                    momentum = -1

        # RSI 过滤动量突破
        if momentum == 1 and rsi > float(self.rsi_cross_long_max):
            momentum = 0
        if momentum == -1 and rsi < float(self.rsi_cross_short_min):
            momentum = 0

        # ── 信号源 3：回调入场（默认关闭）──
        pullback = 0
        if self.use_pullback_entry and ema_f > 0:
            price_to_ema = (price - ema_f) / ema_f
            if trend_dir >= 0:
                if -0.005 <= price_to_ema <= 0.01 and rsi < float(self.rsi_oversold):
                    pullback = 1
            if trend_dir <= 0:
                if -0.01 <= price_to_ema <= 0.005 and rsi > float(self.rsi_overbought):
                    pullback = -1

        # ── 信号合成 ──
        raw = ema_cross or momentum or pullback

        # ── 趋势方向硬过滤：只允许顺 SMA 方向 ──
        if trend_dir == 1 and raw < 0:
            raw = 0
        if trend_dir == -1 and raw > 0:
            raw = 0

        return int(np.sign(raw))

    def _build_order(
        self,
        target_side: int,
        curr_side: int,
        price: float,
        current_pos: float,
        equity: float = 0.0,
        atr: float = 0.0,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if target_side == curr_side:
            return "HOLD", None

        # CLOSE
        if target_side == 0 and curr_side != 0:
            side = "SELL" if curr_side == 1 else "BUY"
            close_qty = round(abs(float(current_pos)), int(self.quantity_precision))
            if close_qty <= 0:
                return "HOLD", None
            return "CLOSE", {
                "symbol": self.symbol, "side": side, "type": "MARKET",
                "quantity": close_qty, "position_side": "BOTH", "reduce_only": True,
            }

        # FLIP → close first
        if target_side != 0 and curr_side != 0 and target_side != curr_side:
            side = "SELL" if curr_side == 1 else "BUY"
            close_qty = round(abs(float(current_pos)), int(self.quantity_precision))
            if close_qty <= 0:
                return "HOLD", None
            return "CLOSE", {
                "symbol": self.symbol, "side": side, "type": "MARKET",
                "quantity": close_qty, "position_side": "BOTH", "reduce_only": True,
            }

        # OPEN
        if target_side != 0 and curr_side == 0:
            side = "BUY" if target_side == 1 else "SELL"

            if self.use_dynamic_sizing and equity > 0 and price > 0:
                risk_amount = equity * float(self.risk_pct)
                sl_price_pct = float(self.sl_roi_pct) / int(self.leverage) / 100.0
                qty = risk_amount / (sl_price_pct * price) if sl_price_pct > 0 else float(self.quantity)
                qty = max(float(self.min_quantity), round(qty, int(self.quantity_precision)))
            else:
                qty = max(float(self.quantity), float(self.min_quantity))
                qty = round(qty, int(self.quantity_precision))

            order: Dict[str, Any] = {
                "symbol": self.symbol, "side": side, "type": "MARKET",
                "quantity": qty, "position_side": "BOTH",
            }
            if self.use_roi_tpsl:
                order["tp_sl_mode"] = "ROI"
                order["take_profit"] = float(self.tp_roi_pct)
                order["stop_loss"] = float(self.sl_roi_pct)
            self._entry_price = float(price)
            return ("LONG" if target_side == 1 else "SHORT"), order

        return "HOLD", None

    # ──────────────────────────────────────────────────────────
    #  __call__
    # ──────────────────────────────────────────────────────────

    async def __call__(
        self,
        df: pd.DataFrame,
        current_pos: float = 0.0,
        equity: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Stage 0) 输入校验
        if df is None or df.empty:
            return {"signal": "HOLD", "order": None}
        if PRICE_COL not in df.columns or VOL_COL not in df.columns:
            return {"signal": "HOLD", "order": None}

        max_period = max(
            int(self.sma_trend_period), int(self.ema_slow),
            int(self.adx_period), int(self.rsi_period), int(self.swing_lookback),
        )
        if len(df) < max_period + 10:
            return {"signal": "HOLD", "order": None}

        # Stage 1) 因子计算
        await factor_manager("sma", df, periods=[int(self.sma_trend_period)])
        await factor_manager("ema", df, periods=[int(self.ema_fast), int(self.ema_slow)])
        await factor_manager("rsi", df, periods=[int(self.rsi_period)])
        await factor_manager("atr", df, periods=[int(self.adx_period)])
        await factor_manager("adx", df, periods=[int(self.adx_period)])

        # Stage 2) 状态更新
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
        prev_last_side = self._last_side
        self._update_hold_state(curr_side, prev_last_side)
        if curr_side == 0 and self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining == 0:
                self._last_signal_dir = 0

        # Stage 3) 指标读取
        ind = self._read_indicators(df)
        price = float(ind["price"])
        atr = float(ind.get("atr", 0.0) or 0.0)

        # Stage 4) 生成信号
        target_side = self._generate_raw_side(ind)

        # Stage 4b) 信号去重：同方向不重复开仓
        if curr_side == 0 and target_side != 0 and target_side == self._last_signal_dir:
            target_side = 0
        if curr_side != 0:
            self._last_signal_dir = curr_side

        # Stage 5) min_hold 门控
        if curr_side != 0 and int(self.min_hold_bars) > 0:
            if (target_side == 0 or target_side != curr_side) and self._hold_bars < int(self.min_hold_bars):
                target_side = curr_side

        # Stage 5b) ATR 策略层止损（默认关闭）
        if self.use_atr_stop and curr_side != 0 and self._entry_price is not None and atr > 0:
            adverse_move = (price - self._entry_price) * curr_side
            if adverse_move < -(float(self.atr_sl_mult) * atr):
                target_side = 0

        # Stage 5c) 冷却期门控
        if curr_side == 0 and self._cooldown_remaining > 0 and target_side != 0:
            target_side = 0

        # Stage 6) 下单
        signal, order = self._build_order(target_side, curr_side, price, current_pos, equity=equity, atr=atr)

        actual_qty = float(order["quantity"]) if order and "quantity" in order else max(float(self.min_quantity), float(self.quantity))
        sizing = {
            "quantity": actual_qty,
            "leverage": int(self.leverage),
            "method": "dynamic_roi" if self.use_dynamic_sizing else "fixed",
        }

        # Stage 7) factors_json
        factor_cols = [
            f"sma_{int(self.sma_trend_period)}",
            f"ema_{int(self.ema_fast)}", f"ema_{int(self.ema_slow)}",
            f"rsi_{int(self.rsi_period)}",
            f"atr_{int(self.adx_period)}", f"adx_{int(self.adx_period)}",
            f"plus_di_{int(self.adx_period)}", f"minus_di_{int(self.adx_period)}",
        ]
        factors_json = build_factors_json(
            df, factor_cols,
            extra={
                "swing_high": ind.get("swing_high"),
                "swing_low": ind.get("swing_low"),
                "vol_surge": bool(ind.get("vol_surge", False)),
                "hold_bars": self._hold_bars,
                "cooldown_remaining": self._cooldown_remaining,
                "target_side": int(target_side),
                "curr_side": int(curr_side),
                "entry_price": self._entry_price,
            },
        )

        return {
            "signal": signal,
            "order": order,
            "leverage": int(self.leverage),
            "sizing": sizing,
            "atr": atr,
            "meta": {"diag": {"reason": "light"}, "factors_json": factors_json},
            "risk": {
                "hold_bars": self._hold_bars,
                "cooldown_remaining": self._cooldown_remaining,
                "consecutive_losses": 0,
            },
            "factors_json": factors_json,
        }
