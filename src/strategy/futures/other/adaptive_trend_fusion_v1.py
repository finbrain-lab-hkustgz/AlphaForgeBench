"""
自适应趋势融合策略 — Binance USDT-M 永续合约 (BTCUSDT)

一个保守的多因子趋势跟踪策略，针对 BTC 永续合约的稳定、低风险盈利进行优化。

设计理念: **少交易、交易好、让盈利奔跑（在 1m 上优先捕捉 squeeze→breakout 的"少数爆发行情"）**
  - 仅在趋势方向明确时入场（EMA 快慢线排列 + 宏观 EMA 方向一致）
  - 需要多个入场条件同时满足（评分系统，而非单一指标）
  - 默认使用 ROI 模式 TP/SL（与 Binance 平台一致）
  - 1m 重点优化：从"压缩禁止交易"升级为"压缩识别→放量突破入场"
  - 可选固定止盈止损 / 追踪止损模式
  - 内置风控机制：止损后冷却、连续亏损暂停（带自动超时重置，防止永久锁死）
  - 手续费/滑点模型，计算真实有效的风险回报比

信号生成（多因子评分系统）:
  1. 市场状态检测: squeeze（BB Width 低位持续）→ 进入"待发射"状态
  2. squeeze breakout 入场（优先级最高）: BB Width 扩张 + 放量 + 突破 swing high/low
  3. 趋势方向判定（趋势延续模式）: 价格 vs EMA 慢线 + EMA 快/慢线排列
  4. 方向偏置过滤: VWAP / 宏观 EMA / ROC 方向一致才允许入场
  5. 趋势延续入场触发条件（需要 >= min_entry_conditions=2.0 分数）:
     - EMA 金叉/死叉（过渡信号）       → 1 分
     - 突破波动高低点 + 成交量放大      → 1 分（lookback=60bar）
     - RSI 从超卖/超买极值回升 (38/62)  → 1 分
     - 价格回踩 EMA 快线后继续趋势方向  → 1 分
  6. 置信度加分项:
     - ADX 确认强趋势（> 20）         → +0.5 分
     - 成交量高于均值                  → +0.3~0.5 分

风控机制:
  - 止损后冷却期（20 bar = 1 分钟图上 20 分钟）
  - 最多连续 3 次亏损 → 暂停开仓（200 bar 后自动重置，防止永久锁死）
  - squeeze breakout 路径有 RSI 极值保护（73/27 阈值）
  - 动态持仓时间: 强趋势持仓最长 2.5 × min_hold_bars(15) = 37 bar（约 37 分钟）
  - 仓位管理: 默认按“投入保证金占比”计算仓位（margin_pct），杠杆由 leverage/min/max_leverage 限制
  - 止损止盈: 默认 ROI 模式（TP +20% / SL -12.5%），可选追踪止损、固定价格 TP/SL
  - 手续费模型: 报告往返手续费影响和扣费后实际风险回报比

⚠ 手续费警告（1 分钟 BTC）:
  在 1 分钟 K 线上，BTC ATR(14) ≈ $20-50。往返 taker 手续费（0.04%）在
  $100k BTC 上 ≈ $80/BTC。这意味着手续费可能超过止损距离。
  追踪止损模式通过让盈利单跑得远超手续费成本来缓解此问题。
  固定止盈止损模式建议仅用于 ≥ 5 分钟以上周期。

所有订单统一发送到 POST /api/order/tpsl 接口。

用法:
    strategy = AdaptiveTrendFusionStrategy(symbol="BTCUSDT")
    result = await strategy(df, current_pos=current_pos, equity=equity)
    if result["order"]:
        await http_client.post("/api/order/tpsl", json=result["order"])
    # 追踪止损更新:
    if result["trailing_stop"]["should_update"]:
        await http_client.post("/api/order/tpsl", json={
            "symbol": "BTCUSDT", "side": "SELL" if current_pos > 0 else "BUY",
            "type": "STOP_MARKET", "stop_loss": result["trailing_stop"]["sl_price"],
            ...
        })
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import Field
import numpy as np
import pandas as pd

from src.strategy.types import Strategy
from src.factor import factor_manager
from src.strategy.futures._utils import (
    PRICE_COL, VOL_COL, DEFAULT_TAKER_FEE,
    compute_fee_info, build_factors_json,
)


# ---------------------------------------------------------------------------
# 仓位管理与杠杆计算工具
# ---------------------------------------------------------------------------

class PositionSizer:
    """仓位计算静态方法集合。

    所有方法返回字典，包含:
      - quantity  : 下单数量（如 0.003 BTC）
      - leverage  : 建议杠杆倍数
      - risk_usdt : 本次交易的美元风险金额
    """

    @staticmethod
    def fixed(
        quantity: float,
        leverage: int,
        price: float,
    ) -> Dict[str, float]:
        """固定数量和杠杆。"""
        return {
            "quantity": quantity,
            "leverage": leverage,
            "notional": quantity * price,
            "risk_usdt": 0.0,
            "method": "fixed",
        }

    @staticmethod
    def notional_percent(
        equity: float,
        notional_pct: float,
        price: float,
        leverage: int = 1,
        max_leverage: int = 20,
        min_quantity: float = 0.0,
        quantity_precision: int = 3,
    ) -> Dict[str, float]:
        """按名义价值占权益比例开仓（notional = equity * notional_pct）。"""
        if equity <= 0 or price <= 0:
            return {
                "quantity": 0.0,
                "leverage": max(1, int(leverage or 1)),
                "notional": 0.0,
                "risk_usdt": 0.0,
                "method": "notional_pct",
            }

        pct = max(0.0, float(notional_pct or 0.0))
        notional = equity * pct
        quantity = notional / price
        if min_quantity and min_quantity > 0:
            quantity = max(quantity, float(min_quantity))
        quantity = round(quantity, quantity_precision)
        notional = quantity * price

        lev = max(1, int(leverage or 1))
        lev = min(lev, int(max_leverage))
        return {
            "quantity": float(quantity),
            "leverage": lev,
            "notional": float(notional),
            "risk_usdt": 0.0,
            "notional_pct": float(pct),
            "method": "notional_pct",
        }

    @staticmethod
    def margin_percent(
        equity: float,
        margin_pct: float,
        price: float,
        leverage: int = 1,
        max_leverage: int = 20,
        min_quantity: float = 0.0,
        quantity_precision: int = 3,
    ) -> Dict[str, float]:
        """按投入保证金占权益比例开仓（margin = equity * margin_pct, notional = margin * leverage）。"""
        if equity <= 0 or price <= 0:
            return {
                "quantity": 0.0,
                "leverage": max(1, int(leverage or 1)),
                "notional": 0.0,
                "risk_usdt": 0.0,
                "method": "margin_pct",
            }

        pct = max(0.0, float(margin_pct or 0.0))
        lev = max(1, int(leverage or 1))
        lev = min(lev, int(max_leverage))
        margin_usdt = equity * pct
        notional = margin_usdt * float(lev)
        quantity = notional / price
        if min_quantity and min_quantity > 0:
            quantity = max(quantity, float(min_quantity))
        quantity = round(quantity, quantity_precision)
        notional = quantity * price
        margin_usdt = notional / float(lev) if lev > 0 else 0.0
        return {
            "quantity": float(quantity),
            "leverage": lev,
            "notional": float(notional),
            "margin_usdt": float(margin_usdt),
            "margin_pct": float(pct),
            "risk_usdt": 0.0,
            "method": "margin_pct",
        }

    @staticmethod
    def risk_percent(
        equity: float,
        risk_pct: float,
        price: float,
        atr: float,
        atr_sl_multiplier: float = 2.0,
        max_leverage: int = 20,
        min_quantity: float = 0.02,
        quantity_precision: int = 3,
    ) -> Dict[str, float]:
        """按权益百分比固定风险。"""
        if atr <= 0 or price <= 0 or equity <= 0:
            return {"quantity": min_quantity, "leverage": 1, "notional": min_quantity * price,
                    "risk_usdt": 0.0, "method": "risk_percent"}

        risk_usdt = equity * risk_pct
        sl_distance = atr * atr_sl_multiplier

        quantity = risk_usdt / sl_distance
        quantity = max(quantity, min_quantity)
        quantity = round(quantity, quantity_precision)

        notional = quantity * price
        leverage = max(1, int(np.ceil(notional / equity)))
        leverage = min(leverage, max_leverage)

        return {
            "quantity": quantity,
            "leverage": leverage,
            "notional": notional,
            "risk_usdt": risk_usdt,
            "sl_distance": sl_distance,
            "method": "risk_percent",
        }

    @staticmethod
    def volatility_target(
        equity: float,
        target_vol: float,
        price: float,
        atr: float,
        max_leverage: int = 20,
        min_quantity: float = 0.02,
        quantity_precision: int = 3,
    ) -> Dict[str, float]:
        """反向波动率仓位管理 — 目标恒定投资组合波动率。"""
        if atr <= 0 or price <= 0 or equity <= 0:
            return {"quantity": min_quantity, "leverage": 1, "notional": min_quantity * price,
                    "risk_usdt": 0.0, "method": "volatility_target"}

        daily_vol_pct = atr / price
        if daily_vol_pct <= 0:
            daily_vol_pct = 0.01

        target_notional = equity * target_vol / daily_vol_pct
        quantity = target_notional / price
        quantity = max(quantity, min_quantity)
        quantity = round(quantity, quantity_precision)

        notional = quantity * price
        leverage = max(1, int(np.ceil(notional / equity)))
        leverage = min(leverage, max_leverage)

        return {
            "quantity": quantity,
            "leverage": leverage,
            "notional": notional,
            "daily_vol_pct": daily_vol_pct,
            "method": "volatility_target",
        }

    @staticmethod
    def kelly(
        equity: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        price: float,
        fraction: float = 0.5,
        max_leverage: int = 20,
        min_quantity: float = 0.02,
        quantity_precision: int = 3,
    ) -> Dict[str, float]:
        """凯利公式 — 数学上最优的仓位管理（使用半凯利更安全）。"""
        if avg_loss <= 0 or equity <= 0 or price <= 0:
            return {"quantity": min_quantity, "leverage": 1, "notional": min_quantity * price,
                    "risk_usdt": 0.0, "kelly_f": 0.0, "method": "kelly"}

        payoff_ratio = avg_win / avg_loss
        kelly_f = win_rate - (1.0 - win_rate) / payoff_ratio
        kelly_f = max(0.0, kelly_f) * fraction

        risk_usdt = equity * kelly_f
        notional = risk_usdt
        quantity = notional / price
        quantity = max(quantity, min_quantity)
        quantity = round(quantity, quantity_precision)

        notional = quantity * price
        leverage = max(1, int(np.ceil(notional / equity)))
        leverage = min(leverage, max_leverage)

        return {
            "quantity": quantity,
            "leverage": leverage,
            "notional": notional,
            "risk_usdt": risk_usdt,
            "kelly_f": kelly_f,
            "method": "kelly",
        }


class AdaptiveTrendFusionStrategy(Strategy):
    """自适应趋势融合策略 — Binance USDT-M 永续合约。

    融合 EMA 趋势过滤、EMA 交叉、RSI、ADX、ATR、波动突破、成交量放大
    等多因子，通过评分系统生成单一方向信号，输出 Binance 合约订单。

    默认模式: ATR 追踪止损（让盈利奔跑，快速止损）。
    """

    # ── 策略元信息 ────────────────────────────────────────────
    name: str = Field(default="adaptive_trend_fusion_v1", description="策略名称")
    description: str = Field(
        default="自适应多因子趋势跟踪策略，用于 Binance USDT-M 永续合约",
        description="策略描述",
    )
    factor_names: List[str] = Field(
        default=["ema", "rsi", "adx", "atr", "bb_width", "vwap", "er", "roc"],
        description="使用的因子名称列表",
    )

    # ── 交易参数 ──────────────────────────────────────────────
    symbol: str = Field(default="BTCUSDT", description="交易对")
    quantity: float = Field(default=0.02, description="固定下单数量（sizing_mode='fixed' 时使用）")
    leverage: int = Field(default=5, description="固定杠杆倍数（sizing_mode='fixed' 时使用）")
    quantity_precision: int = Field(default=3, description="下单数量小数位数")
    min_quantity: float = Field(default=0.001, description="最小下单数量")

    # ── 仓位管理 ──────────────────────────────────────────────
    sizing_mode: str = Field(
        default="margin_pct",
        description="仓位计算模式: fixed/risk_pct/volatility/kelly/notional_pct/margin_pct",
    )
    risk_per_trade: float = Field(default=0.02, description="每笔交易风险占权益比例（2%）")
    atr_sl_multiplier: float = Field(default=2.0, description="止损 ATR 倍数")
    target_vol: float = Field(default=0.10, description="目标日波动率（波动率模式）")
    kelly_win_rate: float = Field(default=0.55, description="历史胜率（凯利模式）")
    kelly_avg_win: float = Field(default=1.5, description="平均盈亏比（凯利模式）")
    kelly_fraction: float = Field(default=0.5, description="凯利分数（0.5 = 半凯利）")
    notional_pct: float = Field(default=1.0, description="名义价值占权益比例（sizing_mode='notional_pct'）")
    margin_pct: float = Field(default=0.25, description="投入保证金占权益比例（sizing_mode='margin_pct'）")

    # ── 硬性限制 ──────────────────────────────────────────────
    min_leverage: int = Field(default=1, description="最小杠杆倍数")
    max_leverage: int = Field(default=10, description="最大杠杆倍数")
    min_notional_pct: float = Field(default=0.0, description="最小名义价值占权益比例（0=不强制）")
    max_notional_pct: float = Field(default=10.0, description="最大名义价值占权益比例（建议≈max_leverage）")

    # ── 出场权威（避免 ROI 挂单 vs 策略市价出场冲突）──────────────
    exit_authority: str = Field(
        default="strategy",
        description="出场权威: strategy(仅策略出场，不下 TP/SL 挂单) / exchange(仅下 TP/SL 挂单，不做策略市价出场)",
    )

    # ── 止盈 / 止损 ──────────────────────────────────────────
    use_trailing_stop: bool = Field(default=True, description="是否使用追踪止损")
    atr_tp_multiplier: float = Field(default=5.0, description="止盈 ATR 倍数（仅固定 TP/SL 模式）")
    use_pct_sl_tp: bool = Field(default=True, description="百分比止损止盈模式")
    sl_pct: float = Field(default=0.03, description="百分比止损幅度（3%）")
    tp_pct: float = Field(default=0.10, description="百分比止盈幅度（10%）")

    # ── ROI 止盈止损 ──────────────────────────────────────────
    use_roi_tpsl: bool = Field(default=True, description="是否使用 ROI 模式的 TP/SL")
    tp_roi_pct: float = Field(
        default=20.0,
        description="止盈 ROI%（5x 杠杆下 ≈ 价格涨 4%；宽 TP 让盈利奔跑，配合 ATR 追踪止损使用）",
    )
    sl_roi_pct: float = Field(
        default=10.0,
        description="止损 ROI%（5x 杠杆下 ≈ 价格跌 2%；硬止损兜底，正常出场依赖 ATR 追踪止损）",
    )
    roi_sl_use_structure: bool = Field(default=False, description="是否允许结构止损覆盖 sl_roi_pct（默认禁用）")

    # ── 手续费模型 ────────────────────────────────────────────
    fee_rate: float = Field(default=DEFAULT_TAKER_FEE, description="每侧预估手续费率（0.04%）")

    # ── 指标参数 ──────────────────────────────────────────────
    ema_fast_period: int = Field(default=50, description="EMA 快线周期（用于交叉/回踩）")
    ema_slow_period: int = Field(default=200, description="EMA 慢线周期（用于趋势锚定）")
    ema_trend_buffer: float = Field(default=0.0006, description="EMA 趋势缓冲区（过严会导致方向判定极少触发）")
    adx_period: int = Field(default=14, description="ADX 周期")
    adx_threshold: float = Field(default=28.0, description="ADX 趋势强度阈值（动态持仓 + 评分加分）：原20.0覆盖80%时间区分度为零，提升至28恢复强趋势识别能力")
    adx_no_trade: float = Field(default=12.0, description="ADX 禁止交易阈值（1m BTC ADX 常处于 10~25，12 为合理下限）")
    rsi_period: int = Field(default=14, description="RSI 周期")
    rsi_oversold: float = Field(default=38.0, description="RSI 超卖阈值（评分触发，适合 1m BTC）")
    rsi_overbought: float = Field(default=62.0, description="RSI 超买阈值（评分触发，适合 1m BTC）")
    rsi_pullback_long_max: float = Field(default=45.0, description="趋势做多：回撤后转强的 RSI 上限（如 35~45 区间）")
    rsi_pullback_short_min: float = Field(default=55.0, description="趋势做空：反弹后转弱的 RSI 下限（如 55~65 区间）")
    rsi_extreme_bonus: float = Field(default=0.5, description="RSI 极值反转额外加分（避免只吃极端也能交易）")

    trend_align_score: float = Field(default=0.4, description="趋势对齐常驻基础分（EMA 快慢线每项）")
    env_align_score: float = Field(default=0.4, description="趋势环境加分（ER/BBWidth/ADX 等综合通过时）")
    min_event_points: float = Field(default=0.5, description="入场至少需要的“事件触发分”最低值（避免纯对齐导致小亏损单）")
    low_event_adx_override: float = Field(default=35.0, description="若 ADX 很强可放宽事件触发要求")
    require_roc_confirm_for_trend_entry: bool = Field(
        default=True,
        description="趋势延续入场是否要求 ROC 方向确认（做多 roc>=1+confirm_buf，做空 roc<=1-confirm_buf）",
    )
    strong_trend_override_requires_roc: bool = Field(
        default=True,
        description="当用强趋势覆盖 event_points 时，仍要求 ROC 确认，避免 ADX 高但动量不足的噪声入场",
    )
    strong_trend_er_boost: float = Field(
        default=0.03,
        description="强趋势覆盖 event_points 时额外要求的 ER 提升（ER >= er_min_entry + boost）",
    )
    roc_confirm_score: float = Field(
        default=0.8,
        description="ROC 顺势确认加分：用于波动市避免过度依赖 EMA 交叉/突破等噪声事件，提升趋势跟随稳定性",
    )
    roc_confirm_event_points: float = Field(
        default=0.5,
        description="ROC 顺势确认贡献的 event_points（0.5=半个事件）：用来满足 min_event_points 以避免纯对齐入场",
    )

    # ── Trigger 4: 回踩入场参数 ──────────────────────────────
    pullback_ema_lo: float = Field(default=-0.001, description="回踩 EMA 偏离下限（做多方向）")
    pullback_ema_hi: float = Field(default=0.0015, description="回踩 EMA 偏离上限（做多方向）")
    pullback_rsi_long_max: float = Field(default=42.0, description="回踩做多 RSI 上限")
    pullback_rsi_short_min: float = Field(default=58.0, description="回踩做空 RSI 下限")
    swing_lookback: int = Field(default=60, description="波动高低点回看周期（60 bar=1小时）")
    vol_surge_lookback: int = Field(default=30, description="成交量放大回看周期（过长会稀释信号）")
    vol_surge_multiplier: float = Field(default=1.2, description="成交量放大倍数阈值（过高会导致几乎不交易）")
    atr_period: int = Field(default=14, description="ATR 周期")

    # ── DI 方向确认 ───────────────────────────────────────────
    use_di_filter: bool = Field(default=True, description="是否启用 +DI/-DI 方向确认")
    di_min_diff: float = Field(default=2.0, description="DI 最小优势差值")

    # ── 趋势度 / 震荡过滤 ────────────────────────────────────
    er_period: int = Field(default=60, description="Efficiency Ratio 周期")
    er_min_entry: float = Field(default=0.10, description="趋势延续入场最低趋势度阈值（1m BTC 震荡期 ER 中位数约 0.10；原0.15导致趋势方向满足时仍有64%时间被ER截流）")
    er_min_entry_breakout: float = Field(default=0.10, description="squeeze→breakout 最低趋势度阈值（比趋势延续稍宽松，但仍需有一定方向性）")
    er_min_reverse_close: float = Field(default=0.30, description="反向平仓最低趋势度")
    er_max_exit: float = Field(default=0.30, description="独立出场 ER 上限（低于此值允许更果断出场）")

    # ── squeeze→breakout ──────────────────────────────────────
    breakout_confirm_window: int = Field(default=12, description="突破后等待回踩确认的窗口 bar 数")
    breakout_retest_buf_pct: float = Field(default=0.0008, description="回踩允许偏离缓冲（0.08%）")
    breakout_confirm_buf_pct: float = Field(default=0.0003, description="确认入场超越缓冲（0.03%）")
    breakout_cancel_buf_pct: float = Field(default=0.0010, description="取消 pending 缓冲（0.1%）")
    breakout_chase_atr: float = Field(
        default=0.6,
        description="趋势延续突破触发时允许追价距离（ATR 倍数）；避免在 RSI 极值后追涨杀跌",
    )

    # ── BB Width squeeze / regime ─────────────────────────────
    bb_width_period: int = Field(default=20, description="BB Width 计算周期")
    bb_width_min_entry: float = Field(default=0.005, description="BB Width squeeze 阈值（0.5%）：原0.8%在1分钟BTC下89%时间被触发导致概念失效，降至0.5%恢复区分度")
    bb_width_trend_min: float = Field(default=0.003, description="趋势延续模式最小 BB Width（1m BTC 正常波动 0.3%+ 即满足）")
    squeeze_min_bars: int = Field(default=20, description="触发 squeeze 的最少连续 bar 数（1m 图上 20 分钟压缩即有效）")
    squeeze_breakout_window: int = Field(default=20, description="squeeze 后允许突破入场的窗口 bar 数")
    bb_width_expansion_pct: float = Field(default=0.15, description="BB Width 扩张幅度阈值（15%）")
    use_squeeze_breakout: bool = Field(default=True, description="是否启用 squeeze→breakout 入场")

    # ── VWAP 方向偏置过滤 ────────────────────────────────────
    vwap_period: int = Field(default=100, description="滚动 VWAP 计算周期")
    use_vwap_filter: bool = Field(default=True, description="是否启用 VWAP 方向偏置过滤")
    vwap_filter_buf_pct: float = Field(
        default=0.003,
        description="VWAP 过滤缓冲（0.3%）：放宽至0.3%使 VWAP 只在价格明显偏离时阻止，避免在趋势确认后因价格远离 VWAP 被反向拦截",
    )

    # ── 追涨杀跌（耗尽段）保护 ───────────────────────────────
    use_exhaustion_filter: bool = Field(default=True, description="是否启用耗尽段过滤（过滤大量小亏损单）")
    rsi_chase_long: float = Field(default=70.0, description="做多追涨 RSI 阈值（1m BTC 上升趋势 RSI 常在 55~70，70 以上才算真正追涨）")
    rsi_chase_short: float = Field(default=30.0, description="做空追跌 RSI 阈值（1m BTC 下降趋势 RSI 常在 30~45，30 以下才算真正追跌）")

    # ── 宏观趋势斜率过滤 ─────────────────────────────────────
    use_macro_slope: bool = Field(default=True, description="是否启用宏观趋势斜率(ROC)过滤")
    macro_slope_period: int = Field(default=120, description="宏观斜率 ROC 周期（1m BTC 用 120=2小时，足以过滤逆势，不需要 4 小时）")
    macro_slope_buf: float = Field(default=0.0002, description="宏观斜率缓冲（0.02%）")
    macro_slope_flat_buf: float = Field(default=0.002, description="宏观斜率平坦缓冲（放宽至 0.2%，只拦截真正横盘，不拦截微弱趋势）")
    macro_slope_confirm_buf: float = Field(
        default=0.0002,
        description="宏观斜率确认缓冲（0.02%）：与 macro_slope_buf 对齐，避免确认门槛远高于阻止门槛导致信号窒息",
    )
    macro_slope_confirm_margin: float = Field(
        default=0.0001,
        description="ROC 确认安全边际（0.01%）：避免贴阈值通过导致的噪声单（波动市常见小亏单来源）",
    )

    # ── 过度延伸过滤（1m 波动市去噪）───────────────────────────
    use_extension_filter: bool = Field(default=True, description="是否启用过度延伸过滤（避免追涨杀跌后的反抽小亏/中亏）")
    max_vwap_extension_atr: float = Field(
        default=4.0,
        description="相对 VWAP 的最大延伸（ATR 倍数）。超过此值视为追到极限，禁入；1m BTC 建议 3~6",
    )
    max_ema_extension_atr: float = Field(
        default=3.0,
        description="相对 EMA 慢线(EMA200) 的最大延伸（ATR 倍数）。超过此值禁入；1m BTC 建议 2~5",
    )

    # ── RSI 追涨杀跌过滤 ─────────────────────────────────────
    rsi_block_long_over: float = Field(default=72.0, description="做多 RSI 上限（趋势延续路径）")
    rsi_block_short_under: float = Field(default=32.0, description="做空 RSI 下限（趋势延续路径）")
    rsi_block_long_squeeze: float = Field(default=73.0, description="做多 RSI 上限（squeeze breakout 路径）")
    rsi_block_short_squeeze: float = Field(default=27.0, description="做空 RSI 下限（squeeze breakout 路径）")

    # ── 手续费/波动门槛过滤 ──────────────────────────────────
    use_fee_edge_filter: bool = Field(default=True, description="是否启用手续费/波动门槛过滤")
    edge_over_fee_mult: float = Field(default=0.5, description="ATR(%) 需达到 round-trip fee 的倍数（1m BTC ATR% 约 0.05~0.10%，0.5 倍即要求 ATR% ≥ 0.04%）")
    extra_cost_bps: float = Field(
        default=3.0,
        description="额外执行成本缓冲（bps）：用来近似点差+滑点+触发单劣化成交等隐性成本；1m BTC 建议 2~8bps",
    )

    # ── 宏观趋势过滤 ─────────────────────────────────────────
    use_macro_trend: bool = Field(default=True, description="是否启用宏观 EMA 趋势过滤")
    ema_macro_period: int = Field(default=240, description="宏观趋势 EMA 周期（1m BTC 用 240=4小时，需配合 window>=250）")
    macro_ema_hard_buf_pct: float = Field(
        default=0.002,
        description="宏观 EMA 硬兜底缓冲（0.2%）：只在价格明显偏离宏观 EMA 时才阻止逆向；避免宏观 EMA 滞后导致方向判断失真",
    )

    # ── 时间过滤 ─────────────────────────────────────────────
    use_time_filter: bool = Field(default=False, description="是否启用 UTC 小时过滤")
    allowed_utc_hours: List[int] = Field(
        default=list(range(7, 23)),
        description="允许交易的 UTC 小时列表",
    )

    # ── 结构止损 ─────────────────────────────────────────────
    use_structure_sl: bool = Field(default=True, description="是否启用结构止损（swing high/low）")
    structure_sl_buffer_atr: float = Field(default=0.2, description="结构止损缓冲（ATR 倍数）")
    structure_sl_min_pct: float = Field(default=0.015, description="结构止损最小风险百分比（1.5%）")
    structure_sl_max_pct: float = Field(default=0.03, description="结构止损最大风险百分比（3%）")

    # ── 风控参数 ──────────────────────────────────────────────
    min_hold_bars: int = Field(default=15, description="基础最小持仓周期（15 bar=15分钟）")
    min_position_bars: int = Field(default=500, description="最小持仓周期（bars）。除紧急止损外，禁止提前出场。")
    max_hold_bars: int = Field(default=360, description="最大持仓 bar 数保护（1m 图上 360=6小时）：原240会在趋势仍在运行时强制截断盈利单（Trade4实证）")
    cooldown_bars: int = Field(default=20, description="止损后冷却周期（20 bar=20分钟）")
    max_consecutive_losses: int = Field(default=3, description="最大连续亏损次数")
    loss_pause_timeout: int = Field(default=200, description="连续亏损暂停自动重置超时（200 bar）")
    emergency_exit_pct: float = Field(default=0.04, description="紧急止损触发阈值（ROI 4% 亏损，无视 min_hold_bars；与杠杆无关）")
    exit_fee_buffer_mult: float = Field(
        default=1.5,
        description="（已弃用，保留兼容性）弱反转出场的手续费缓冲倍数",
    )
    # ── 出场参数（替代 EMA 交叉出场）────────────────────────────
    exit_ema_atr_mult: float = Field(
        default=2.5,
        description="结构反转出场：价格穿越 EMA 慢线幅度超过此倍数 × ATR 时出场（1m BTC 建议 2.0~3.0；原1.5过小，入场1分钟内正常回调即触发出场）",
    )
    exit_trail_atr: float = Field(
        default=2.0,
        description="ATR 追踪止损倍数：价格从最高水位回落超过此倍数 × ATR 时出场（1m BTC 建议 2.0~3.0）",
    )
    trail_activate_atr: float = Field(
        default=0.8,
        description="追踪止损激活所需的最小浮盈（ATR 倍数）：避免未走出成本区就被噪声洗出；走出后用追踪锁定利润",
    )
    exit_ema_atr_mult_long: Optional[float] = Field(
        default=None,
        description="做多结构反转出场 ATR 倍数（可选覆盖 exit_ema_atr_mult）",
    )
    exit_ema_atr_mult_short: Optional[float] = Field(
        default=None,
        description="做空结构反转出场 ATR 倍数（可选覆盖 exit_ema_atr_mult；做空更容易遇到急反抽，常需要更小的倍数）",
    )
    exit_trail_atr_long: Optional[float] = Field(
        default=None,
        description="做多 ATR 追踪止损倍数（可选覆盖 exit_trail_atr）",
    )
    exit_trail_atr_short: Optional[float] = Field(
        default=None,
        description="做空 ATR 追踪止损倍数（可选覆盖 exit_trail_atr；做空反抽风险更大，常需要更小的倍数）",
    )
    max_hold_bars_long: Optional[int] = Field(
        default=None,
        description="做多最大持仓 bar 数（可选覆盖 max_hold_bars）",
    )
    max_hold_bars_short: Optional[int] = Field(
        default=None,
        description="做空最大持仓 bar 数（可选覆盖 max_hold_bars；做空可更短以降低反抽风险）",
    )

    # ── 入场门槛（1m 永续去噪）─────────────────────────────────
    require_trend_env_for_trend_entry: bool = Field(
        default=True,
        description="趋势延续入场是否要求趋势环境（ADX/BBWidth）作为硬门槛；打开可显著减少震荡期噪声单",
    )
    min_entry_conditions: float = Field(default=2.0, description="最低入场评分（满分 5.0）")
    long_only: bool = Field(default=False, description="只做多模式")

    # ── 反向信号处理 ─────────────────────────────────────────
    flip_mode: str = Field(
        default="close_then_wait",
        description="反向信号处理: flip / close_only / close_then_wait",
    )
    reverse_cooldown_bars: int = Field(default=15, description="close_then_wait 后等待 bar 数")

    # ── 金字塔加仓（默认禁用）────────────────────────────────
    max_pyramid_adds: int = Field(default=0, description="同方向最多加仓次数（0=禁用）")
    pyramid_score_boost: float = Field(default=1.0, description="加仓比首次入场额外需要的评分")
    pyramid_min_interval: int = Field(default=5, description="两次加仓之间最少间隔 bar 数")
    pyramid_profit_atr: float = Field(default=1.0, description="价格需朝有利方向移动 >= 此值 × ATR 后才允许加仓")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_side: int = 0
        self._hold_bars: int = 0
        self._cooldown_remaining: int = 0
        self._consecutive_losses: int = 0
        self._loss_pause_bars: int = 0
        self._last_position: float = 0.0
        self._entry_price: Optional[float] = None
        self._best_price: Optional[float] = None
        self._pending_breakout_is_squeeze: bool = False
        self._squeeze_count: int = 0
        self._squeeze_armed: bool = False
        self._squeeze_window_left: int = 0
        self._last_swing_high: Optional[float] = None
        self._last_swing_low: Optional[float] = None
        self._reverse_cooldown_remaining: int = 0
        self._pending_breakout_dir: int = 0
        self._pending_breakout_level: Optional[float] = None
        self._pending_breakout_ttl: int = 0
        self._pending_breakout_retested: bool = False
        self._last_ema_fast: Optional[float] = None
        self._last_ema_slow: Optional[float] = None
        self._pyramid_count: int = 0
        self._last_add_bar: int = 0
        self._entry_leverage: int = self.min_leverage

    # ==================================================================
    # 静态辅助
    # ==================================================================

    @staticmethod
    def _find_swing_points(
        high: pd.Series, low: pd.Series, lookback: int = 20,
    ) -> Tuple[Optional[float], Optional[float]]:
        """查找近期波动高低点。排除当前 bar 以避免前瞻偏差。"""
        if len(high) < lookback * 2:
            return None, None
        return high.iloc[-(lookback + 1):-1].max(), low.iloc[-(lookback + 1):-1].min()

    @staticmethod
    def _check_volume_surge(
        volumes: pd.Series, current_vol: float,
        lookback: int = 20, multiplier: float = 1.5,
    ) -> bool:
        """检测当前成交量是否为近期均值的 multiplier 倍。"""
        if len(volumes) < lookback:
            return False
        avg = volumes.iloc[-lookback:].mean()
        return avg > 0 and current_vol >= avg * multiplier

    # ==================================================================
    # 风控辅助
    # ==================================================================

    def _dynamic_hold_bars(self, adx: float) -> int:
        """根据 ADX 计算动态最小持仓周期（强趋势持仓更久，上限 2.5×）。"""
        base = self.min_hold_bars
        if adx > self.adx_threshold:
            multiplier = min(adx / self.adx_threshold, 2.5)
        else:
            multiplier = 1.0
        return int(base * multiplier)

    def _clear_pending(self) -> None:
        """Reset all pending-breakout state."""
        self._pending_breakout_dir = 0
        self._pending_breakout_level = None
        self._pending_breakout_ttl = 0
        self._pending_breakout_retested = False
        self._pending_breakout_is_squeeze = False

    def _update_risk_state(self, current_pos: float, current_price: float):
        """更新冷却期和连续亏损状态（每 bar 调用一次）。

        说明：若外部能提供真实平仓成交价（如回测 sim_client 的 last_exit_price），
        应优先使用以避免用下一根 close 推断导致的输赢误判。
        """
        inferred_pnl: Optional[float] = None

        position_just_closed = (self._last_position != 0.0 and current_pos == 0.0)
        position_reversed = (
            (self._last_position > 0 and current_pos < 0)
            or (self._last_position < 0 and current_pos > 0)
        )
        old_position_exited = position_just_closed or position_reversed

        exit_price = getattr(self, "_last_exit_price", None)
        px = float(exit_price) if exit_price is not None else float(current_price)

        if old_position_exited and self._entry_price is not None:
            if self._last_position > 0:
                inferred_pnl = (px - self._entry_price) * abs(self._last_position)
            else:
                inferred_pnl = (self._entry_price - px) * abs(self._last_position)

        if inferred_pnl is not None:
            if inferred_pnl < 0:
                self._consecutive_losses += 1
                self._cooldown_remaining = self.cooldown_bars
                self._loss_pause_bars = 0
            else:
                self._consecutive_losses = 0
                self._loss_pause_bars = 0

        if current_pos != 0 and self._last_position == 0:
            self._best_price = None
        elif old_position_exited:
            self._best_price = None
            self._entry_price = None
            self._pyramid_count = 0
            self._last_add_bar = 0
            self._entry_leverage = self.min_leverage

        self._last_position = current_pos

    # ==================================================================
    # 指标读取
    # ==================================================================

    def _read_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """从 DataFrame 读取所有因子值，返回统一的指标字典。"""
        closes = df[PRICE_COL].astype(float)
        highs = df["high"].astype(float) if "high" in df.columns else closes
        lows = df["low"].astype(float) if "low" in df.columns else closes
        volumes = df[VOL_COL].astype(float)

        def _last(col: str) -> float:
            return float(df[col].iloc[-1]) if col in df.columns else float("nan")

        rsi_col = f"rsi_{self.rsi_period}"
        rsi = float(df[rsi_col].iloc[-1])
        rsi_prev = float(df[rsi_col].iloc[-2]) if len(df) > 1 else rsi

        bb_width_col = f"bb_width_{self.bb_width_period}"
        bb_width_prev = float("nan")
        if bb_width_col in df.columns and len(df) >= 2 and not np.isnan(df[bb_width_col].iloc[-2]):
            bb_width_prev = float(df[bb_width_col].iloc[-2])

        swing_high, swing_low = self._find_swing_points(highs, lows, self.swing_lookback)
        self._last_swing_high = swing_high
        self._last_swing_low = swing_low
        vol_surge = self._check_volume_surge(
            volumes, float(volumes.iloc[-1]),
            self.vol_surge_lookback, self.vol_surge_multiplier,
        )
        vol_mean = float(volumes.iloc[-self.vol_surge_lookback:].mean()) if len(volumes) >= self.vol_surge_lookback else float(volumes.mean())
        vol_ratio = float(volumes.iloc[-1] / vol_mean) if (vol_mean is not None and vol_mean > 0) else float("nan")

        return {
            "closes": closes, "highs": highs, "lows": lows, "volumes": volumes,
            "current_price": float(closes.iloc[-1]),
            "current_vol": float(volumes.iloc[-1]),
            "ema_fast_val": _last(f"ema_{self.ema_fast_period}"),
            "ema_slow_val": _last(f"ema_{self.ema_slow_period}"),
            "ema_macro_val": _last(f"ema_{self.ema_macro_period}"),
            "rsi": rsi,
            "rsi_prev": rsi_prev,
            "adx": float(df[f"adx_{self.adx_period}"].iloc[-1]),
            "plus_di": _last(f"plus_di_{self.adx_period}"),
            "minus_di": _last(f"minus_di_{self.adx_period}"),
            "bb_width_val": _last(bb_width_col),
            "bb_width_prev": bb_width_prev,
            "vwap_val": _last(f"vwap_{self.vwap_period}"),
            "er_val": _last(f"er_{self.er_period}"),
            "atr_val": _last(f"atr_{self.atr_period}"),
            "roc_macro": _last(f"roc_{self.macro_slope_period}"),
            "swing_high": swing_high,
            "swing_low": swing_low,
            "vol_surge": vol_surge,
            "vol_ratio": vol_ratio,
        }

    # ==================================================================
    # 统一方向过滤（消除 3 处重复）
    # ==================================================================

    def _is_entry_blocked(
        self, d: int, ind: Dict[str, Any], *,
        is_squeeze: bool = False,
        require_trend_env: bool = False,
        check_di: bool = True,
        check_long_only: bool = True,
    ) -> bool:
        """统一的方向过滤器。返回 True 表示该方向入场被阻止。

        is_squeeze=True 时 RSI 使用宽松阈值(73/27)，ER 使用宽松阈值。
        require_trend_env=True 时额外检查 ADX / BB Width / ER 趋势环境。
        """
        price = ind["current_price"]

        # 宏观斜率（ROC）：下跌趋势(roc<1)阻止做多，上涨趋势(roc>1)阻止做空
        if self.use_macro_slope and not np.isnan(ind["roc_macro"]):
            roc = float(ind["roc_macro"])
            _sb = float(self.macro_slope_buf)
            # 1m 永续的“弱趋势”也足以影响胜率；不应因处于 flat_buf 就跳过方向过滤
            if (d == 1 and roc < 1.0 - _sb) or (d == -1 and roc > 1.0 + _sb):
                return True
        # 宏观趋势 EMA
        if self.use_macro_trend and not np.isnan(ind["ema_macro_val"]):
            # 宏观 EMA 只作为“硬兜底”：当价格明显偏离宏观锚点时才阻止逆向
            buf = float(self.macro_ema_hard_buf_pct)
            if (d == 1 and price < ind["ema_macro_val"] * (1.0 - buf)) or \
               (d == -1 and price > ind["ema_macro_val"] * (1.0 + buf)):
                return True

        # VWAP 方向偏置
        if self.use_vwap_filter and not np.isnan(ind["vwap_val"]):
            vbuf = float(self.vwap_filter_buf_pct)
            if (d == 1 and price < ind["vwap_val"] * (1.0 - vbuf)) or \
               (d == -1 and price > ind["vwap_val"] * (1.0 + vbuf)):
                return True

        # 过度延伸过滤：顺势但过度偏离均值/锚点时，反抽概率显著升高（尤其 1m 永续）
        if self.use_extension_filter:
            atr = ind.get("atr_val", float("nan"))
            if not np.isnan(atr) and atr > 0:
                vwap = ind.get("vwap_val", float("nan"))
                if not np.isnan(vwap):
                    ext_vwap = (price - vwap) / atr
                    if d == 1 and ext_vwap > float(self.max_vwap_extension_atr):
                        return True
                    if d == -1 and ext_vwap < -float(self.max_vwap_extension_atr):
                        return True

                ema_s = ind.get("ema_slow_val", float("nan"))
                if not np.isnan(ema_s):
                    ext_ema = (price - ema_s) / atr
                    if d == 1 and ext_ema > float(self.max_ema_extension_atr):
                        return True
                    if d == -1 and ext_ema < -float(self.max_ema_extension_atr):
                        return True

        # RSI 极值（squeeze 用宽松阈值，趋势延续用严格阈值）
        if is_squeeze:
            if (d == 1 and ind["rsi"] >= self.rsi_block_long_squeeze) or \
               (d == -1 and ind["rsi"] <= self.rsi_block_short_squeeze):
                return True
        else:
            if (d == 1 and ind["rsi"] >= self.rsi_block_long_over) or \
               (d == -1 and ind["rsi"] <= self.rsi_block_short_under):
                return True

        # 耗尽段过滤：过滤大量“刚进场就小亏出场”的噪声单
        if self.use_exhaustion_filter:
            rsi = float(ind["rsi"])
            if d == 1 and rsi >= self.rsi_chase_long:
                return True
            if d == -1 and rsi <= self.rsi_chase_short:
                return True

        # ER 过滤（所有路径通用）：ER 不足说明市场纯震荡，禁止入场
        if not np.isnan(ind["er_val"]):
            er_thresh = self.er_min_entry_breakout if is_squeeze else self.er_min_entry
            if ind["er_val"] < er_thresh:
                return True

        # 趋势环境（ADX/BBWidth，仅在 require_trend_env 时作为硬门槛）
        if require_trend_env:
            if ind["adx"] < self.adx_no_trade:
                return True
            if not np.isnan(ind["bb_width_val"]) and ind["bb_width_val"] < self.bb_width_trend_min:
                return True

        # long_only
        if check_long_only and self.long_only and d == -1:
            return True

        # DI 方向确认
        if check_di and self.use_di_filter \
                and not np.isnan(ind["plus_di"]) and not np.isnan(ind["minus_di"]):
            if (d == 1 and ind["plus_di"] - ind["minus_di"] < self.di_min_diff) or \
               (d == -1 and ind["minus_di"] - ind["plus_di"] < self.di_min_diff):
                return True

        return False

    # ==================================================================
    # 信号生成子方法
    # ==================================================================

    def _check_risk_gates(self, curr_side: int) -> bool:
        """检查风控门控。返回 True 表示禁止交易。"""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return True

        if self._consecutive_losses >= self.max_consecutive_losses:
            self._loss_pause_bars += 1
            if self._loss_pause_bars >= self.loss_pause_timeout:
                self._consecutive_losses = 0
                self._loss_pause_bars = 0
            else:
                return True

        if curr_side == 0 and self._reverse_cooldown_remaining > 0:
            self._reverse_cooldown_remaining -= 1
            return True

        return False

    def _check_env_filters(self, ind: Dict[str, Any], curr_side: int) -> bool:
        """环境级过滤器（手续费门槛、NaN 安全）。返回 True 表示禁止交易。"""
        price = ind["current_price"]
        atr = ind["atr_val"]
        bb_w = ind["bb_width_val"]

        if (self.use_fee_edge_filter and curr_side == 0
                and not np.isnan(atr) and price > 0):
            atr_pct = atr / price
            # 真实执行成本 = taker fee(双边) + 额外隐性成本缓冲（点差/滑点/触发单劣化）
            round_trip_cost = (self.fee_rate * 2.0) + (float(self.extra_cost_bps) / 10000.0)
            if atr_pct < round_trip_cost * self.edge_over_fee_mult:
                return True

        if any(np.isnan(v) for v in [
            ind["ema_fast_val"], ind["ema_slow_val"],
            ind["rsi"], ind["adx"],
        ]):
            return True

        return False

    def _process_pending_breakout(
        self, ind: Dict[str, Any], curr_side: int,
    ) -> Optional[float]:
        """处理 pending breakout 回踩确认。返回信号值或 None（无 pending）。"""
        if not (curr_side == 0
                and self._pending_breakout_dir != 0
                and self._pending_breakout_ttl > 0
                and self._pending_breakout_level is not None):
            return None

        lvl = float(self._pending_breakout_level)
        d = int(self._pending_breakout_dir)
        price = ind["current_price"]
        self._pending_breakout_ttl -= 1

        if (d == 1 and price < lvl * (1 - self.breakout_cancel_buf_pct)) or \
           (d == -1 and price > lvl * (1 + self.breakout_cancel_buf_pct)):
            self._clear_pending()
            return 0.0

        if d == 1:
            if float(ind["lows"].iloc[-1]) <= lvl * (1 + self.breakout_retest_buf_pct):
                self._pending_breakout_retested = True
        else:
            if float(ind["highs"].iloc[-1]) >= lvl * (1 - self.breakout_retest_buf_pct):
                self._pending_breakout_retested = True

        if self._pending_breakout_retested:
            confirmed = (
                (d == 1 and price > lvl * (1 + self.breakout_confirm_buf_pct))
                or (d == -1 and price < lvl * (1 - self.breakout_confirm_buf_pct))
            )
            if confirmed:
                if self.long_only and d == -1:
                    self._clear_pending()
                    return 0.0
                is_sq = bool(self._pending_breakout_is_squeeze)
                blocked = self._is_entry_blocked(
                    d, ind,
                    is_squeeze=is_sq,
                    require_trend_env=False,  # ADX/BBWidth/ER 不作为 pending 确认的硬门槛
                    check_di=False,
                    check_long_only=False,
                )
                if blocked:
                    self._clear_pending()
                    return 0.0
                if self.use_di_filter \
                        and not np.isnan(ind["plus_di"]) and not np.isnan(ind["minus_di"]):
                    di_ok = (
                        (d == 1 and ind["plus_di"] - ind["minus_di"] >= self.di_min_diff)
                        or (d == -1 and ind["minus_di"] - ind["plus_di"] >= self.di_min_diff)
                    )
                    if not di_ok:
                        return 0.0
                self._clear_pending()
                return float(d)

        if self._pending_breakout_ttl <= 0:
            self._clear_pending()
        return 0.0

    def _is_time_blocked(self, df: pd.DataFrame, curr_side: int) -> bool:
        """UTC 小时过滤。返回 True 表示当前时间禁止交易。"""
        if not self.use_time_filter or curr_side != 0:
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
                if int(ts.hour) not in self.allowed_utc_hours:
                    return True
            except Exception:
                pass
        return False

    def _detect_squeeze_breakout(
        self, ind: Dict[str, Any], curr_side: int,
    ) -> Optional[float]:
        """Squeeze->breakout 检测。返回 0.0（进入 pending）或 None（无事件）。"""
        bb_w = ind["bb_width_val"]
        if not (self.use_squeeze_breakout and curr_side == 0 and not np.isnan(bb_w)):
            return None

        in_squeeze = bb_w < self.bb_width_min_entry
        if in_squeeze:
            self._squeeze_count += 1
            if self._squeeze_count >= self.squeeze_min_bars:
                self._squeeze_armed = True
                self._squeeze_window_left = self.squeeze_breakout_window
        else:
            if self._squeeze_armed and self._squeeze_window_left > 0:
                self._squeeze_window_left -= 1
            else:
                self._squeeze_armed = False
                self._squeeze_count = 0
                self._squeeze_window_left = 0

        expanding = (
            not np.isnan(ind["bb_width_prev"]) and ind["bb_width_prev"] > 0
            and (bb_w - ind["bb_width_prev"]) / ind["bb_width_prev"] >= self.bb_width_expansion_pct
        )

        swing_high, swing_low = ind["swing_high"], ind["swing_low"]
        if not (self._squeeze_armed and expanding and ind["vol_surge"]
                and swing_high is not None and swing_low is not None):
            return None

        price = ind["current_price"]
        prev_close = float(ind["closes"].iloc[-2])
        if price > swing_high and prev_close <= swing_high:
            breakout_dir = 1
        elif price < swing_low and prev_close >= swing_low:
            breakout_dir = -1
        else:
            return None

        if self._is_entry_blocked(breakout_dir, ind, is_squeeze=True):
            return None

        self._squeeze_armed = False
        self._squeeze_count = 0
        self._squeeze_window_left = 0
        self._pending_breakout_dir = breakout_dir
        self._pending_breakout_level = float(swing_high if breakout_dir == 1 else swing_low)
        self._pending_breakout_ttl = max(1, int(self.breakout_confirm_window))
        self._pending_breakout_retested = False
        self._pending_breakout_is_squeeze = True
        return 0.0

    def _is_trend_environment(self, ind: Dict[str, Any]) -> bool:
        """检查是否处于适合趋势交易的环境（ADX + ER + BB Width）。"""
        if ind["adx"] < self.adx_no_trade:
            return False
        if not np.isnan(ind["bb_width_val"]) and ind["bb_width_val"] < self.bb_width_trend_min:
            return False
        if not np.isnan(ind["er_val"]) and ind["er_val"] < self.er_min_entry:
            return False
        return True

    def _strong_trend_event_override_ok(self, ind: Dict[str, Any], direction: int) -> bool:
        """当 event_points 不足时，是否允许用“强趋势”覆盖。

        目的：避免 1m 高频噪声下“纯对齐”随意开仓，同时在真正强趋势中允许持仓跟随。
        """
        if not ind.get("trend_env_ok", False):
            return False
        adx = float(ind.get("adx", float("nan")))
        if np.isnan(adx) or adx < float(self.low_event_adx_override):
            return False

        er = ind.get("er_val", float("nan"))
        if not np.isnan(er) and er < float(self.er_min_entry) + float(self.strong_trend_er_boost):
            return False

        if self.use_macro_slope and bool(self.strong_trend_override_requires_roc):
            roc = ind.get("roc_macro", float("nan"))
            if not np.isnan(roc):
                cbuf = float(self.macro_slope_confirm_buf) + float(self.macro_slope_confirm_margin)
                if direction == 1 and roc < 1.0 + cbuf:
                    return False
                if direction == -1 and roc > 1.0 - cbuf:
                    return False

        return True

    def _check_exit(self, ind: Dict[str, Any], curr_side: int) -> Optional[float]:
        """独立出场逻辑，按优先级依次检查：

        1. 紧急止损：ROI 亏损超过 emergency_exit_pct（无视持仓时长）
        2. 结构性反转：价格实质性穿越 EMA 慢线（>= exit_ema_atr_mult × ATR）
        3. ATR 追踪止损：价格从最高水位回落超过 exit_trail_atr × ATR
        4. 最大持仓时长保护

        EMA 快慢线交叉不再作为出场条件——在 1m BTC 上每小时发生数次，
        是极度滞后且噪声极大的信号，会导致在价格已经大幅下跌后才出场。
        """
        if curr_side == 0:
            return None

        # exchange 权威：策略不主动出场，避免与挂单 TP/SL 冲突
        if str(self.exit_authority).lower() == "exchange":
            return None

        price = ind["current_price"]
        ema_s = ind["ema_slow_val"]
        atr = ind["atr_val"]
        ema_mult = (
            float(self.exit_ema_atr_mult_long)
            if (curr_side == 1 and self.exit_ema_atr_mult_long is not None)
            else float(self.exit_ema_atr_mult_short)
            if (curr_side == -1 and self.exit_ema_atr_mult_short is not None)
            else float(self.exit_ema_atr_mult)
        )
        trail_mult = (
            float(self.exit_trail_atr_long)
            if (curr_side == 1 and self.exit_trail_atr_long is not None)
            else float(self.exit_trail_atr_short)
            if (curr_side == -1 and self.exit_trail_atr_short is not None)
            else float(self.exit_trail_atr)
        )
        max_hold = (
            int(self.max_hold_bars_long)
            if (curr_side == 1 and self.max_hold_bars_long is not None)
            else int(self.max_hold_bars_short)
            if (curr_side == -1 and self.max_hold_bars_short is not None)
            else int(self.max_hold_bars)
        )

        pnl_pct = 0.0
        if self._entry_price is not None and self._entry_price > 0:
            pnl_pct = (price - self._entry_price) / self._entry_price * curr_side

        # ── 持仓以来最佳价格跟踪（无论是否启用 ROI TP/SL，都必须更新；否则追踪止损会失效）
        if self._best_price is None:
            self._best_price = price
        else:
            if curr_side == 1 and price > self._best_price:
                self._best_price = price
            if curr_side == -1 and price < self._best_price:
                self._best_price = price

        # ── 1. 紧急止损（ROI 硬止损，无视 min_hold）
        roi_pnl = pnl_pct * self._entry_leverage
        if roi_pnl < -self.emergency_exit_pct:
            return -float(curr_side)

        # ── 最小持仓周期（除紧急止损外禁止提前出场）──
        try:
            min_pos = int(self.min_position_bars)
        except Exception:
            min_pos = 0
        # 避免 min_position_bars > max_hold_bars 时出现“到点立刻被 max_hold 强制平仓”
        if min_pos > 0:
            max_hold = max(int(max_hold), int(min_pos))
        if min_pos > 0 and self._hold_bars < min_pos:
            return None

        # ── 2. 结构性反转：价格穿越慢线的幅度超过 exit_ema_atr_mult × ATR
        # 不用 EMA 交叉（1m 交叉频率过高），改用价格与慢线距离的绝对量判断
        if not np.isnan(ema_s) and not np.isnan(atr) and atr > 0:
            atr_buf = atr * ema_mult
            if curr_side == 1 and price < ema_s - atr_buf:
                return -float(curr_side)
            if curr_side == -1 and price > ema_s + atr_buf:
                return -float(curr_side)

        # ── 3. ATR 追踪止损（需满足动态最小持仓时长，避免噪声洗出）
        min_hold = self._dynamic_hold_bars(ind["adx"])
        if self._hold_bars < min_hold:
            return None

        if not np.isnan(atr) and atr > 0 and self._best_price is not None:
            # 只有当“最佳价格相对入场价”的浮盈超过一定阈值，才启用追踪止损锁定利润；
            # 否则在 1m 噪声下容易频繁小亏/手续费亏。
            if self._entry_price is not None and self._entry_price > 0:
                best_profit = (
                    (self._best_price - self._entry_price)
                    if curr_side == 1
                    else (self._entry_price - self._best_price)
                )
                if best_profit < atr * float(self.trail_activate_atr):
                    return None

            trail_dist = atr * trail_mult
            if curr_side == 1 and price < self._best_price - trail_dist:
                return -float(curr_side)
            if curr_side == -1 and price > self._best_price + trail_dist:
                return -float(curr_side)

        # ── 4. 最大持仓时长保护
        if self._hold_bars >= max_hold:
            return -float(curr_side)

        return None


    def _update_hold_bars(self, curr_side: int) -> None:
        """更新持仓 bar 计数。"""
        if curr_side == 0:
            self._hold_bars = 0
            self._last_side = 0
            self._pyramid_count = 0
            self._last_add_bar = 0
        elif curr_side == self._last_side:
            self._hold_bars += 1
        else:
            self._hold_bars = 1
            self._last_side = curr_side

    def _evaluate_trend_entry(
        self, ind: Dict[str, Any], curr_side: int,
    ) -> float:
        """多因子评分系统趋势延续入场。

        触发条件（每项 1 分）:
          1. EMA 金叉/死叉刚发生
          2. 突破波动高低点 + 成交量放大
          3. RSI 从超卖/超买极值回升
          4. 价格回踩 EMA 快线

        置信度加分项（每项 0.3~0.5 分）:
          A. ADX 确认强趋势
          B. 成交量高于均值

        返回方向信号或 0.0。
        """
        price = ind["current_price"]
        buf = self.ema_trend_buffer
        ema_fast_val = ind["ema_fast_val"]
        ema_slow_val = ind["ema_slow_val"]

        if np.isnan(ema_fast_val) or np.isnan(ema_slow_val):
            self._last_ema_fast = ema_fast_val
            self._last_ema_slow = ema_slow_val
            self._update_hold_bars(curr_side)
            return 0.0

        trend_long = (
            price > ema_slow_val * (1 + buf)
            and ema_fast_val > ema_slow_val * 1.0005
        )
        trend_short = (
            price < ema_slow_val * (1 - buf)
            and ema_fast_val < ema_slow_val * 0.9995
        )

        if not trend_long and not trend_short:
            self._last_ema_fast = ema_fast_val
            self._last_ema_slow = ema_slow_val
            self._update_hold_bars(curr_side)
            return 0.0

        direction = 1 if trend_long else -1

        if curr_side == 0:
            # 趋势延续入场：只检查方向性过滤（宏观趋势/VWAP/RSI极值/耗尽），
            # ADX/BBWidth/ER 趋势环境不再作为硬门槛（已在 env_align_score 中体现）
            if self._is_entry_blocked(direction, ind, require_trend_env=bool(self.require_trend_env_for_trend_entry)):
                self._last_ema_fast = ema_fast_val
                self._last_ema_slow = ema_slow_val
                self._update_hold_bars(curr_side)
                return 0.0

            # 方向动量确认：避免 ROC 仅“没反向”但仍缺乏顺势动量时开仓
            if self.use_macro_slope and bool(self.require_roc_confirm_for_trend_entry) and not np.isnan(ind["roc_macro"]):
                roc = float(ind["roc_macro"])
                cbuf = float(self.macro_slope_confirm_buf) + float(self.macro_slope_confirm_margin)
                if direction == 1 and roc < 1.0 + cbuf:
                    self._update_hold_bars(curr_side)
                    return 0.0
                if direction == -1 and roc > 1.0 - cbuf:
                    self._update_hold_bars(curr_side)
                    return 0.0

        if self.long_only and direction == -1:
            self._last_ema_fast = ema_fast_val
            self._last_ema_slow = ema_slow_val
            self._update_hold_bars(curr_side)
            return 0.0

        # ── 趋势延续摆动突破 → pending ──────────────────────
        if (curr_side == 0 and self._pending_breakout_dir == 0
                and ind["swing_high"] is not None and ind["swing_low"] is not None):
            prev_close = float(ind["closes"].iloc[-2])
            sh, sl = float(ind["swing_high"]), float(ind["swing_low"])
            # 成交量不再作为硬条件：满足“放量 or 强趋势”即可进入 pending
            vol_ok = bool(ind.get("vol_surge", False))
            adx_ok = (not np.isnan(ind["adx"]) and ind["adx"] >= self.adx_threshold)
            if not (vol_ok or adx_ok):
                self._update_hold_bars(curr_side)
                return 0.0
            if price > sh and prev_close <= sh:
                self._pending_breakout_dir = 1
                self._pending_breakout_level = sh
                self._pending_breakout_ttl = max(1, int(self.breakout_confirm_window))
                self._pending_breakout_retested = False
                self._pending_breakout_is_squeeze = False
                self._last_ema_fast = ema_fast_val
                self._last_ema_slow = ema_slow_val
                return 0.0
            if price < sl and prev_close >= sl:
                self._pending_breakout_dir = -1
                self._pending_breakout_level = sl
                self._pending_breakout_ttl = max(1, int(self.breakout_confirm_window))
                self._pending_breakout_retested = False
                self._pending_breakout_is_squeeze = False
                self._last_ema_fast = ema_fast_val
                self._last_ema_slow = ema_slow_val
                return 0.0

        # ── 多因子评分 ──────────────────────────────────────
        bull_score = 0.0
        bear_score = 0.0

        # 常驻基础分（状态型）：趋势方向对齐时就给分，避免完全依赖“事件型触发”导致几乎不交易
        if direction == 1:
            if price > ema_slow_val * (1 + buf):
                bull_score += self.trend_align_score
            if ema_fast_val > ema_slow_val * 1.0005:
                bull_score += self.trend_align_score
        else:
            if price < ema_slow_val * (1 - buf):
                bear_score += self.trend_align_score
            if ema_fast_val < ema_slow_val * 0.9995:
                bear_score += self.trend_align_score

        # 趋势环境加分（ADX/ER/BBWidth）
        if self._is_trend_environment(ind):
            if direction == 1:
                bull_score += self.env_align_score
            else:
                bear_score += self.env_align_score

        # ROC 顺势确认：既作为硬门槛（上面已检查），也作为“半事件”参与评分，
        # 以适配 1m 永续的波动市场（减少对 EMA 交叉/突破的噪声依赖）。
        if self.use_macro_slope and not np.isnan(ind["roc_macro"]):
            roc = float(ind["roc_macro"])
            cbuf = float(self.macro_slope_confirm_buf) + float(self.macro_slope_confirm_margin)
            if direction == 1 and roc >= 1.0 + cbuf:
                bull_score += float(self.roc_confirm_score)
            if direction == -1 and roc <= 1.0 - cbuf:
                bear_score += float(self.roc_confirm_score)

        # Trigger 1: EMA 金叉/死叉（1 分）
        event_points = 0.0
        # ROC 确认计入 event_points（半事件），避免纯对齐开仓
        if self.use_macro_slope and not np.isnan(ind["roc_macro"]):
            roc = float(ind["roc_macro"])
            cbuf = float(self.macro_slope_confirm_buf) + float(self.macro_slope_confirm_margin)
            if direction == 1 and roc >= 1.0 + cbuf:
                event_points += float(self.roc_confirm_event_points)
            if direction == -1 and roc <= 1.0 - cbuf:
                event_points += float(self.roc_confirm_event_points)
        if self._last_ema_fast is not None and self._last_ema_slow is not None:
            prev_fast_above = float(self._last_ema_fast) > float(self._last_ema_slow)
            curr_fast_above = float(ema_fast_val) > float(ema_slow_val)
            if not prev_fast_above and curr_fast_above:
                bull_score += 1.0
                event_points += 1.0
            elif prev_fast_above and not curr_fast_above:
                bear_score += 1.0
                event_points += 1.0

        self._last_ema_fast = ema_fast_val
        self._last_ema_slow = ema_slow_val

        # Trigger 2: 突破波动高低点 + 成交量放大（1 分）
        atr_val = ind["atr_val"]
        if ind["swing_high"] is not None and ind["swing_low"] is not None:
            sh, sl = float(ind["swing_high"]), float(ind["swing_low"])
            if not np.isnan(atr_val) and atr_val > 0:
                # 只给“不过度追价”的突破加分，避免在 RSI 极值附近追涨杀跌
                if price > sh and (price - sh) <= self.breakout_chase_atr * atr_val:
                    bull_score += 1.0
                    event_points += 1.0
                if price < sl and (sl - price) <= self.breakout_chase_atr * atr_val:
                    bear_score += 1.0
                    event_points += 1.0
            else:
                if price > sh:
                    bull_score += 1.0
                    event_points += 1.0
                if price < sl:
                    bear_score += 1.0
                    event_points += 1.0
            # 成交量从硬条件降级为加分项
            if ind.get("vol_surge", False):
                if direction == 1:
                    bull_score += 0.5
                else:
                    bear_score += 0.5

        # Trigger 3: RSI 从超卖/超买极值回升（1 分）
        rsi = ind["rsi"]
        rsi_prev = ind["rsi_prev"]
        # 趋势做多：回撤后转强（更常见：35~45 区间抬头），极值反转额外加分
        if direction == 1 and rsi_prev <= self.rsi_pullback_long_max and rsi > rsi_prev:
            bull_score += 1.0
            event_points += 1.0
            if rsi_prev < self.rsi_oversold:
                bull_score += self.rsi_extreme_bonus
        # 趋势做空：反弹后转弱（更常见：55~65 区间拐头），极值反转额外加分
        if direction == -1 and rsi_prev >= self.rsi_pullback_short_min and rsi < rsi_prev:
            bear_score += 1.0
            event_points += 1.0
            if rsi_prev > self.rsi_overbought:
                bear_score += self.rsi_extreme_bonus

        # Trigger 4: 价格回踩 EMA 快线后继续趋势方向（1 分）
        if ema_fast_val > 0:
            price_to_ema = (price - ema_fast_val) / ema_fast_val
            if direction == 1 and self.pullback_ema_lo <= price_to_ema <= self.pullback_ema_hi and rsi < self.pullback_rsi_long_max:
                bull_score += 1.0
                event_points += 1.0
            if direction == -1 and -self.pullback_ema_hi <= price_to_ema <= -self.pullback_ema_lo and rsi > self.pullback_rsi_short_min:
                bear_score += 1.0
                event_points += 1.0

        # Boost A: ADX 确认强趋势（+0.5 分）
        if ind["adx"] > self.adx_threshold:
            if direction == 1:
                bull_score += 0.5
            else:
                bear_score += 0.5

        # Boost B: 成交量高于均值（+0.5 分）
        vr = ind.get("vol_ratio", float("nan"))
        if not np.isnan(vr):
            if vr >= 1.0:
                if direction == 1:
                    bull_score += 0.3
                else:
                    bear_score += 0.3
            if vr >= 1.3:
                if direction == 1:
                    bull_score += 0.2
                else:
                    bear_score += 0.2

        # ── 评分门槛判断 ────────────────────────────────────
        score = bull_score if direction == 1 else bear_score

        if score < self.min_entry_conditions:
            self._update_hold_bars(curr_side)
            return 0.0

        # 过滤“纯对齐”导致的小亏损单：至少需要一定事件触发；
        # 但在真正强趋势中（ADX+ER+ROC 同时确认）允许覆盖，否则容易错过趋势跟随机会。
        if curr_side == 0 and event_points < self.min_event_points:
            if not self._strong_trend_event_override_ok(ind, direction):
                self._update_hold_bars(curr_side)
                return 0.0

        signal = float(direction)
        self._update_hold_bars(curr_side)

        # 同方向仓位 → 金字塔加仓检查
        if curr_side != 0 and int(np.sign(signal)) == curr_side:
            atr_val = ind["atr_val"]
            can_add = True
            if self._pyramid_count >= self.max_pyramid_adds:
                can_add = False
            if score < self.min_entry_conditions + self.pyramid_score_boost:
                can_add = False
            if (self._hold_bars - self._last_add_bar) < self.pyramid_min_interval:
                can_add = False
            if self._entry_price is not None and not np.isnan(atr_val) and atr_val > 0:
                if curr_side == 1:
                    profit_move = price - self._entry_price
                else:
                    profit_move = self._entry_price - price
                if profit_move < self.pyramid_profit_atr * atr_val:
                    can_add = False
            if not can_add:
                return 0.0
            self._pyramid_count += 1
            self._last_add_bar = self._hold_bars
            return signal

        # 反向信号 → 持仓期限 + ER 检查
        if curr_side != 0 and int(np.sign(signal)) != curr_side:
            if self._hold_bars < self._dynamic_hold_bars(ind["adx"]):
                return 0.0
            if not np.isnan(ind["er_val"]) and ind["er_val"] < self.er_min_reverse_close:
                return 0.0

        return signal

    # ==================================================================
    # 信号生成（编排器）
    # ==================================================================

    def _generate_signal(self, df: pd.DataFrame, current_pos: float) -> float:
        """多因子评分信号生成器 — 编排子方法完成过滤级联 + 评分入场。

        返回 +1（做多）、-1（做空）或 0（观望/持仓不变）。
        """
        if df is None or df.empty:
            return 0.0
        if PRICE_COL not in df.columns or VOL_COL not in df.columns:
            return 0.0

        max_period = max(
            self.ema_slow_period,
            self.ema_macro_period,
            self.adx_period,
            self.rsi_period,
            self.atr_period,
            self.swing_lookback,
            self.bb_width_period,
            self.vwap_period,
            self.er_period,
            (self.macro_slope_period if self.use_macro_slope else 1),
        )
        if len(df) < max_period + 10:
            return 0.0

        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)

        if self._check_risk_gates(curr_side):
            return 0.0

        ind = self._read_indicators(df)

        if self._check_env_filters(ind, curr_side):
            return 0.0

        sig = self._process_pending_breakout(ind, curr_side)
        if sig is not None:
            return sig

        if self._is_time_blocked(df, curr_side):
            return 0.0

        sig = self._detect_squeeze_breakout(ind, curr_side)
        if sig is not None:
            return sig

        # 趋势环境不再作为“硬门槛”直接拦截（会导致几乎不交易）
        # 改为在评分系统中提供 env_align_score 加分，让策略“能出手但不乱出手”
        if curr_side == 0:
            ind["trend_env_ok"] = self._is_trend_environment(ind)
        else:
            ind["trend_env_ok"] = True

        sig = self._check_exit(ind, curr_side)
        if sig is not None:
            return sig

        return self._evaluate_trend_entry(ind, curr_side)

    # ==================================================================
    # 仓位管理
    # ==================================================================

    def _clamp_sizing(self, sizing: Dict[str, float], price: float, equity: float) -> Dict[str, float]:
        """强制执行杠杆和名义价值的硬性限制。"""
        qty = sizing["quantity"]
        lev = sizing["leverage"]
        notional = qty * price

        lev = max(int(self.min_leverage), min(int(self.max_leverage), int(lev)))

        if equity > 0:
            min_pct = float(self.min_notional_pct or 0.0)
            max_pct = float(self.max_notional_pct or 0.0)
            min_notional = equity * min_pct if min_pct > 0 else 0.0
            max_notional = equity * max_pct if max_pct > 0 else 0.0
            # 不允许名义价值超过“权益×最大杠杆”（假设最多使用全权益做保证金）
            max_notional_by_lev = equity * float(self.max_leverage)
            if max_notional <= 0:
                max_notional = max_notional_by_lev
            else:
                max_notional = min(max_notional, max_notional_by_lev)

            if min_notional > 0 and notional < min_notional:
                notional = min_notional
            if max_notional > 0 and notional > max_notional:
                notional = max_notional

            qty = notional / price if price > 0 else qty
            # 若资金过小导致“最小下单量”本身就超过最大允许名义价值，则禁止开仓，
            # 避免通过 min_quantity 反向放大仓位。
            min_qty_notional = float(self.min_quantity) * float(price) if price > 0 else 0.0
            if max_notional > 0 and min_qty_notional > max_notional:
                sizing.update({
                    "quantity": 0.0,
                    "leverage": int(self.min_leverage),
                    "notional": 0.0,
                    "blocked": "min_quantity_over_max_notional",
                    "blocked_min_qty_notional": round(min_qty_notional, 4),
                    "blocked_max_notional": round(max_notional, 4),
                    "clamped": True,
                })
                return sizing

            qty = max(qty, self.min_quantity)
            qty = round(qty, self.quantity_precision)
            notional = qty * price
            # round() 可能把 notional 推过 max_notional：若超出则向下取一档 precision
            if max_notional > 0 and price > 0 and notional > max_notional:
                step = 10 ** int(self.quantity_precision)
                qty = np.floor((max_notional / price) * step) / step
                qty = max(float(qty), 0.0)
                notional = qty * price

            required_lev = max(1, int(np.ceil(notional / equity)))
            # 只在“杠杆不足以支撑名义价值”时上调；不再覆盖上游 sizing 的杠杆选择
            lev = max(int(lev), int(required_lev))
            lev = max(int(self.min_leverage), min(int(self.max_leverage), int(lev)))

        sl_distance = sizing.get("sl_distance", 0.0)
        actual_risk_usdt = qty * sl_distance if sl_distance > 0 else 0.0
        actual_risk_pct = actual_risk_usdt / equity if equity > 0 else 0.0

        sizing["quantity"] = float(qty)
        sizing["leverage"] = int(lev)
        sizing["notional"] = float(notional)
        if equity > 0:
            sizing["notional_pct"] = float(round(float(notional) / float(equity), 4))
        sizing["actual_risk_usdt"] = float(round(float(actual_risk_usdt), 4))
        sizing["actual_risk_pct"] = float(round(float(actual_risk_pct), 6))
        sizing["clamped"] = True
        return sizing

    def _calculate_sizing(self, df: pd.DataFrame, equity: float) -> Dict[str, float]:
        """根据 sizing_mode 计算数量和杠杆，然后应用硬性限制。"""
        closes = df[PRICE_COL].astype(float)
        price = closes.iloc[-1]

        if self.sizing_mode == "fixed":
            raw = PositionSizer.fixed(self.quantity, self.leverage, price)
            return self._clamp_sizing(raw, price, equity)

        # 这两种模式不依赖 ATR：直接由权益占比推导 quantity
        if self.sizing_mode == "notional_pct":
            raw = PositionSizer.notional_percent(
                equity=equity,
                notional_pct=self.notional_pct,
                price=price,
                leverage=self.leverage,
                max_leverage=self.max_leverage,
                min_quantity=self.min_quantity,
                quantity_precision=self.quantity_precision,
            )
            return self._clamp_sizing(raw, price, equity)
        if self.sizing_mode == "margin_pct":
            raw = PositionSizer.margin_percent(
                equity=equity,
                margin_pct=self.margin_pct,
                price=price,
                leverage=self.leverage,
                max_leverage=self.max_leverage,
                min_quantity=self.min_quantity,
                quantity_precision=self.quantity_precision,
            )
            return self._clamp_sizing(raw, price, equity)

        atr_col = f"atr_{self.atr_period}"
        atr = float(df[atr_col].iloc[-1]) if atr_col in df.columns else 0.0
        if np.isnan(atr) or atr <= 0:
            raw = PositionSizer.fixed(self.quantity, self.leverage, price)
            return self._clamp_sizing(raw, price, equity)

        if self.sizing_mode == "risk_pct":
            raw = PositionSizer.risk_percent(
                equity=equity, risk_pct=self.risk_per_trade, price=price, atr=atr,
                atr_sl_multiplier=self.atr_sl_multiplier, max_leverage=self.max_leverage,
                min_quantity=self.min_quantity, quantity_precision=self.quantity_precision,
            )
        elif self.sizing_mode == "volatility":
            raw = PositionSizer.volatility_target(
                equity=equity, target_vol=self.target_vol, price=price, atr=atr,
                max_leverage=self.max_leverage, min_quantity=self.min_quantity,
                quantity_precision=self.quantity_precision,
            )
        elif self.sizing_mode == "kelly":
            raw = PositionSizer.kelly(
                equity=equity, win_rate=self.kelly_win_rate, avg_win=self.kelly_avg_win,
                avg_loss=1.0, price=price, fraction=self.kelly_fraction,
                max_leverage=self.max_leverage, min_quantity=self.min_quantity,
                quantity_precision=self.quantity_precision,
            )
        else:
            raw = PositionSizer.fixed(self.quantity, self.leverage, price)

        return self._clamp_sizing(raw, price, equity)

    # ==================================================================
    # 追踪止损
    # ==================================================================

    def _compute_trailing_stop(self, current_pos: float, price: float, atr: float) -> Dict[str, Any]:
        """计算当前追踪止损价位（基于持仓以来最佳价格）。"""
        _inactive = {"active": False, "should_update": False, "sl_price": None,
                     "best_price": None, "entry_price": self._entry_price}
        # exchange 权威：由交易所端 TP/SL 处理，策略不更新追踪止损单
        if str(self.exit_authority).lower() == "exchange":
            return _inactive
        # strategy 权威时允许追踪止损更新；ROI 仅在 exchange 权威时生效
        if self.use_roi_tpsl and str(self.exit_authority).lower() != "strategy":
            return _inactive
        if current_pos == 0 or not self.use_trailing_stop or (atr <= 0 and not self.use_pct_sl_tp):
            return _inactive

        if self.use_pct_sl_tp:
            ref_price = self._entry_price if self._entry_price else price
            sl_distance = ref_price * self.sl_pct
        else:
            sl_distance = atr * self.atr_sl_multiplier

        if current_pos > 0:
            if self._best_price is None or price > self._best_price:
                self._best_price = price
            trailing_sl = round(self._best_price - sl_distance, 2)
        else:
            if self._best_price is None or price < self._best_price:
                self._best_price = price
            trailing_sl = round(self._best_price + sl_distance, 2)

        should_update = False
        if self._entry_price is not None:
            if current_pos > 0:
                initial_sl = self._entry_price - sl_distance
                should_update = trailing_sl > initial_sl + sl_distance * 0.1
            else:
                initial_sl = self._entry_price + sl_distance
                should_update = trailing_sl < initial_sl - sl_distance * 0.1

        return {
            "active": True,
            "should_update": should_update,
            "sl_price": trailing_sl,
            "best_price": self._best_price,
            "entry_price": self._entry_price,
            "sl_distance": round(sl_distance, 2),
            "unrealized_from_best": round(
                abs(price - self._best_price) if self._best_price else 0.0, 2,
            ),
        }

    # ==================================================================
    # SL/TP 距离计算
    # ==================================================================

    def _calc_sl_tp_distances(
        self, signal: int, price: float, atr: float,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """计算止损和止盈距离（含结构止损），返回 (sl_distance, tp_distance, sl_meta)。"""
        sl_distance = 0.0
        tp_distance = 0.0
        sl_meta: Dict[str, Any] = {}

        if self.use_pct_sl_tp:
            sl_distance = price * self.sl_pct
        elif atr > 0:
            sl_distance = atr * self.atr_sl_multiplier

        if self.use_structure_sl and sl_distance > 0:
            buffer = (atr * self.structure_sl_buffer_atr) if atr and atr > 0 else (price * 0.001)
            struct_sl: Optional[float] = None
            if signal == 1 and self._last_swing_low is not None:
                struct_sl = float(self._last_swing_low) - buffer
                if struct_sl >= price:
                    struct_sl = None
            if signal == -1 and self._last_swing_high is not None:
                struct_sl = float(self._last_swing_high) + buffer
                if struct_sl <= price:
                    struct_sl = None

            if struct_sl is not None:
                struct_dist = abs(price - struct_sl)
                min_dist = price * self.structure_sl_min_pct
                max_dist = price * self.structure_sl_max_pct
                struct_dist = min(max(struct_dist, min_dist), max_dist)
                sl_distance = struct_dist
                sl_meta["sl_structure"] = {
                    "enabled": True,
                    "swing_high": self._last_swing_high,
                    "swing_low": self._last_swing_low,
                    "buffer": round(buffer, 2),
                }

        if sl_distance > 0 and not self.use_roi_tpsl and not self.use_trailing_stop:
            if self.use_pct_sl_tp:
                tp_distance = price * self.tp_pct
            else:
                tp_distance = atr * self.atr_tp_multiplier

        return sl_distance, tp_distance, sl_meta

    # ==================================================================
    # 构建订单
    # ==================================================================

    def _build_order(
        self, signal: int, current_pos: float,
        sizing: Dict[str, float], price: float, atr: float,
    ) -> Dict[str, Any]:
        """根据信号构建 Binance 合约订单。"""
        cur_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
        cur_amt = abs(current_pos)

        if signal == 0:
            return {"signal": "HOLD", "order": None, "meta": {}}

        meta: Dict[str, Any] = {}
        open_qty = sizing["quantity"]
        if open_qty is None or float(open_qty) <= 0:
            meta["blocked"] = sizing.get("blocked", "invalid_quantity")
            meta["sizing"] = sizing
            return {"signal": "HOLD", "order": None, "meta": meta}

        if cur_side != 0 and cur_side != signal and self.flip_mode in ("close_only", "close_then_wait"):
            close_side = "SELL" if current_pos > 0 else "BUY"
            close_qty = round(cur_amt, self.quantity_precision)
            if close_qty <= 0:
                return {"signal": "HOLD", "order": None, "meta": {}}

            order: Dict[str, Any] = {
                "symbol": self.symbol, "side": close_side, "type": "MARKET",
                "quantity": close_qty, "position_side": "BOTH", "reduce_only": True,
            }
            meta = {
                "mode": "close_only", "flip_mode": self.flip_mode,
                "close_qty": close_qty, "close_side": close_side,
                "requested_signal": "LONG" if signal == 1 else "SHORT",
            }
            if self.flip_mode == "close_then_wait":
                self._reverse_cooldown_remaining = max(0, int(self.reverse_cooldown_bars))
                meta["reverse_cooldown_bars"] = int(self.reverse_cooldown_bars)
            return {"signal": "CLOSE", "order": order, "meta": meta}

        if cur_side != 0 and cur_side != signal:
            total_qty = cur_amt + open_qty
        else:
            total_qty = open_qty
        total_qty = round(total_qty, self.quantity_precision)

        order_side = "BUY" if signal == 1 else "SELL"
        order: Dict[str, Any] = {
            "symbol": self.symbol, "side": order_side, "type": "MARKET",
            "quantity": total_qty, "position_side": "BOTH",
        }

        sl_distance, tp_distance, sl_meta = self._calc_sl_tp_distances(signal, price, atr)
        meta.update(sl_meta)

        lev_for_roi = max(1, int(sizing.get("leverage", 1) or 1))

        # strategy 权威：不下 TP/SL 挂单，避免与策略出场冲突
        if str(self.exit_authority).lower() == "strategy":
            meta["tp_sl_mode"] = "NONE"
        elif self.use_roi_tpsl:
            # ROI 模式：TP/SL 使用“相对入场价格的百分比”（TP=+20%，SL=-10%~-15%）
            # 触发价由 client 侧按 entry_price 计算（与 dashboard 显示一致：百分比）。
            sl_pct = float(self.sl_roi_pct)
            tp_pct = float(self.tp_roi_pct)

            # 结构止损（如启用）在 ROI(%) 口径下取更“紧”的那个（更小的 adverse pct）
            sl_pct_eff = sl_pct
            sl_meta_choice = "pct"
            if self.roi_sl_use_structure and sl_distance > 0:
                # struct sl_distance 是“价格距离”，需要换算成 ROI(%)：price_pct * leverage
                sl_roi_struct_pct = (sl_distance / price) * 100.0 * float(lev_for_roi)
                if sl_roi_struct_pct > 0:
                    sl_pct_eff = min(sl_pct, float(sl_roi_struct_pct))
                    sl_meta_choice = "structure" if sl_pct_eff == float(sl_roi_struct_pct) else "pct"

            # ROI(%) → 等价“价格距离”（用于 fee/rr 估算）
            # ROI ≈ price_pct * leverage  →  price_pct ≈ ROI / leverage
            sl_distance_eff = price * (sl_pct_eff / 100.0) / float(lev_for_roi)
            tp_distance_eff = price * (tp_pct / 100.0) / float(lev_for_roi)
            sl_distance = sl_distance_eff
            tp_distance = tp_distance_eff

            order["tp_sl_mode"] = "ROI"
            order["stop_loss"] = round(sl_pct_eff, 6)
            order["take_profit"] = round(tp_pct, 6)

            meta.update({
                "tp_sl_mode": "ROI_PCT",
                "sl_pct": round(sl_pct_eff, 4),
                "tp_pct": round(tp_pct, 4),
                "sl_source": sl_meta_choice,
                "sl_distance": round(sl_distance_eff, 2),
                "tp_distance": round(tp_distance_eff, 2),
                "raw_rr": round(tp_distance_eff / sl_distance_eff, 2) if sl_distance_eff > 0 else None,
            })
        elif sl_distance > 0 and str(self.exit_authority).lower() != "strategy":
            sl_price = round(price - sl_distance, 2) if signal == 1 else round(price + sl_distance, 2)
            order["tp_sl_mode"] = "PRICE"
            order["stop_loss"] = sl_price
            meta.update({
                "sl_price": sl_price,
                "sl_distance": round(sl_distance, 2),
                "sl_pct_actual": round(sl_distance / price * 100, 3),
            })

            if tp_distance > 0:
                tp_price = round(price + tp_distance, 2) if signal == 1 else round(price - tp_distance, 2)
                order["take_profit"] = tp_price
                meta.update({
                    "tp_price": tp_price,
                    "tp_distance": round(tp_distance, 2),
                    "tp_pct_actual": round(tp_distance / price * 100, 3),
                    "raw_rr": round(tp_distance / sl_distance, 2),
                })

        self._entry_price = price
        self._best_price = price
        self._entry_leverage = lev_for_roi

        meta["fee"] = compute_fee_info(price, open_qty, sl_distance, tp_distance, self.fee_rate)
        meta["sizing"] = sizing
        meta["entry_price"] = price
        meta["open_qty"] = open_qty
        if str(self.exit_authority).lower() == "strategy":
            meta["mode"] = "strategy_exit"
        elif self.use_roi_tpsl:
            meta["mode"] = "roi_tpsl"
        else:
            meta["mode"] = "trailing_stop" if self.use_trailing_stop else "fixed_tpsl"

        sig_name = "LONG" if signal == 1 else "SHORT"
        return {"signal": sig_name, "order": order, "meta": meta}

    # ==================================================================
    # __call__ — 主入口
    # ==================================================================

    async def __call__(
        self,
        df: pd.DataFrame,
        current_pos: float = 0.0,
        equity: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """根据输入 K 线数据生成 Binance 合约订单。"""
        # 外部若提供“真实平仓成交价”（回测/实盘 fills），用于更准的连败/冷却统计
        self._last_exit_price = kwargs.get("last_exit_price")

        await factor_manager("ema", df)
        await factor_manager("rsi", df)
        await factor_manager("atr", df)
        await factor_manager("adx", df)
        await factor_manager("bb_width", df, periods=[self.bb_width_period])
        await factor_manager("vwap", df, periods=[self.vwap_period])
        if self.use_macro_slope:
            await factor_manager("roc", df, windows=[self.macro_slope_period])
        await factor_manager("er", df, periods=[self.er_period])

        # Ensure EMA columns exist (factor may not generate arbitrary spans)
        for p in (self.ema_fast_period, self.ema_slow_period, self.ema_macro_period):
            col = f"ema_{p}"
            if col not in df.columns:
                df[col] = df["close"].ewm(span=int(p), adjust=False).mean()

        price = float(df["close"].iloc[-1])

        self._update_risk_state(current_pos, price)

        signal = self._generate_signal(df, current_pos)
        sig_int = int(np.sign(signal)) if signal != 0 else 0

        sizing = self._calculate_sizing(df, equity)

        atr_col = f"atr_{self.atr_period}"
        current_atr = float(df[atr_col].iloc[-1]) if atr_col in df.columns and not np.isnan(df[atr_col].iloc[-1]) else 0.0

        result = self._build_order(sig_int, current_pos, sizing, price, current_atr)

        effective_pos = current_pos
        if result["signal"] in ("LONG", "SHORT"):
            effective_pos = sizing["quantity"] if result["signal"] == "LONG" else -sizing["quantity"]
        trailing_stop = self._compute_trailing_stop(effective_pos, price, current_atr)

        factor_cols = [
            f"ema_{self.ema_fast_period}",
            f"ema_{self.ema_slow_period}",
            f"ema_{self.ema_macro_period}",
            f"rsi_{self.rsi_period}",
            f"adx_{self.adx_period}", f"plus_di_{self.adx_period}", f"minus_di_{self.adx_period}",
            f"atr_{self.atr_period}",
            f"bb_width_{self.bb_width_period}",
            f"vwap_{self.vwap_period}",
            f"er_{self.er_period}",
        ]
        if self.use_macro_slope:
            factor_cols.append(f"roc_{self.macro_slope_period}")

        factors_json = build_factors_json(df, factor_cols, extra={
            "hold_bars":          self._hold_bars,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_remaining": self._cooldown_remaining,
        })

        meta = result["meta"]
        meta["factors_json"] = factors_json

        return {
            "signal": result["signal"],
            "order": result["order"],
            "leverage": sizing["leverage"],
            "sizing": sizing,
            "atr": current_atr,
            "meta": meta,
            "factors_json": factors_json,
            "risk": {
                "cooldown_remaining": self._cooldown_remaining,
                "consecutive_losses": self._consecutive_losses,
                "loss_pause_bars": self._loss_pause_bars,
                "hold_bars": self._hold_bars,
            },
            "trailing_stop": trailing_stop,
        }
