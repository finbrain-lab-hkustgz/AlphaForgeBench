"""
LLM Agent 元策略 — Binance USDT-M 永续合约 (BTCUSDT)

一个基于大语言模型的元策略，能够根据实时市场状态从若干子策略中
动态选择最适合当前行情的策略进行交易决策。

设计理念: **智能策略调度，自适应市场环境**
  - Agent 本身不产生交易信号，仅负责"选将"和"监督"
  - 使用 LLM 综合分析市场指标，结合策略的适用场景与禁忌场景，
    选择最匹配当前行情的子策略
  - **周期性复盘**: 当子策略 PnL 低于软阈值时触发 LLM 复盘，
    评估是否继续或强制停止。正常运行时跳过，节省 LLM 开销
  - **历史学习**: 每个周期结束后生成总结报告（含 PnL、交易统计、
    失败原因、市场变化对比），反馈给下次策略选择，使 Agent 迭代改进
  - 子策略保持独立的风控状态，Agent 切换回时状态延续
  - 持仓期间不切换策略，切换前先平仓，避免上下文不一致
  - LLM 失败时自动降级为规则引擎，保证系统持续运行

可选子策略:
  - adaptive_trend_fusion   : 保守多因子趋势跟踪 (ADX > 20)
  - mean_reversion          : 均值回归 (ADX < 22, 震荡市)
  - funding_rate_arbitrage  : 资金费率套利 (极端费率)
  - momentum_breakout       : 动量突破 (波动率压缩后突破)
  - volatility_grid         : 波动率网格 (ADX < 25, 震荡/弱趋势)
  - multi_timeframe_confluence : 多时间框架共振 (多 TF 方向一致)
  - liquidation_cascade     : 清算瀑布猎杀 (成交量爆发 + 价格加速)

用法:
    strategy = AgentFuturesStrategy(symbol="BTCUSDT")
    result = await strategy(df, current_pos=0.0, equity=10000.0, funding_rate=0.0001)
"""

import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, create_model
import numpy as np
import pandas as pd
import time

from src.strategy.types import Strategy
from src.model import model_manager
from src.message.types import HumanMessage, SystemMessage
from src.logger import logger
from src.utils import dedent

from src.strategy.futures.adaptive_trend_fusion import AdaptiveTrendFusionStrategy
from src.strategy.futures.mean_reversion import MeanReversionStrategy
from src.strategy.futures.funding_rate_arbitrage import FundingRateArbitrageStrategy
from src.strategy.futures.momentum_breakout import MomentumBreakoutStrategy
from src.strategy.futures.volatility_grid import VolatilityGridStrategy
from src.strategy.futures.multi_timeframe_confluence import MultiTimeframeConfluenceStrategy
from src.strategy.futures.liquidation_cascade import LiquidationCascadeStrategy


# ---------------------------------------------------------------------------
# 子策略注册表
# ---------------------------------------------------------------------------

_STRATEGY_REGISTRY: Dict[str, type] = {
    "adaptive_trend_fusion": AdaptiveTrendFusionStrategy,
    "mean_reversion": MeanReversionStrategy,
    "funding_rate_arbitrage": FundingRateArbitrageStrategy,
    "momentum_breakout": MomentumBreakoutStrategy,
    "volatility_grid": VolatilityGridStrategy,
    "multi_timeframe_confluence": MultiTimeframeConfluenceStrategy,
    "liquidation_cascade": LiquidationCascadeStrategy,
}

# 策略描述：包含适用场景、失败模式、预期信号模式
_STRATEGY_PROFILES: Dict[str, Dict[str, str]] = {
    "adaptive_trend_fusion": {
        "style": "保守",
        "leverage": "5-10x",
        "description": (
            "自适应趋势融合策略，包含三条入场路径：\n"
            "1) Trend-Follow：Supertrend(1m) 定方向，7 个因子（ADX/ATR/CHOP/MTF/EMA位置/EMA距离/RSI）"
            "连续评分加权，总分超阈值才入场；内置追空末端保护（RSI 极端超卖 + 宏观 EMA 斜率不够陡时禁止追空）"
            "和低位追空禁入（近期超卖且价格处于 2h 区间底部时禁止做空）。\n"
            "2) Range/均值回归：CHOP ≥ 60 判定为震荡行情，独立评分（ADX低位 + ATR + EMA距离 + RSI极端 + EMA位置），"
            "带宏观趋势过滤（EMA 斜率强下行时禁止抄底做多）。\n"
            "3) Chase/极端动量追单：价格大幅偏离 EMA（≥6 ATR）+ 1m 插针或 5-bar 动量突破时快速追入，"
            "使用独立的更紧 TP/SL 和最大持仓 bar 数限制。\n"
            "出场：策略层 TP/SL（ROI%）+ ATR 波动率飙升退出 + 交易所层 STOP_MARKET/TAKE_PROFIT_MARKET 条件单双重保护。"
        ),
        "best_for": (
            "明确的单向趋势行情（ADX > 25，Supertrend + MTF EMA 方向一致）。"
            "趋势越强、越持久，收益越好。震荡行情下自动切换为均值回归子策略。"
        ),
        "fails_when": (
            "长期窄幅横盘（CHOP 长时间处于 transition 区间 46~60），策略会频繁跳过不交易。"
            "V 形急反转中可能因指标滞后而在末端追入被止损。"
        ),
        "signal_pattern": (
            "大部分时间 HOLD，等待多因子评分严格满足才开仓。交易频率极低，HOLD 占比 95%+ 是正常的。"
            "transition 制度期间（CHOP 介于趋势/震荡之间）完全不交易也是设计如此。"
        ),
    },
    "mean_reversion": {
        "style": "平衡",
        "leverage": "3-5x",
        "description": "基于 BB、RSI、Z-Score、VWAP 偏离的多因子梯度评分，在价格偏离均值时逆向入场，回归时获利。",
        "best_for": "震荡市、区间盘整（ADX < 22，价格在 BB 上下轨之间波动）。",
        "fails_when": "强趋势行情中逆势入场会被单边行情碾压。突破后的趋势延续阶段是最大敌人。",
        "signal_pattern": "中等频率交易，持仓时间较短。BUY/SELL 信号较频繁。",
    },
    "funding_rate_arbitrage": {
        "style": "超保守",
        "leverage": "2-3x",
        "description": "当资金费率绝对值超阈值时反向开仓收取费率。依赖 funding_rate 参数。",
        "best_for": "资金费率极端时期（|funding_rate| > 0.0001），尤其是费率持续偏离时。",
        "fails_when": "费率极端但价格剧烈反向运动时（费率收入不足以覆盖价格损失）。费率快速回归正常时持仓无意义。",
        "signal_pattern": "交易非常稀少，大部分时间 HOLD 等待极端费率出现。",
    },
    "momentum_breakout": {
        "style": "适度激进",
        "leverage": "5-8x",
        "description": "检测 BB 收窄（squeeze）后的价格突破，配合 MACD/ROC 动量确认和 EMA 趋势过滤。",
        "best_for": "波动率长期压缩后的突破行情（BB 宽度极低 → 突然放大）。",
        "fails_when": "假突破频繁的行情（突破后立即回落）。震荡市中 BB 反复收窄放大但无方向。",
        "signal_pattern": "等待 squeeze 条件满足，大部分时间 HOLD。突破时激进入场。",
    },
    "volatility_grid": {
        "style": "平衡",
        "leverage": "2-3x",
        "description": "以 EMA 为中心、ATR 为间距构建网格，在网格线触发时均值回归交易。",
        "best_for": "震荡和弱趋势行情（ADX < 25），价格围绕均值上下波动。",
        "fails_when": "强趋势行情中价格持续偏离网格中心，导致网格击穿。高波动率环境下间距过小也会频繁止损。",
        "signal_pattern": "中等频率交易，在网格线触发时开仓。BUY/SELL 交替出现。",
    },
    "multi_timeframe_confluence": {
        "style": "平衡",
        "leverage": "5-7x",
        "description": "通过 1 分钟 + 5 分钟 K 线的 EMA、RSI、MACD、ADX 等多 TF 确认来提高胜率。",
        "best_for": "多个时间框架方向一致时（1m 和 5m 趋势 + 动量共振）。全面确认机制适合不确定行情。",
        "fails_when": "不同时间框架信号矛盾时（1m 看多、5m 看空），会频繁错过机会。快速反转行情中多 TF 确认太慢。",
        "signal_pattern": "由于需要多重确认，信号较稀疏。HOLD 占比高是正常的。",
    },
    "liquidation_cascade": {
        "style": "适度激进",
        "leverage": "5-8x",
        "description": "检测疑似清算瀑布（ROC 急变 + 成交量爆发），等待回调后入场抄底/摸顶。",
        "best_for": "成交量爆发 + 价格急速运动的清算事件。这些事件稀少但利润丰厚。",
        "fails_when": "误判普通波动为清算事件，或清算后无回调直接继续暴跌/暴涨。",
        "signal_pattern": "极低频交易，绝大部分时间 HOLD 等待清算事件。HOLD 占比 98%+ 是正常的。",
    },
}


# ---------------------------------------------------------------------------
# LLM 结构化输出模型
# ---------------------------------------------------------------------------

class AgentDecision(BaseModel):
    """LLM 策略选择决策。"""
    strategy_name: str = Field(
        description="要激活的策略名称（从可用策略列表中选择）"
    )
    reasoning: str = Field(
        description="4-5句话中文解释: (1) 当前市场处于什么状态 (2) 为什么选这个策略 (3) 如果参考了上一周期教训，说明如何避免同样的错误"
    )
    hold_bars: int = Field(
        description="建议持续使用该策略的 bar 数 (60-480)"
    )


def _make_decision_model(strategy_names: List[str]) -> type:
    """动态创建 AgentDecision 模型，strategy_name description 只包含当前启用的策略。"""
    names_str = ", ".join(strategy_names)
    return create_model(
        "AgentDecision",
        __base__=AgentDecision,
        strategy_name=(
            str,
            Field(description=f"要激活的策略名称，必须是以下之一: {names_str}"),
        ),
    )


class AgentReview(BaseModel):
    """LLM 周期性复盘决策。"""
    action: str = Field(
        description="必须是 'continue'（继续使用当前策略）或 'force_stop'（强制停止并切换）"
    )
    reasoning: str = Field(
        description="4-5句话中文解释决策理由: (1) 策略表现评估 (2) 市场状态是否已发生根本变化 (3) 如果 force_stop，说明失败原因"
    )


# ---------------------------------------------------------------------------
# 系统提示词 — 策略选择
# ---------------------------------------------------------------------------

SELECTION_SYSTEM_PROMPT = dedent("""
    你是一个专业的量化交易策略调度 Agent。你的任务是根据当前的市场状态，
    从可用的子策略中选择最适合的一个来执行交易。

    你**不需要**决定具体的买卖方向，只需要选择最合适的策略。

    ## 可用策略

    {strategy_profiles}

    ## 核心决策框架

    **第一步: 识别当前市场状态**
    - 趋势强度（ADX）、趋势方向（EMA 排列）、波动率水平（ATR/BB 宽度）、
      成交量特征（是否异常爆发）、资金费率

    **第二步: 匹配策略的「适用场景」，同时排除「禁忌场景」**
    - 每个策略都有明确的适用场景和失败模式，选择时**必须同时检查两者**
    - 如果当前市场同时满足某策略的适用场景但也部分符合其禁忌场景，
      应降低信心或选择更安全的替代方案

    **第三步: 参考历史教训**
    - 如果提供了上一周期的总结报告，**必须**分析上一策略为什么成功或失败
    - 如果上一策略因为「市场状态与策略不匹配」而亏损，确保这次不犯同样的错误
    - 不要因为上一策略亏损就盲目切换——如果市场状态没变，同一策略仍可能合适

    ## 重要约束

    - **切换有成本**: 每次切换需要平仓（手续费 + 滑点），新策略冷启动无内部状态。
      如果当前市场状态与上一周期相似，考虑继续使用同一策略而非切换
    - **hold_bars 应反映市场状态的稳定性**: 如果市场处于稳定趋势中，可以设较大值
      (240-480)；如果市场正在转型或不确定，设较小值 (60-120) 以便更快复盘
    - **不确定时选最保守的选项**: 如果无法判断市场状态，multi_timeframe_confluence
      的多重确认机制提供了最全面的安全网
    """)

# ---------------------------------------------------------------------------
# 系统提示词 — 周期性复盘
# ---------------------------------------------------------------------------

REVIEW_SYSTEM_PROMPT = dedent("""
    你是一个量化交易策略的监督者。你的任务是评估当前正在运行的子策略的表现，
    决定是继续运行还是立即停止。

    ## 决策框架

    **continue（继续）** — 满足以下任意条件:
    - 累计 PnL 为正，策略在正常盈利
    - 累计 PnL 微亏（> -1%），且 PnL 在改善（从低点回升），是正常回撤
    - 市场状态与选择时相比无根本性变化（关键指标对比显示稳定）
    - 策略的信号模式符合其预期特征（如保守策略大量 HOLD 是正常的）

    **force_stop（强制停止）** — 满足以下任意条件:
    - 累计 PnL < -1.5% 且仍在恶化（与上次复盘相比亏损扩大）
    - 市场状态已经发生根本性变化（如 ADX 从 30 降到 15，趋势消失;
      或 ADX 从 15 升到 30，震荡变趋势），当前策略不再适用
    - 策略运行了大量 bar（> 100）但完全无交易且无合理等待理由

    ## 重要约束

    - **给策略足够的运行空间**: 短期波动和小幅亏损（< -1%）是正常的，不应轻易停止
    - **参考策略预期信号模式**: 保守策略（adaptive_trend_fusion、liquidation_cascade）
      大量 HOLD 是设计如此，不是策略失灵
    - **只在明确证据支持时才 force_stop**: 宁可多给一次机会，也不要频繁切换
      （每次切换有手续费成本）
    """)


# ---------------------------------------------------------------------------
# Agent 元策略
# ---------------------------------------------------------------------------

class AgentFuturesStrategy(Strategy):
    """LLM Agent 元策略 — 智能选择最优子策略并周期性复盘。"""

    # ── 策略元信息 ────────────────────────────────────────────
    name: str = Field(default="agent_futures", description="策略名称")
    description: str = Field(
        default="LLM Agent 元策略，智能调度子策略并周期性复盘",
        description="策略描述",
    )
    factor_names: List[str] = Field(
        default=[],
        description="所有子策略因子的并集（__init__ 中动态计算）",
    )

    # ── 交易参数 ──────────────────────────────────────────────
    symbol: str = Field(default="BTCUSDT", description="交易对")

    # ── LLM 配置 ─────────────────────────────────────────────
    model_name: str = Field(
        default="openrouter/gemini-3-flash-preview",
        description="LLM 模型名称",
    )
    llm_timeout: float = Field(
        default=30.0,
        description="LLM 调用超时秒数",
    )

    # ── 子策略配置 ────────────────────────────────────────────
    enabled_strategies: List[str] = Field(
        default=[
            "adaptive_trend_fusion",
            # "mean_reversion",
            "funding_rate_arbitrage",
            # "momentum_breakout",
            # "volatility_grid",
            # "multi_timeframe_confluence",
            # "liquidation_cascade",
        ],
        description="启用的子策略名称列表（默认全部 7 个）",
    )

    # ── 切换控制 ─────────────────────────────────────────────
    min_hold_bars: int = Field(
        default=60,
        description="最少持续 bar 数，期间不触发复盘",
    )
    max_hold_bars: int = Field(
        default=480,
        description="最多持续 bar 数，到期强制重新评估",
    )
    review_interval: int = Field(
        default=60,
        description="复盘检查间隔（bar 数）。仅当 PnL 低于软阈值时才实际调用 LLM",
    )
    review_soft_threshold: float = Field(
        default=-0.005,
        description="复盘软阈值: 周期 PnL 低于此值时才触发 LLM 复盘，否则跳过节省开销",
    )
    emergency_loss_pct: float = Field(
        default=-0.03,
        description="紧急止损阈值（相对于周期起始权益），不等 LLM 立即触发停止",
    )

    # ── 市场分析参数 ──────────────────────────────────────────
    lookback_bars: int = Field(
        default=100,
        description="传给 LLM 的市场快照回看 bar 数",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ── 实例化子策略 ──────────────────────────────────────
        self._strategies: Dict[str, Strategy] = {}
        all_factor_names: set = set()
        for sname in self.enabled_strategies:
            cls = _STRATEGY_REGISTRY.get(sname)
            if cls is None:
                logger.warning(f"| AgentFutures: 未知策略名 '{sname}'，跳过")
                continue
            instance = cls(symbol=self.symbol)
            self._strategies[sname] = instance
            all_factor_names.update(instance.factor_names)
        self.factor_names = sorted(all_factor_names)

        # ── Agent 全局状态 ────────────────────────────────────
        self._active_strategy_name: Optional[str] = None
        self._hold_bars_remaining: int = 0
        self._last_reasoning: str = ""
        self._bar_count: int = 0
        self._last_equity: float = 0.0

        # ── 周期内状态（_execute_switch 时重置）──────────────
        self._period_start_equity: float = 0.0
        self._period_start_bar: int = 0
        self._last_review_bar: int = 0
        self._last_review_pnl_pct: Optional[float] = None  # 上次复盘时的 PnL（ratio，非百分比）
        self._period_signal_counts: Dict[str, int] = {}  # 信号计数器
        self._period_trade_count: int = 0             # 交易次数（信号转换）
        self._last_signal: str = "HOLD"               # 上一 bar 信号，用于检测转换
        self._selection_time_snapshot: Dict[str, float] = {}  # 选择时的市场快照

        # (已移除: pending_decision / pending_snapshot — 切换不再需要等待平仓)

        # ── 跨周期持久状态 ────────────────────────────────────
        self._last_period_summary: Optional[Dict[str, Any]] = None
        self._switch_history: List[Dict[str, Any]] = []
        self._period_history: List[Dict[str, Any]] = []  # 所有周期的总结记录

    # ==================================================================
    # 工具方法
    # ==================================================================

    @staticmethod
    def _safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
        """安全读取 DataFrame 最后一行某列的值。"""
        if col in df.columns:
            val = df[col].iloc[-1]
            if val is not None and not pd.isna(val):
                return float(val)
        return default

    def _get_market_snapshot(self, df: pd.DataFrame, funding_rate: float = 0.0) -> Dict[str, float]:
        """提取关键市场指标快照（用于选择时保存和复盘时对比）。"""
        price = float(df["close"].iloc[-1])
        return {
            "price": price,
            "adx": self._safe_col(df, "adx_14"),
            "rsi": self._safe_col(df, "rsi_14"),
            "atr": self._safe_col(df, "atr_14"),
            "ema_20": self._safe_col(df, "ema_20"),
            "ema_50": self._safe_col(df, "ema_50"),
            "bb_width_pct": self._calc_bb_width_pct(df, price),
            "vol_ratio": self._calc_vol_ratio(df),
            "funding_rate": funding_rate,
        }

    @staticmethod
    def _calc_bb_width_pct(df: pd.DataFrame, price: float) -> float:
        bb_upper = AgentFuturesStrategy._safe_col(df, "bb_upper_20")
        bb_lower = AgentFuturesStrategy._safe_col(df, "bb_lower_20")
        if bb_upper > 0 and bb_lower > 0 and price > 0:
            return (bb_upper - bb_lower) / price * 100
        return 0.0

    @staticmethod
    def _calc_vol_ratio(df: pd.DataFrame) -> float:
        if "volume" not in df.columns or len(df) < 50:
            return 1.0
        vol_cur = float(df["volume"].iloc[-1])
        vol_avg = float(df["volume"].iloc[-51:-1].mean()) if len(df) >= 51 else float(df["volume"].iloc[:-1].mean()) if len(df) >= 2 else vol_cur
        return vol_cur / vol_avg if vol_avg > 0 else 1.0

    def _get_period_pnl_pct(self, equity: float) -> float:
        """计算当前周期 PnL 百分比。"""
        if self._period_start_equity > 0:
            return (equity - self._period_start_equity) / self._period_start_equity
        return 0.0

    # ==================================================================
    # 信号追踪与交易计数
    # ==================================================================

    def _record_signal(self, signal: str) -> None:
        """记录信号并检测交易转换。"""
        self._period_signal_counts[signal] = self._period_signal_counts.get(signal, 0) + 1

        # 检测交易事件（信号转换）
        if signal != self._last_signal:
            if self._last_signal == "HOLD" and signal in ("LONG", "SHORT"):
                self._period_trade_count += 1  # 开仓
            elif self._last_signal in ("LONG", "SHORT") and signal in ("LONG", "SHORT"):
                self._period_trade_count += 1  # 方向切换
            self._last_signal = signal

    # ==================================================================
    # 周期总结
    # ==================================================================

    def _build_period_summary(
        self,
        outcome: str,
        equity: float,
        extra_reason: str = "",
    ) -> Dict[str, Any]:
        """构建周期总结报告。"""
        bars_run = self._bar_count - self._period_start_bar
        pnl = equity - self._period_start_equity if self._period_start_equity > 0 else 0.0
        pnl_pct = self._get_period_pnl_pct(equity) * 100

        summary: Dict[str, Any] = {
            "strategy": self._active_strategy_name,
            "outcome": outcome,
            "bars_run": bars_run,
            "pnl_usdt": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 3),
            "start_equity": round(self._period_start_equity, 2),
            "end_equity": round(equity, 2),
            "trade_count": self._period_trade_count,
            "signal_distribution": dict(self._period_signal_counts),
            "selection_time_snapshot": self._selection_time_snapshot.copy(),
        }
        if extra_reason:
            summary["stop_reason"] = extra_reason

        # 保存到周期历史
        self._period_history.append(summary)
        if len(self._period_history) > 20:
            self._period_history = self._period_history[-20:]

        logger.info(
            f"| AgentFutures: 周期总结 [{outcome}] | "
            f"策略={self._active_strategy_name} | "
            f"运行 {bars_run} bar | PnL={pnl:+.2f} ({pnl_pct:+.2f}%) | "
            f"交易 {self._period_trade_count} 次"
        )
        return summary

    # ==================================================================
    # 市场上下文提取（供 LLM 策略选择）
    # ==================================================================

    def _extract_market_context(
        self,
        df: pd.DataFrame,
        funding_rate: float = 0.0,
    ) -> str:
        n = min(len(df), self.lookback_bars)
        recent = df.tail(n)
        price = float(df["close"].iloc[-1])

        # 回报计算使用完整 df，覆盖更远历史（1h=60bar, 4h=240bar）
        df_len = len(df)
        price_1h_ago = float(df["close"].iloc[-60]) if df_len >= 60 else float(df["close"].iloc[0])
        price_4h_ago = float(df["close"].iloc[-240]) if df_len >= 240 else float(df["close"].iloc[0])
        ret_1h = (price / price_1h_ago - 1) * 100 if price_1h_ago > 0 else 0.0
        ret_4h = (price / price_4h_ago - 1) * 100 if price_4h_ago > 0 else 0.0

        adx_val = self._safe_col(df, "adx_14")
        ema_20 = self._safe_col(df, "ema_20")
        ema_50 = self._safe_col(df, "ema_50")
        rsi_val = self._safe_col(df, "rsi_14")
        atr_val = self._safe_col(df, "atr_14")
        atr_pct = (atr_val / price * 100) if price > 0 and atr_val > 0 else 0.0
        bb_width = self._calc_bb_width_pct(df, price)
        vol_ratio = self._calc_vol_ratio(df)

        ema_trend = "UNKNOWN"
        if ema_20 > 0 and ema_50 > 0:
            if price > ema_20 > ema_50:
                ema_trend = "BULLISH (价格 > EMA20 > EMA50)"
            elif price < ema_20 < ema_50:
                ema_trend = "BEARISH (价格 < EMA20 < EMA50)"
            else:
                ema_trend = "MIXED (无明确排列)"

        lines = [
            "## 当前市场状态",
            "",
            f"- 价格: {price:,.2f} USDT",
            f"- 近 {n} bar 最高/最低: {float(recent['high'].max()):,.2f} / {float(recent['low'].min()):,.2f}",
            f"- 1h 回报: {ret_1h:+.2f}%  |  4h 回报: {ret_4h:+.2f}%",
            "",
            "### 趋势",
            f"- ADX(14): {adx_val:.1f}" + (" — 强趋势" if adx_val > 25 else " — 弱趋势/震荡" if adx_val < 20 else " — 中等"),
            f"- EMA 排列: {ema_trend}",
            f"- RSI(14): {rsi_val:.1f}" + (" — 超买" if rsi_val > 70 else " — 超卖" if rsi_val < 30 else ""),
            "",
            "### 波动率",
            f"- ATR(14): {atr_val:.2f} ({atr_pct:.3f}%)",
            f"- BB 宽度: {bb_width:.3f}%" + (" — 极窄，可能即将突破" if 0 < bb_width < 0.5 else ""),
            "",
            "### 成交量与其他",
            f"- 成交量/均值比: {vol_ratio:.2f}x" + (" — 异常放大" if vol_ratio > 3.0 else ""),
            f"- 资金费率: {funding_rate:.6f}" + (
                " — 极端正费率" if funding_rate > 0.0001 else
                " — 极端负费率" if funding_rate < -0.0001 else ""
            ),
        ]

        # ── 周期历史（仅显示有记录的）────────────────────────
        if self._period_history:
            lines.append("")
            lines.append("## 历史周期记录（最近 5 个）")
            for rec in self._period_history[-5:]:
                sname = rec.get("strategy", "?")
                outcome = rec.get("outcome", "?")
                pnl_pct = rec.get("pnl_pct", 0)
                trades = rec.get("trade_count", 0)
                bars = rec.get("bars_run", 0)
                reason = rec.get("stop_reason", "")
                line = f"- {sname}: {outcome}, PnL={pnl_pct:+.2f}%, {trades} 笔交易, {bars} bar"
                if reason:
                    line += f" | 停止原因: {reason}"
                lines.append(line)

        # ── 上一周期总结报告 ────────────────────────────────
        if self._last_period_summary:
            s = self._last_period_summary
            lines.append("")
            lines.append("## 上一周期总结 — 请务必参考")
            lines.append(f"- 策略: {s.get('strategy', 'N/A')}")
            lines.append(f"- 结局: {s.get('outcome', 'N/A')}")
            lines.append(f"- 运行 {s.get('bars_run', 0)} bar, PnL={s.get('pnl_usdt', 0):+.2f} USDT ({s.get('pnl_pct', 0):+.2f}%)")
            lines.append(f"- 交易 {s.get('trade_count', 0)} 次, 信号分布: {s.get('signal_distribution', {})}")
            snap = s.get("selection_time_snapshot", {})
            if snap:
                lines.append(f"- 选择时市场: ADX={snap.get('adx', 0):.1f}, RSI={snap.get('rsi', 0):.1f}, BB宽={snap.get('bb_width_pct', 0):.2f}%")
            if s.get("stop_reason"):
                lines.append(f"- 停止原因: {s['stop_reason']}")
            lines.append("")
            lines.append("请分析上一策略成功/失败的原因，在本次选择中避免重复同样的错误。")

        return "\n".join(lines)

    # ==================================================================
    # 复盘上下文提取
    # ==================================================================

    def _extract_review_context(
        self,
        df: pd.DataFrame,
        equity: float,
        funding_rate: float = 0.0,
    ) -> str:
        bars_run = self._bar_count - self._period_start_bar
        pnl_pct = self._get_period_pnl_pct(equity) * 100
        pnl_usdt = equity - self._period_start_equity if self._period_start_equity > 0 else 0.0

        # PnL 趋势方向（_last_review_pnl_pct 存 ratio，pnl_pct 已是百分比）
        pnl_trend = "首次复盘"
        if self._last_review_pnl_pct is not None:
            last_pnl_display = self._last_review_pnl_pct * 100
            if pnl_pct > last_pnl_display + 0.1:
                pnl_trend = "改善中（比上次复盘好）"
            elif pnl_pct < last_pnl_display - 0.1:
                pnl_trend = "恶化中（比上次复盘差）"
            else:
                pnl_trend = "基本持平"

        # 策略预期信号模式
        profile = _STRATEGY_PROFILES.get(self._active_strategy_name or "", {})
        expected_pattern = profile.get("signal_pattern", "未知")

        # 市场状态对比（选择时 vs 当前）
        snap = self._selection_time_snapshot
        cur_adx = self._safe_col(df, "adx_14")
        cur_rsi = self._safe_col(df, "rsi_14")
        cur_price = float(df["close"].iloc[-1])

        lines = [
            "## 当前运行策略复盘",
            "",
            f"- 策略: {self._active_strategy_name}",
            f"- 已运行: {bars_run} bar / 预期 {bars_run + self._hold_bars_remaining} bar",
            f"- 当初选择理由: {self._last_reasoning}",
            "",
            "### 表现数据",
            f"- 累计 PnL: {pnl_usdt:+.2f} USDT ({pnl_pct:+.2f}%)",
            f"- PnL 趋势: {pnl_trend}",
            f"- 交易次数: {self._period_trade_count}",
            f"- 信号分布: {dict(self._period_signal_counts)}",
            f"- 该策略预期信号模式: {expected_pattern}",
            "",
            "### 市场状态对比（选择时 → 当前）",
            f"- ADX: {snap.get('adx', 0):.1f} → {cur_adx:.1f}" + (
                " ⚠ 变化显著" if abs(snap.get('adx', 0) - cur_adx) > 10 else ""
            ),
            f"- RSI: {snap.get('rsi', 0):.1f} → {cur_rsi:.1f}",
            f"- 价格: {snap.get('price', 0):,.2f} → {cur_price:,.2f}",
            f"- BB宽度: {snap.get('bb_width_pct', 0):.2f}% → {self._calc_bb_width_pct(df, cur_price):.2f}%",
            f"- 成交量比: {snap.get('vol_ratio', 1):.2f}x → {self._calc_vol_ratio(df):.2f}x",
            f"- 资金费率: {snap.get('funding_rate', 0):.6f} → {funding_rate:.6f}",
        ]

        return "\n".join(lines)

    # ==================================================================
    # LLM 调用 — 策略选择（带超时）
    # ==================================================================

    async def _select_strategy_via_llm(
        self,
        df: pd.DataFrame,
        funding_rate: float = 0.0,
    ) -> Optional[AgentDecision]:
        # 构建策略 profile 文本
        profile_lines = []
        for sname in self._strategies:
            p = _STRATEGY_PROFILES.get(sname, {})
            profile_lines.append(
                f"### {sname} ({p.get('style', '?')} | {p.get('leverage', '?')})\n"
                f"- 描述: {p.get('description', sname)}\n"
                f"- 适用: {p.get('best_for', '?')}\n"
                f"- 禁忌: {p.get('fails_when', '?')}\n"
                f"- 信号模式: {p.get('signal_pattern', '?')}"
            )
        profiles_text = "\n\n".join(profile_lines)

        system_prompt = SELECTION_SYSTEM_PROMPT.replace("{strategy_profiles}", profiles_text)
        user_prompt = self._extract_market_context(df, funding_rate)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # 动态模型：strategy_name description 只列出当前启用的策略
        DecisionModel = _make_decision_model(list(self._strategies.keys()))

        try:
            logger.info(f"| AgentFutures: 调用 LLM 进行策略选择...")
            response = await asyncio.wait_for(
                model_manager(self.model_name, messages=messages, response_format=DecisionModel),
                timeout=self.llm_timeout,
            )

            if not response.success:
                logger.warning(f"| AgentFutures: LLM 选择失败: {response.message}")
                return None

            decision = getattr(response.extra, "parsed_model", None)
            if decision and isinstance(decision, AgentDecision):
                if decision.strategy_name not in self._strategies:
                    logger.warning(f"| AgentFutures: LLM 返回无效策略名 '{decision.strategy_name}'")
                    return None
                decision.hold_bars = max(self.min_hold_bars, min(self.max_hold_bars, decision.hold_bars))
                logger.info(
                    f"| AgentFutures: LLM 选择 → {decision.strategy_name} "
                    f"(hold_bars={decision.hold_bars}) | {decision.reasoning}"
                )
                return decision
            logger.warning("| AgentFutures: LLM 未返回有效 parsed_model")
            return None

        except asyncio.TimeoutError:
            logger.error(f"| AgentFutures: LLM 选择超时 ({self.llm_timeout}s)")
            return None
        except Exception as e:
            logger.error(f"| AgentFutures: LLM 选择异常: {e}")
            return None

    # ==================================================================
    # LLM 调用 — 周期性复盘（带超时）
    # ==================================================================

    async def _review_strategy_via_llm(
        self,
        df: pd.DataFrame,
        equity: float,
        funding_rate: float = 0.0,
    ) -> Optional[AgentReview]:
        user_prompt = self._extract_review_context(df, equity, funding_rate)
        messages = [
            SystemMessage(content=REVIEW_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            logger.info(f"| AgentFutures: 调用 LLM 进行周期复盘...")
            response = await asyncio.wait_for(
                model_manager(self.model_name, messages=messages, response_format=AgentReview),
                timeout=self.llm_timeout,
            )

            if not response.success:
                logger.warning(f"| AgentFutures: LLM 复盘失败: {response.message}")
                return None

            review = getattr(response.extra, "parsed_model", None)
            if review and isinstance(review, AgentReview):
                review.action = review.action.strip().lower()
                if review.action not in ("continue", "force_stop"):
                    review.action = "continue"
                logger.info(f"| AgentFutures: LLM 复盘 → {review.action} | {review.reasoning}")
                return review
            logger.warning("| AgentFutures: LLM 复盘未返回有效结果")
            return None

        except asyncio.TimeoutError:
            logger.error(f"| AgentFutures: LLM 复盘超时 ({self.llm_timeout}s)")
            return None
        except Exception as e:
            logger.error(f"| AgentFutures: LLM 复盘异常: {e}")
            return None

    # ==================================================================
    # 规则降级 — 策略选择
    # ==================================================================

    def _fallback_select(
        self,
        df: pd.DataFrame,
        funding_rate: float = 0.0,
    ) -> AgentDecision:
        adx = self._safe_col(df, "adx_14")
        bb_width = self._calc_bb_width_pct(df, float(df["close"].iloc[-1]))
        vol_ratio = self._calc_vol_ratio(df)

        if abs(funding_rate) > 0.0001 and "funding_rate_arbitrage" in self._strategies:
            return AgentDecision(strategy_name="funding_rate_arbitrage",
                                reasoning=f"资金费率极端 ({funding_rate:.6f})", hold_bars=self.min_hold_bars)
        if vol_ratio > 5.0 and "liquidation_cascade" in self._strategies:
            return AgentDecision(strategy_name="liquidation_cascade",
                                reasoning=f"成交量异常爆发 ({vol_ratio:.1f}x)", hold_bars=self.min_hold_bars)
        if adx > 25 and "adaptive_trend_fusion" in self._strategies:
            return AgentDecision(strategy_name="adaptive_trend_fusion",
                                reasoning=f"ADX={adx:.1f}>25 强趋势", hold_bars=240)
        if 0 < bb_width < 0.5 and "momentum_breakout" in self._strategies:
            return AgentDecision(strategy_name="momentum_breakout",
                                reasoning=f"BB宽度{bb_width:.3f}% 极窄", hold_bars=120)
        if adx < 20:
            if "mean_reversion" in self._strategies:
                return AgentDecision(strategy_name="mean_reversion",
                                    reasoning=f"ADX={adx:.1f}<20 震荡市", hold_bars=180)
            if "volatility_grid" in self._strategies:
                return AgentDecision(strategy_name="volatility_grid",
                                    reasoning=f"ADX={adx:.1f}<20 震荡市", hold_bars=180)
        default = "multi_timeframe_confluence"
        if default not in self._strategies:
            default = next(iter(self._strategies))
        return AgentDecision(strategy_name=default, reasoning="状态不明确，使用多TF确认", hold_bars=120)

    # ==================================================================
    # 规则降级 — 复盘
    # ==================================================================

    def _fallback_review(self, equity: float) -> AgentReview:
        pnl_pct = self._get_period_pnl_pct(equity)
        if pnl_pct < -0.015:
            return AgentReview(action="force_stop",
                               reasoning=f"PnL={pnl_pct*100:+.2f}% 亏损超过 1.5%")
        return AgentReview(action="continue",
                           reasoning=f"PnL={pnl_pct*100:+.2f}% 可接受范围")

    # ==================================================================
    # 切换执行
    # ==================================================================

    def _execute_switch(self, decision: AgentDecision, equity: float, snapshot: Dict[str, float]) -> None:
        prev = self._active_strategy_name
        self._active_strategy_name = decision.strategy_name
        self._hold_bars_remaining = decision.hold_bars
        self._last_reasoning = decision.reasoning

        # 重置周期状态
        self._period_start_equity = equity
        self._period_start_bar = self._bar_count
        self._last_review_bar = self._bar_count
        self._last_review_pnl_pct = None
        self._period_signal_counts = {}
        self._period_trade_count = 0
        self._last_signal = "HOLD"
        self._selection_time_snapshot = snapshot

        self._switch_history.append({
            "bar": self._bar_count,
            "from": prev,
            "to": decision.strategy_name,
            "reasoning": decision.reasoning,
            "hold_bars": decision.hold_bars,
            "timestamp": time.time(),
        })
        if len(self._switch_history) > 20:
            self._switch_history = self._switch_history[-20:]

        is_renewal = prev == decision.strategy_name
        action_label = "续期" if is_renewal else "切换"
        logger.info(
            f"| AgentFutures: 策略{action_label} "
            f"{prev or 'None'} → {decision.strategy_name} "
            f"(hold_bars={decision.hold_bars}, equity={equity:,.2f})"
        )

    # ==================================================================
    # agent_meta 构建
    # ==================================================================

    def _build_agent_meta(self) -> Dict[str, Any]:
        period_pnl = 0.0
        if self._period_start_equity > 0 and self._last_equity > 0:
            period_pnl = self._last_equity - self._period_start_equity
        return {
            "active_strategy": self._active_strategy_name,
            "hold_bars_remaining": self._hold_bars_remaining,
            "last_reasoning": self._last_reasoning,
            "pending_switch": False,
            "bar_count": self._bar_count,
            "period_pnl": round(period_pnl, 2),
            "period_pnl_pct": round(self._get_period_pnl_pct(self._last_equity) * 100, 3),
            "period_bars_run": self._bar_count - self._period_start_bar,
            "period_trade_count": self._period_trade_count,
            "last_period_summary": self._last_period_summary,
            "switch_history": self._switch_history[-5:],
        }

    # ==================================================================
    # 计算基础因子
    # ==================================================================

    async def _ensure_base_factors(self, df: pd.DataFrame) -> None:
        from src.factor import factor_manager
        try:
            await factor_manager("ema", df)
            await factor_manager("atr", df)
            await factor_manager("adx", df)
            await factor_manager("rsi", df)
            try:
                await factor_manager("bb", df)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"| AgentFutures: 基础因子计算失败: {e}")

    # ==================================================================
    # 委托给活跃子策略
    # ==================================================================

    async def _delegate_to_active(
        self,
        df: pd.DataFrame,
        current_pos: float,
        equity: float,
        funding_rate: float,
    ) -> Dict[str, Any]:
        active = self._strategies.get(self._active_strategy_name)
        if active is None:
            logger.error(f"| AgentFutures: 活跃策略 '{self._active_strategy_name}' 不存在")
            return {
                "signal": "HOLD", "order": None, "leverage": 1, "sizing": {},
                "atr": 0.0, "meta": {"error": "active_strategy_not_found"},
                "risk": {}, "trailing_stop": {"active": False, "should_update": False,
                                              "sl_price": None, "best_price": None, "entry_price": None},
                "agent_meta": self._build_agent_meta(),
            }

        call_kwargs: Dict[str, Any] = {}
        if self._active_strategy_name == "funding_rate_arbitrage":
            call_kwargs["funding_rate"] = funding_rate

        result = await active(df, current_pos=current_pos, equity=equity, **call_kwargs)

        # 记录信号和交易
        sig = result.get("signal", "HOLD")
        self._record_signal(sig)

        self._hold_bars_remaining = max(0, self._hold_bars_remaining - 1)
        result["agent_meta"] = self._build_agent_meta()
        return result

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
        """三阶段流程:
          Phase 1: 处理 pending_switch（等待平仓完成）
          Phase 2: 活跃策略运行中 → 紧急止损 / 周期到期 / 条件式复盘
          Phase 3: 策略选择（冷启动 / 周期结束后）
        """
        self._bar_count += 1
        funding_rate = kwargs.get("funding_rate", 0.0)
        self._last_equity = equity

        # ==============================================================
        # Phase 1: 活跃策略运行中
        # ==============================================================
        if self._active_strategy_name is not None:

            bars_in_period = self._bar_count - self._period_start_bar
            period_pnl_pct = self._get_period_pnl_pct(equity)

            # ── 1a. 紧急止损 → 直接切换到新策略 ───────────────
            if self._period_start_equity > 0 and period_pnl_pct <= self.emergency_loss_pct:
                logger.warning(
                    f"| AgentFutures: 紧急止损! PnL={period_pnl_pct*100:+.2f}% "
                    f"<= {self.emergency_loss_pct*100:.1f}%"
                )
                self._last_period_summary = self._build_period_summary(
                    "emergency_stopped", equity,
                    f"紧急止损: 周期亏损 {period_pnl_pct*100:+.2f}%",
                )
                self._active_strategy_name = None
                # fall through to Phase 2 (选新策略)

            # 仅当仍有活跃策略时继续
            if self._active_strategy_name is not None:

                # ── 1b. 周期到期 → 续期或切换 ────────────────
                if self._hold_bars_remaining <= 0:
                    self._last_period_summary = self._build_period_summary("period_expired", equity)

                    await self._ensure_base_factors(df)
                    _snap = self._get_market_snapshot(df, funding_rate)
                    _dec = await self._select_strategy_via_llm(df, funding_rate)
                    if _dec is None:
                        _dec = self._fallback_select(df, funding_rate)

                    # 无论同策略还是不同策略，直接切换（新策略继承当前仓位）
                    logger.info(
                        f"| AgentFutures: 周期到期 → 切换到 {_dec.strategy_name} "
                        f"(hold_bars={_dec.hold_bars})"
                    )
                    self._execute_switch(_dec, equity, _snap)
                    return await self._delegate_to_active(df, current_pos, equity, funding_rate)

                # ── 1c. 条件式 LLM 复盘 ──────────────────────
                elif (
                    bars_in_period >= self.min_hold_bars
                    and (self._bar_count - self._last_review_bar) >= self.review_interval
                    and period_pnl_pct < self.review_soft_threshold
                ):
                    await self._ensure_base_factors(df)

                    review = await self._review_strategy_via_llm(df, equity, funding_rate)
                    if review is None:
                        review = self._fallback_review(equity)

                    self._last_review_pnl_pct = period_pnl_pct
                    self._last_review_bar = self._bar_count

                    if review.action == "force_stop":
                        self._last_period_summary = self._build_period_summary(
                            "force_stopped", equity, review.reasoning,
                        )
                        logger.info(f"| AgentFutures: 复盘决定强制停止 | {review.reasoning}")
                        self._active_strategy_name = None
                        # fall through to Phase 2 (选新策略)
                    else:
                        return await self._delegate_to_active(df, current_pos, equity, funding_rate)

                # ── 1d. 正常委托 ─────────────────────────────
                else:
                    return await self._delegate_to_active(df, current_pos, equity, funding_rate)

        # ==============================================================
        # Phase 2: 策略选择（冷启动 / 紧急止损后 / 复盘停止后）
        # ==============================================================
        self._active_strategy_name = None

        await self._ensure_base_factors(df)
        snapshot = self._get_market_snapshot(df, funding_rate)
        decision = await self._select_strategy_via_llm(df, funding_rate)
        if decision is None:
            logger.info("| AgentFutures: LLM 失败，使用规则降级")
            decision = self._fallback_select(df, funding_rate)

        self._execute_switch(decision, equity, snapshot)

        return await self._delegate_to_active(df, current_pos, equity, funding_rate)
