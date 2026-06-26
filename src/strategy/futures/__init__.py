"""
Futures trading strategies for Binance USDT-M perpetual contracts.

These strategies integrate directly with the Binance Futures API to
execute real trades, unlike the backtesting strategies in single_trading/.

策略矩阵:
  ┌─────────────────────────────────┬──────────┬──────────┬──────────────────────┐
  │ 策略                            │ 风格     │ 杠杆     │ 适用市场状态         │
  ├─────────────────────────────────┼──────────┼──────────┼──────────────────────┤
  │ AdaptiveTrendFusion (保守趋势)  │ 保守     │ 5-10x    │ 强趋势 (ADX > 20)    │
  │ MeanReversion (均值回归)        │ 平衡     │ 3-5x     │ 震荡市 (ADX < 22)    │
  │ FundingRateArbitrage (资金费率) │ 超保守   │ 2-3x     │ 资金费率极端         │
  │ MomentumBreakout (动量突破)     │ 适度激进 │ 5-8x     │ 波动率压缩后突破     │
  │ VolatilityGrid (波动率网格)     │ 平衡     │ 2-3x     │ 震荡/弱趋势 (ADX<25) │
  │ MultiTimeframeConfluence (多TF) │ 平衡     │ 5-7x     │ 多TF方向一致         │
  │ LiquidationCascade (清算猎杀)   │ 适度激进 │ 5-8x     │ 成交量爆发+价格加速  │
  │ AgentFutures (LLM Agent)        │ 自适应   │ 动态     │ LLM 智能调度上述策略 │
  └─────────────────────────────────┴──────────┴──────────┴──────────────────────┘

组合建议:
  - 强趋势: AdaptiveTrendFusion + MomentumBreakout
  - 震荡市: MeanReversion + VolatilityGrid
  - 全天候: FundingRateArbitrage (持续) + MultiTimeframeConfluence (精选)
  - 事件驱动: LiquidationCascade (清算瀑布)
"""

from src.strategy.futures.adaptive_trend_fusion import AdaptiveTrendFusionStrategy
from src.strategy.futures.mean_reversion import MeanReversionStrategy
from src.strategy.futures.funding_rate_arbitrage import FundingRateArbitrageStrategy
from src.strategy.futures.momentum_breakout import MomentumBreakoutStrategy
from src.strategy.futures.volatility_grid import VolatilityGridStrategy
from src.strategy.futures.multi_timeframe_confluence import MultiTimeframeConfluenceStrategy
from src.strategy.futures.liquidation_cascade import LiquidationCascadeStrategy
from src.strategy.futures.agent_futures import AgentFuturesStrategy

__all__ = [
    "AdaptiveTrendFusionStrategy",
    "MeanReversionStrategy",
    "FundingRateArbitrageStrategy",
    "MomentumBreakoutStrategy",
    "VolatilityGridStrategy",
    "MultiTimeframeConfluenceStrategy",
    "LiquidationCascadeStrategy",
    "AgentFuturesStrategy",
]
