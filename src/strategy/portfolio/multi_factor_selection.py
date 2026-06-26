"""
Multi-Factor Stock Selection Strategy

A comprehensive portfolio strategy that:
1. Calculates multiple factors from a large stock pool
2. Combines factors using weighted scoring
3. Filters and ranks stocks by composite factor score
4. Selects top N stocks and allocates weights

Example:
- Stock pool: 500+ stocks
- Calculate: momentum, volatility, volume, custom factors
- Filter: top 20% by composite score, remove low liquidity
- Allocate: equal weight or factor-weighted to top 20 stocks
"""

from typing import Dict, Optional, List, Callable
from pandas import DataFrame
from pydantic import Field
import pandas as pd
import numpy as np

from src.strategy.types import Strategy


class MultiFactorSelection(Strategy):
    """Multi-factor stock selection portfolio strategy.

    This strategy implements a complete factor-based stock selection workflow:
    1. Calculate multiple factors for all stocks in the pool
    2. Normalize and combine factors with weights
    3. Apply filters (liquidity, data quality, custom rules)
    4. Rank stocks by composite factor score
    5. Select top N stocks
    6. Allocate portfolio weights (equal or factor-weighted)

    Strategy Output Format:
        {
            "AAPL": 0.1,    # 10% weight (top factor score)
            "MSFT": 0.08,   # 8% weight
            ...
            "XYZ": 0.0,     # Not selected (below threshold)
        }
    """

    name: str = Field(default="multi_factor_selection", description="Strategy name")
    description: str = Field(
        default="Multi-factor stock selection - combines multiple factors to select and weight stocks",
        description="Strategy description"
    )
    factor_names: list = Field(default=[], description="List of factor names used")

    # Stock selection parameters
    top_n: int = Field(default=20, description="Number of stocks to select for portfolio")
    top_pct: Optional[float] = Field(default=None, description="Select top X% of stocks (overrides top_n if set)")
    min_stocks: int = Field(default=5, description="Minimum number of stocks to hold")
    max_stocks: int = Field(default=50, description="Maximum number of stocks to hold")

    # Factor calculation parameters
    momentum_lookback: int = Field(default=20, description="Lookback period for momentum factor")
    volatility_lookback: int = Field(default=20, description="Lookback period for volatility factor")
    volume_lookback: int = Field(default=20, description="Lookback period for volume factor")

    # Factor weights for composite score
    momentum_weight: float = Field(default=0.5, description="Weight for momentum factor")
    volatility_weight: float = Field(default=-0.3, description="Weight for volatility factor (negative = prefer low vol)")
    volume_weight: float = Field(default=0.2, description="Weight for volume factor")

    # Filter parameters
    min_data_points: int = Field(default=30, description="Minimum data points required")
    min_price: float = Field(default=1.0, description="Minimum stock price (filter penny stocks)")
    min_volume_pct: float = Field(default=0.1, description="Minimum average volume percentile (0-1)")

    # Weight allocation method
    weight_method: str = Field(default="equal", description="Weight allocation method: 'equal' or 'factor_weighted'")
    min_weight: float = Field(default=0.01, description="Minimum weight per stock (1%)")
    max_weight: float = Field(default=0.2, description="Maximum weight per stock (20%)")

    async def __call__(
        self,
        data: Dict[str, DataFrame],
        current_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Generate target portfolio weights using multi-factor selection.

        Args:
            data: Dict mapping symbol to DataFrame (price + features)
                 Each DataFrame has columns: open, high, low, close, volume, etc.
            current_weights: Current portfolio weights (optional, for reference)

        Returns:
            Dict mapping symbol to target weight (sum <= 1.0)
        """
        if current_weights is None:
            current_weights = {symbol: 0.0 for symbol in data.keys()}

        # Step 1: Calculate individual factors for all stocks
        factor_scores = {}

        for symbol, df in data.items():
            # Pre-filter: data quality check
            if len(df) < self.min_data_points:
                continue

            # Pre-filter: price check
            current_price = df['close'].iloc[-1]
            if current_price < self.min_price:
                continue

            # Calculate factors
            factors = self._calculate_factors(df)
            if factors is not None:
                factor_scores[symbol] = factors

        if not factor_scores:
            # No stocks pass filters, return all zeros
            return {symbol: 0.0 for symbol in data.keys()}

        # Step 2: Normalize factors and calculate composite score
        composite_scores = self._calculate_composite_score(factor_scores)

        # Step 3: Apply volume filter (relative to stock pool)
        composite_scores = self._apply_volume_filter(data, composite_scores)

        # Step 4: Rank and select top stocks
        selected_symbols = self._select_stocks(composite_scores)

        if not selected_symbols:
            # No stocks selected, return all zeros
            return {symbol: 0.0 for symbol in data.keys()}

        # Step 5: Allocate weights to selected stocks
        target_weights = self._allocate_weights(selected_symbols, composite_scores)

        # Step 6: Fill in zeros for non-selected stocks
        final_weights = {}
        for symbol in data.keys():
            final_weights[symbol] = target_weights.get(symbol, 0.0)

        # Step 7: Normalize to ensure sum <= 1.0
        total_weight = sum(final_weights.values())
        if total_weight > 1.0:
            final_weights = {
                symbol: weight / total_weight
                for symbol, weight in final_weights.items()
            }

        return final_weights

    def _calculate_factors(self, df: DataFrame) -> Optional[Dict[str, float]]:
        """Calculate individual factors for a single stock.

        Args:
            df: DataFrame with price and volume data

        Returns:
            Dictionary of factor values, or None if calculation fails
        """
        try:
            factors = {}

            # Momentum factor: price return over lookback period
            if len(df) >= self.momentum_lookback:
                current_price = df['close'].iloc[-1]
                past_price = df['close'].iloc[-self.momentum_lookback]
                factors['momentum'] = (current_price - past_price) / past_price if past_price > 0 else 0.0
            else:
                factors['momentum'] = 0.0

            # Volatility factor: standard deviation of returns
            if len(df) >= self.volatility_lookback:
                returns = df['close'].pct_change().iloc[-self.volatility_lookback:]
                factors['volatility'] = returns.std() if len(returns) > 0 else 0.0
            else:
                factors['volatility'] = 0.0

            # Volume factor: recent volume vs historical average
            if len(df) >= self.volume_lookback and 'volume' in df.columns:
                recent_volume = df['volume'].iloc[-5:].mean()
                historical_volume = df['volume'].iloc[-self.volume_lookback:].mean()
                factors['volume'] = (recent_volume / historical_volume - 1.0) if historical_volume > 0 else 0.0
            else:
                factors['volume'] = 0.0

            return factors

        except Exception as e:
            return None

    def _calculate_composite_score(self, factor_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate composite factor score using weighted combination.

        Args:
            factor_scores: Dict mapping symbol to factor dict

        Returns:
            Dict mapping symbol to composite score
        """
        # Extract factor values into arrays for normalization
        symbols = list(factor_scores.keys())

        momentum_values = np.array([factor_scores[s]['momentum'] for s in symbols])
        volatility_values = np.array([factor_scores[s]['volatility'] for s in symbols])
        volume_values = np.array([factor_scores[s]['volume'] for s in symbols])

        # Normalize factors to [0, 1] using min-max scaling
        def normalize(values):
            min_val, max_val = values.min(), values.max()
            if max_val > min_val:
                return (values - min_val) / (max_val - min_val)
            else:
                return np.zeros_like(values)

        momentum_norm = normalize(momentum_values)
        volatility_norm = normalize(volatility_values)
        volume_norm = normalize(volume_values)

        # Calculate composite score (weighted sum)
        composite_scores = {}
        for i, symbol in enumerate(symbols):
            score = (
                self.momentum_weight * momentum_norm[i] +
                self.volatility_weight * volatility_norm[i] +
                self.volume_weight * volume_norm[i]
            )
            composite_scores[symbol] = score

        return composite_scores

    def _apply_volume_filter(self, data: Dict[str, DataFrame], scores: Dict[str, float]) -> Dict[str, float]:
        """Filter out stocks with low volume relative to the stock pool.

        Args:
            data: Dict mapping symbol to DataFrame
            scores: Dict mapping symbol to composite score

        Returns:
            Filtered scores dict
        """
        # Calculate average volume for each stock
        avg_volumes = {}
        for symbol in scores.keys():
            if symbol in data and 'volume' in data[symbol].columns:
                df = data[symbol]
                if len(df) >= self.volume_lookback:
                    avg_volumes[symbol] = df['volume'].iloc[-self.volume_lookback:].mean()
                else:
                    avg_volumes[symbol] = 0.0
            else:
                avg_volumes[symbol] = 0.0

        if not avg_volumes:
            return scores

        # Calculate volume percentile threshold
        volume_values = list(avg_volumes.values())
        volume_threshold = np.percentile(volume_values, self.min_volume_pct * 100)

        # Filter stocks below volume threshold
        filtered_scores = {
            symbol: score
            for symbol, score in scores.items()
            if avg_volumes.get(symbol, 0) >= volume_threshold
        }

        return filtered_scores

    def _select_stocks(self, scores: Dict[str, float]) -> List[str]:
        """Select top stocks by composite score.

        Args:
            scores: Dict mapping symbol to composite score

        Returns:
            List of selected symbols
        """
        # Rank stocks by score
        ranked_symbols = sorted(
            scores.keys(),
            key=lambda s: scores[s],
            reverse=True  # Higher score is better
        )

        # Determine number of stocks to select
        if self.top_pct is not None:
            # Select by percentile
            n_select = max(self.min_stocks, int(len(ranked_symbols) * self.top_pct))
        else:
            # Select by absolute number
            n_select = self.top_n

        # Apply min/max constraints
        n_select = max(self.min_stocks, min(n_select, self.max_stocks))
        n_select = min(n_select, len(ranked_symbols))

        selected_symbols = ranked_symbols[:n_select]
        return selected_symbols

    def _allocate_weights(self, selected_symbols: List[str], scores: Dict[str, float]) -> Dict[str, float]:
        """Allocate weights to selected stocks.

        Args:
            selected_symbols: List of selected symbols
            scores: Dict mapping symbol to composite score

        Returns:
            Dict mapping symbol to weight
        """
        weights = {}

        if self.weight_method == "equal":
            # Equal weight allocation
            equal_weight = 1.0 / len(selected_symbols)
            for symbol in selected_symbols:
                weights[symbol] = equal_weight

        elif self.weight_method == "factor_weighted":
            # Weight proportional to factor score
            selected_scores = {s: max(0.0, scores[s]) for s in selected_symbols}
            total_score = sum(selected_scores.values())

            if total_score > 0:
                for symbol in selected_symbols:
                    weights[symbol] = selected_scores[symbol] / total_score
            else:
                # Fall back to equal weight
                equal_weight = 1.0 / len(selected_symbols)
                for symbol in selected_symbols:
                    weights[symbol] = equal_weight

        # Apply weight constraints
        for symbol in weights.keys():
            weights[symbol] = max(self.min_weight, min(weights[symbol], self.max_weight))

        # Renormalize after applying constraints
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {symbol: w / total_weight for symbol, w in weights.items()}

        return weights
