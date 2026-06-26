#!/usr/bin/env python3
"""
Factor Validator - Validates and dynamically computes missing factors.

This module scans LLM-generated code for factor references, identifies missing factors,
computes them dynamically, and marks invalid factors.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from src.logger import logger


@dataclass
class FactorSpec:
    """Specification for a factor to be computed."""
    factor_type: str  # e.g., "ema", "ma", "bb"
    period: Optional[int] = None  # e.g., 14, 20, 30
    variant: Optional[str] = None  # e.g., "upper", "lower", "middle" for bb
    no_underscore: bool = False  # True for factors like kmid2, cntp (no underscore before variant)

    @property
    def name(self) -> str:
        """Get the full factor name."""
        if self.variant:
            if self.no_underscore:
                # e.g., kmid2, cntp_10, vsump_20
                if self.period:
                    return f"{self.factor_type}{self.variant}_{self.period}"
                else:
                    return f"{self.factor_type}{self.variant}"
            else:
                # e.g., bb_upper_20, stoch_k_14, macd_signal
                if self.period:
                    return f"{self.factor_type}_{self.variant}_{self.period}"
                else:
                    return f"{self.factor_type}_{self.variant}"
        elif self.period:
            return f"{self.factor_type}_{self.period}"
        else:
            return self.factor_type


@dataclass
class InvalidFactor:
    """Information about an invalid factor."""
    factor_name: str
    reason: str
    affected_samples: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of factor validation."""
    computable_factors: List[FactorSpec] = field(default_factory=list)
    invalid_factors: List[InvalidFactor] = field(default_factory=list)
    computed_factors: List[str] = field(default_factory=list)
    invalid_samples: Dict[str, List[str]] = field(default_factory=dict)  # sample_file -> invalid_factors


class FactorValidator:
    """Validates and dynamically computes missing factors."""

    # Supported factor types (from scripts/generate_features.py)
    SUPPORTED_FACTORS = [
        'atr', 'bb', 'beta', 'cci', 'cnt', 'cord', 'corr', 'ema',
        'imax', 'imin', 'imxd', 'kdj', 'klen', 'klow', 'kmid', 'kup',
        'ksft', 'logvol', 'ma', 'macd', 'max', 'mfi', 'min', 'obv',
        'qtld', 'qtlu', 'rank', 'roc', 'rsi', 'rsv', 'sma', 'std',
        'sum', 'vsum', 'vma', 'vstd', 'wvma',
    ]

    # OHLCV columns (base columns that don't need computation)
    OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'timestamp']

    # Factors without period parameter
    NO_PERIOD_FACTORS = ['macd', 'obv', 'logvol', 'klen', 'klow', 'kmid', 'kup', 'ksft']

    # Factors with variants (e.g., bb_upper_20, bb_lower_20)
    # Format: factor_type -> list of variants
    # Naming: {factor_type}_{variant}_{period} or {factor_type}{variant}_{period}
    VARIANT_FACTORS = {
        'bb': ['upper', 'middle', 'lower'],
        'macd': ['signal', 'hist'],  # macd, macd_signal, macd_hist
        'kdj': ['k', 'd', 'j'],
        'stoch': ['k', 'd'],
        'cnt': ['p', 'n', 'd'],  # cntp, cntn, cntd
        'sum': ['p', 'n', 'd'],  # sump, sumn, sumd
        'vsum': ['p', 'n', 'd'],  # vsump, vsumn, vsumd
        'klow': ['2'],  # klow, klow2
        'kmid': ['2'],  # kmid, kmid2
        'kup': ['2'],  # kup, kup2
        'ksft': ['2'],  # ksft, ksft2
    }

    # Factors where variant is concatenated without underscore (e.g., kmid2 not kmid_2)
    NO_UNDERSCORE_VARIANT_FACTORS = {'klow', 'kmid', 'kup', 'ksft', 'cnt', 'sum', 'vsum'}

    # Factor type aliases (map alternative names to actual factor types)
    FACTOR_ALIASES = {
        'stoch': 'kdj',  # stoch_k_14, stoch_d_14 are computed by kdj factor
    }

    def __init__(self, config: Any, strict_mode: bool = True, use_auto_meta: bool = False):
        """
        Initialize the factor validator.

        Args:
            config: BenchmarkConfig instance
            strict_mode: If True, factors not in meta_info.json will be treated as errors.
                        If False, missing factors will be computed automatically.
            use_auto_meta: If True, use meta_info_auto.json instead of meta_info.json.
                          This is useful when auto-computing factors to avoid modifying
                          the original meta_info.json.
        """
        self.config = config
        self.strict_mode = strict_mode
        self.use_auto_meta = use_auto_meta
        self.data_path = Path(config.data_path)
        self.symbols = config.symbols

        # Determine feature data tag based on data_config
        # Support both data_config (single) and data_configs (list)
        if hasattr(config, 'data_configs'):
            data_configs = config.data_configs
        else:
            # Convert single data_config to list
            data_configs = [config.data_config]

        # Find the price config to get asset_name, source, level
        price_config = None
        for dc in data_configs:
            if dc.get("data_type") == "price":
                price_config = dc
                break
        if price_config is None:
            price_config = data_configs[0]

        self.feature_tag = f"{price_config['asset_name']}_feature_{price_config['level']}"
        self.price_tag = f"{price_config['asset_name']}_price_{price_config['level']}"

        # Store backtest time range for coverage check
        self.start_ts = pd.to_datetime(config.start_ts)
        self.end_ts = pd.to_datetime(config.end_ts)

        # Load existing factor names from meta_info.json or meta_info_auto.json
        meta_filename = "meta_info_auto.json" if self.use_auto_meta else "meta_info.json"
        self.meta_info_path = self.data_path / meta_filename

        # If using auto meta and it doesn't exist, copy from original meta_info.json
        if self.use_auto_meta and not self.meta_info_path.exists():
            original_meta = self.data_path / "meta_info.json"
            if original_meta.exists():
                import shutil
                shutil.copy(original_meta, self.meta_info_path)
                logger.info(f"Created {meta_filename} from meta_info.json")

        self.existing_factors = self._load_existing_factors()
        self.existing_factors_by_symbol = self._load_existing_factors_by_symbol()

        # Check if feature files cover the backtest time range
        self._feature_time_valid = self._check_feature_time_coverage()

    def _load_existing_factors(self) -> Set[str]:
        """Load existing factor names from meta_info.json (global union)."""
        if not self.meta_info_path.exists():
            logger.warning(f"meta_info.json not found at {self.meta_info_path}")
            return set()

        with open(self.meta_info_path, "r", encoding="utf-8") as f:
            meta_info = json.load(f)

        # Get feature names for the current data tag
        feature_info = meta_info.get("data_info", {}).get(self.feature_tag, {})
        names = feature_info.get("names", [])

        # Also include price columns
        price_info = meta_info.get("data_info", {}).get(self.price_tag, {})
        price_names = price_info.get("names", [])

        return set(names) | set(price_names)

    def _load_existing_factors_by_symbol(self) -> Dict[str, Set[str]]:
        """Load existing factor names per symbol from actual data files."""
        result = {}

        # Get price columns (same for all symbols)
        price_cols = set()
        if self.meta_info_path.exists():
            with open(self.meta_info_path, "r", encoding="utf-8") as f:
                meta_info = json.load(f)
            price_info = meta_info.get("data_info", {}).get(self.price_tag, {})
            price_cols = set(price_info.get("names", []))

        for symbol in self.symbols:
            feature_path = self.data_path / self.feature_tag / f"{symbol}.jsonl"
            if feature_path.exists():
                # Read only first row to get column names
                df = pd.read_json(feature_path, lines=True, nrows=1)
                result[symbol] = set(df.columns.tolist()) | price_cols
            else:
                result[symbol] = price_cols

        return result

    def _check_feature_time_coverage(self) -> bool:
        """Check if feature files cover the backtest time range.

        Allows a small tolerance (7 days) for end date to account for
        non-trading days (weekends, holidays).

        Returns:
            True if all feature files cover the backtest time range.
        """
        tolerance = pd.Timedelta(days=7)

        for symbol in self.symbols:
            feature_path = self.data_path / self.feature_tag / f"{symbol}.jsonl"
            if not feature_path.exists():
                logger.warning(f"Feature file not found: {feature_path}")
                return False

            try:
                df = pd.read_json(feature_path, lines=True)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                file_start = df["timestamp"].min()
                file_end = df["timestamp"].max()

                # Check start date (strict) and end date (with tolerance)
                if file_start > self.start_ts:
                    logger.warning(
                        f"Feature file {symbol} starts at {file_start}, "
                        f"after backtest start {self.start_ts}"
                    )
                    return False

                if file_end + tolerance < self.end_ts:
                    logger.warning(
                        f"Feature file {symbol} ends at {file_end}, "
                        f"before backtest end {self.end_ts} (tolerance: {tolerance})"
                    )
                    return False

            except Exception as e:
                logger.warning(f"Error checking feature file {symbol}: {e}")
                return False

        return True

    def extract_factor_references(self, code: str) -> Set[str]:
        """
        Extract all df["xxx"] or df['xxx'] references from code.

        Args:
            code: Python code string

        Returns:
            Set of factor names referenced in the code
        """
        if not code:
            return set()

        # Match df["xxx"], df['xxx'], df.xxx patterns
        patterns = [
            r'df\[[\"\']([a-zA-Z_][a-zA-Z0-9_]*)[\"\']\]',  # df["xxx"] or df['xxx']
            r'df\.([a-zA-Z_][a-zA-Z0-9_]*)',  # df.xxx (but not df.method())
        ]

        references = set()
        for pattern in patterns:
            matches = re.findall(pattern, code)
            references.update(matches)

        # Extract f-string patterns like df[f"ema_{self.fast_period}"]
        # and try to resolve them using Field defaults
        fstring_refs = self._extract_fstring_factors(code)
        references.update(fstring_refs)

        # Filter out common DataFrame methods and attributes
        df_methods = {
            'iloc', 'loc', 'values', 'index', 'columns', 'shape', 'dtypes',
            'head', 'tail', 'describe', 'info', 'copy', 'drop', 'dropna',
            'fillna', 'apply', 'map', 'groupby', 'merge', 'join', 'concat',
            'sort_values', 'sort_index', 'reset_index', 'set_index',
            'rolling', 'shift', 'diff', 'pct_change', 'cumsum', 'cumprod',
            'mean', 'std', 'var', 'min', 'max', 'sum', 'count', 'median',
            'quantile', 'corr', 'cov', 'rank', 'abs', 'round', 'clip',
            'astype', 'to_numpy', 'to_list', 'to_dict', 'to_json', 'to_csv',
            # DataFrame attributes and additional methods
            'empty', 'get', 'attrs', 'size', 'ndim', 'T', 'axes', 'keys',
            'items', 'iterrows', 'itertuples', 'pop', 'insert', 'where',
            'mask', 'query', 'eval', 'select_dtypes', 'isin', 'between',
            'duplicated', 'nunique', 'idxmax', 'idxmin', 'mode', 'sample',
            'nlargest', 'nsmallest', 'first', 'last', 'nth', 'any', 'all',
        }
        references -= df_methods

        return references

    def _extract_fstring_factors(self, code: str) -> Set[str]:
        """
        Extract factor names from f-string patterns like df[f"ema_{self.period}"].
        Resolves self.xxx using Field defaults from the class definition.
        """
        factors = set()

        # Extract Field defaults: xxx: type = Field(default=value)
        field_defaults = {}
        field_pattern = r'(\w+):\s*\w+\s*=\s*Field\s*\(\s*default\s*=\s*(\d+)'
        for match in re.finditer(field_pattern, code):
            field_defaults[match.group(1)] = match.group(2)

        # Match f-string patterns: df[f"prefix_{self.xxx}"] or df[f"prefix_{self.xxx}_suffix"]
        fstring_pattern = r'df\[f["\']([a-zA-Z_]+)_\{self\.(\w+)\}(?:_([a-zA-Z_]+))?\s*["\']\]'
        for match in re.finditer(fstring_pattern, code):
            prefix = match.group(1)
            var_name = match.group(2)
            suffix = match.group(3) or ""

            if var_name in field_defaults:
                value = field_defaults[var_name]
                if suffix:
                    factor_name = f"{prefix}_{value}_{suffix}"
                else:
                    factor_name = f"{prefix}_{value}"
                factors.add(factor_name)

        return factors

    def parse_factor_name(self, name: str) -> Tuple[Optional[FactorSpec], Optional[str]]:
        """
        Parse a factor name into a FactorSpec.

        Args:
            name: Factor name (e.g., "ema_14", "bb_upper_20", "macd")

        Returns:
            Tuple of (FactorSpec or None, error_reason or None)
        """
        # Skip OHLCV columns
        if name in self.OHLCV_COLUMNS:
            return None, None

        # Try to match different patterns
        # Pattern 1: {type}_{variant}_{period} (e.g., bb_upper_20)
        match = re.match(r'^([a-z]+)_([a-z]+)_(\d+)$', name)
        if match:
            factor_type, variant, period_str = match.groups()
            # Check if it's a valid variant factor
            if factor_type in self.VARIANT_FACTORS:
                if variant in self.VARIANT_FACTORS[factor_type]:
                    no_underscore = factor_type in self.NO_UNDERSCORE_VARIANT_FACTORS
                    return FactorSpec(factor_type, int(period_str), variant, no_underscore), None
            # Could be a factor like "stoch_k_14"
            base_type = f"{factor_type}_{variant}"
            if base_type in ['stoch_k', 'stoch_d']:
                return FactorSpec('kdj', int(period_str), variant, no_underscore=False), None

        # Pattern 2: {type}_{period} (e.g., ema_14, ma_30) or {type}{variant}_{period} (e.g., cntp_10)
        match = re.match(r'^([a-z]+)_(\d+)$', name)
        if match:
            factor_type, period_str = match.groups()
            # First check for no-period variant factors with wrong underscore format
            # e.g., klow_2 -> klow2, kmid_2 -> kmid2 (LLM error: used underscore)
            if factor_type in self.NO_PERIOD_FACTORS and factor_type in self.VARIANT_FACTORS:
                if period_str in self.VARIANT_FACTORS[factor_type]:
                    # This is actually a variant, not a period (e.g., klow_2 means klow2)
                    return FactorSpec(factor_type, None, period_str, no_underscore=True), None
            # Then check normal factors with period
            if factor_type in self.SUPPORTED_FACTORS:
                return FactorSpec(factor_type, int(period_str)), None
            # Check for variant factors without explicit variant
            # e.g., cntp_10, cntd_10, sump_10
            for base_type, variants in self.VARIANT_FACTORS.items():
                for v in variants:
                    if factor_type == f"{base_type}{v}":
                        no_underscore = base_type in self.NO_UNDERSCORE_VARIANT_FACTORS
                        return FactorSpec(base_type, int(period_str), v, no_underscore), None
            return None, f"Unknown factor type: '{factor_type}'"

        # Pattern 3: {type} (e.g., macd, obv, logvol)
        if name in self.NO_PERIOD_FACTORS:
            return FactorSpec(name), None

        # Pattern 4: {type}_{variant} (e.g., macd_signal, macd_hist)
        match = re.match(r'^([a-z]+)_([a-z0-9]+)$', name)
        if match:
            factor_type, variant = match.groups()
            if factor_type in self.VARIANT_FACTORS:
                if variant in self.VARIANT_FACTORS[factor_type]:
                    no_underscore = factor_type in self.NO_UNDERSCORE_VARIANT_FACTORS
                    return FactorSpec(factor_type, None, variant, no_underscore), None

        # Pattern 5: {type}{variant} without underscore (e.g., kmid2, klow2, kup2, ksft2)
        for base_type, variants in self.VARIANT_FACTORS.items():
            if base_type in self.NO_UNDERSCORE_VARIANT_FACTORS:
                for v in variants:
                    if name == f"{base_type}{v}":
                        return FactorSpec(base_type, None, v, no_underscore=True), None

        # Check if it's a known factor without period
        if name in self.SUPPORTED_FACTORS:
            return FactorSpec(name), None

        return None, f"Cannot parse factor name: '{name}'"

    def get_missing_factors(
        self,
        references: Set[str],
        sample_id: str,
    ) -> Tuple[List[FactorSpec], List[InvalidFactor]]:
        """
        Identify missing factors from references.

        Checks each symbol to see if the factor exists. A factor is considered
        missing if ANY symbol doesn't have it.

        Args:
            references: Set of factor names referenced in code
            sample_id: Sample identifier for error tracking

        Returns:
            Tuple of (computable_factors, invalid_factors)
        """
        computable = []
        invalid = []

        for name in references:
            # Skip OHLCV columns
            if name in self.OHLCV_COLUMNS:
                continue

            # Check if factor exists in ALL symbols
            # A factor is missing if any symbol doesn't have it
            missing_in_any_symbol = False
            if self._feature_time_valid:
                for symbol in self.symbols:
                    symbol_factors = self.existing_factors_by_symbol.get(symbol, set())
                    if name not in symbol_factors:
                        missing_in_any_symbol = True
                        break
            else:
                # If time range is invalid, we need to recompute all factors
                missing_in_any_symbol = True

            if not missing_in_any_symbol:
                continue

            # Try to parse the factor name
            spec, error = self.parse_factor_name(name)

            if spec:
                computable.append(spec)
            elif error:
                # Check if we already have this invalid factor
                existing = next((f for f in invalid if f.factor_name == name), None)
                if existing:
                    if sample_id not in existing.affected_samples:
                        existing.affected_samples.append(sample_id)
                else:
                    invalid.append(InvalidFactor(
                        factor_name=name,
                        reason=error,
                        affected_samples=[sample_id],
                    ))

        return computable, invalid

    async def compute_missing_factors(
        self,
        missing_specs: List[FactorSpec],
    ) -> List[str]:
        """
        Compute missing factors and append to data files.

        Args:
            missing_specs: List of FactorSpec to compute

        Returns:
            List of computed factor names (union of all symbols)
        """
        if not missing_specs:
            return []

        # Import factor_manager here to avoid circular imports
        from src.factor import factor_manager

        # Group specs by factor type for efficient computation
        specs_by_type: Dict[str, List[FactorSpec]] = {}
        for spec in missing_specs:
            if spec.factor_type not in specs_by_type:
                specs_by_type[spec.factor_type] = []
            specs_by_type[spec.factor_type].append(spec)

        # Initialize factor manager with required factors (apply aliases)
        factor_types_for_init = set()
        for ft in specs_by_type.keys():
            actual_ft = self.FACTOR_ALIASES.get(ft, ft)
            factor_types_for_init.add(actual_ft)
        factor_types_for_init = list(factor_types_for_init)
        logger.info(f"Initializing factor manager for: {factor_types_for_init}")
        await factor_manager.initialize(factor_names=factor_types_for_init)

        # Track computed factors per symbol
        computed_by_symbol: Dict[str, List[str]] = {symbol: [] for symbol in self.symbols}

        # Process each symbol
        for symbol in self.symbols:
            price_path = self.data_path / self.price_tag / f"{symbol}.jsonl"
            feature_path = self.data_path / self.feature_tag / f"{symbol}.jsonl"

            if not price_path.exists():
                logger.warning(f"Price file not found: {price_path}")
                continue

            # Read price data
            price_df = pd.read_json(price_path, lines=True)
            price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])
            price_df = price_df.sort_values(by="timestamp").reset_index(drop=True)

            # Read existing feature data
            if feature_path.exists():
                feature_df = pd.read_json(feature_path, lines=True)
            else:
                feature_df = pd.DataFrame()
                feature_df["timestamp"] = price_df["timestamp"].apply(
                    lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
                )

            # Get existing columns for this symbol
            existing_cols = set(feature_df.columns.tolist())
            price_row_count = len(price_df)
            feature_row_count = len(feature_df)

            # Check if feature data needs to be extended (new price data added)
            need_recompute_all = feature_row_count < price_row_count

            # Compute each factor type (only if symbol needs it)
            for factor_type, specs in specs_by_type.items():
                # Get unique periods for this factor type
                periods = list(set(s.period for s in specs if s.period))

                # Determine expected output column names for this factor
                actual_factor_type = self.FACTOR_ALIASES.get(factor_type, factor_type)
                if periods:
                    if factor_type in self.VARIANT_FACTORS:
                        expected_cols = []
                        for p in periods:
                            for v in self.VARIANT_FACTORS[factor_type]:
                                if factor_type in self.NO_UNDERSCORE_VARIANT_FACTORS:
                                    expected_cols.append(f"{factor_type}{v}_{p}")
                                else:
                                    expected_cols.append(f"{factor_type}_{v}_{p}")
                    else:
                        expected_cols = [f"{factor_type}_{p}" for p in periods]
                else:
                    # No-period factors - check what columns are missing
                    expected_cols = [s.name for s in specs]

                # Check which columns need to be computed
                # Case 1: Column doesn't exist -> need to compute
                # Case 2: Feature data has fewer rows than price data -> need to recompute
                missing_cols = [c for c in expected_cols if c not in existing_cols]
                cols_to_update = missing_cols.copy()

                if need_recompute_all:
                    # Add existing columns that need to be recomputed due to new data
                    for c in expected_cols:
                        if c in existing_cols and c not in cols_to_update:
                            cols_to_update.append(c)

                if not cols_to_update:
                    # All columns exist and data is complete, skip computation
                    continue

                # Get factor instance
                factor = await factor_manager.get(actual_factor_type)
                if factor is None:
                    logger.warning(f"Factor {actual_factor_type} not found")
                    continue

                if periods:
                    # Factors with periods (e.g., ema_14, bb_upper_20)
                    # Set periods/windows based on what attribute the factor uses
                    if hasattr(factor, 'windows'):
                        factor.windows = periods
                        # Regenerate names for factors with variants
                        if factor_type in self.VARIANT_FACTORS:
                            names = []
                            for p in periods:
                                for v in self.VARIANT_FACTORS[factor_type]:
                                    # Check if underscore is needed
                                    if factor_type in self.NO_UNDERSCORE_VARIANT_FACTORS:
                                        names.append(f"{factor_type}{v}_{p}")
                                    else:
                                        names.append(f"{factor_type}_{v}_{p}")
                            factor.names = names
                        else:
                            factor.names = [f"{factor_type}_{p}" for p in periods]
                    elif hasattr(factor, 'periods'):
                        factor.periods = periods
                        if factor_type in self.VARIANT_FACTORS:
                            names = []
                            for p in periods:
                                for v in self.VARIANT_FACTORS[factor_type]:
                                    if factor_type in self.NO_UNDERSCORE_VARIANT_FACTORS:
                                        names.append(f"{factor_type}{v}_{p}")
                                    else:
                                        names.append(f"{factor_type}_{v}_{p}")
                            factor.names = names
                        else:
                            factor.names = [f"{factor_type}_{p}" for p in periods]
                    else:
                        factor.names = [f"{factor_type}_{p}" for p in periods]
                else:
                    # No-period factors (e.g., kmid, kmid2, macd, macd_signal)
                    # These factors have fixed output names, don't modify factor.names
                    # Just use the factor's default names
                    pass

                # Execute factor computation (for both with-period and no-period factors)
                try:
                    new_cols_df = await factor(price_df.copy())
                except Exception as e:
                    logger.warning(f"Failed to compute factor {factor_type}: {e}")
                    continue

                # Add/update columns in feature_df
                # Always overwrite to handle partial data (e.g., new data appended)
                for col in new_cols_df.columns:
                    if col in cols_to_update:
                        feature_df[col] = new_cols_df[col].values
                        computed_by_symbol[symbol].append(col)

            # Save updated feature data
            feature_df.to_json(feature_path, orient="records", lines=True)
            logger.info(f"Updated {feature_path} with {len(computed_by_symbol[symbol])} new factors")

        # Update meta_info.json with per-symbol tracking
        self._update_meta_info(computed_by_symbol)

        # Update local cache for existing_factors_by_symbol
        for symbol, new_names in computed_by_symbol.items():
            if symbol in self.existing_factors_by_symbol:
                self.existing_factors_by_symbol[symbol].update(new_names)

        # Return union of all computed factors
        all_computed = set()
        for names in computed_by_symbol.values():
            all_computed.update(names)
        return sorted(all_computed)

    def _update_meta_info(self, computed_by_symbol: Dict[str, List[str]]) -> None:
        """
        Update meta_info.json with new factor names, tracking per-symbol.

        Args:
            computed_by_symbol: Dict mapping symbol to list of computed factor names
        """
        # Get union of all new factor names
        all_new_names = set()
        for names in computed_by_symbol.values():
            all_new_names.update(names)

        if not all_new_names:
            return

        if not self.meta_info_path.exists():
            logger.warning(f"meta_info.json not found: {self.meta_info_path}")
            return

        with open(self.meta_info_path, "r", encoding="utf-8") as f:
            meta_info = json.load(f)

        # Update feature names
        if self.feature_tag in meta_info.get("data_info", {}):
            feature_info = meta_info["data_info"][self.feature_tag]

            # Update global names list (union of all symbols, for backward compatibility)
            existing_names = set(feature_info.get("names", []))
            updated_names = sorted(existing_names | all_new_names)
            feature_info["names"] = updated_names

            # Update per-symbol names tracking
            if "names_by_symbol" not in feature_info:
                feature_info["names_by_symbol"] = {}

            for symbol, new_names in computed_by_symbol.items():
                if new_names:  # Only update if there are new factors
                    existing_symbol_names = set(feature_info["names_by_symbol"].get(symbol, []))
                    # If symbol not tracked yet, read from actual file
                    if not existing_symbol_names:
                        feature_path = self.data_path / self.feature_tag / f"{symbol}.jsonl"
                        if feature_path.exists():
                            import pandas as pd
                            df = pd.read_json(feature_path, lines=True, nrows=1)
                            existing_symbol_names = set(df.columns.tolist())
                    updated_symbol_names = sorted(existing_symbol_names | set(new_names))
                    feature_info["names_by_symbol"][symbol] = updated_symbol_names

        # Save updated meta_info
        with open(self.meta_info_path, "w", encoding="utf-8") as f:
            json.dump(meta_info, f, indent=4, ensure_ascii=False)

        # Update local cache
        self.existing_factors.update(all_new_names)
        logger.info(f"Updated meta_info.json with {len(all_new_names)} new factors")

    async def validate_and_compute(
        self,
        extracted_codes: List[Dict[str, Any]],
    ) -> ValidationResult:
        """
        Validate all extracted codes and compute missing factors.

        Args:
            extracted_codes: List of extracted code dictionaries

        Returns:
            ValidationResult with computed and invalid factors
        """
        result = ValidationResult()
        all_computable: Dict[str, FactorSpec] = {}  # name -> spec (deduplicated)
        all_invalid: Dict[str, InvalidFactor] = {}  # name -> InvalidFactor

        # Step 1: Scan all codes for factor references
        logger.info("Scanning codes for factor references...")
        for code_info in extracted_codes:
            strategy_code = code_info.get("strategy_code")
            if not strategy_code:
                continue

            sample_id = f"{code_info.get('model', 'unknown')}_{code_info.get('query_id', 'unknown')}_{code_info.get('sample_id', 0)}"

            # Extract references from strategy code
            references = self.extract_factor_references(strategy_code)

            # Also extract from factor codes if present
            for factor_code in code_info.get("factor_codes", []):
                references.update(self.extract_factor_references(factor_code))

            # Get missing factors
            computable, invalid = self.get_missing_factors(references, sample_id)

            # Deduplicate computable factors
            for spec in computable:
                if spec.name not in all_computable:
                    all_computable[spec.name] = spec

            # Merge invalid factors
            for inv in invalid:
                if inv.factor_name in all_invalid:
                    # Merge affected samples
                    existing = all_invalid[inv.factor_name]
                    for s in inv.affected_samples:
                        if s not in existing.affected_samples:
                            existing.affected_samples.append(s)
                else:
                    all_invalid[inv.factor_name] = inv

        # Convert to lists
        result.computable_factors = list(all_computable.values())
        result.invalid_factors = list(all_invalid.values())

        logger.info(f"Found {len(result.computable_factors)} computable missing factors")
        logger.info(f"Found {len(result.invalid_factors)} invalid factors")

        # Step 2: Handle missing factors based on strict_mode
        if result.computable_factors:
            if self.strict_mode:
                # In strict mode, treat computable factors as errors
                missing_names = [spec.name for spec in result.computable_factors]
                logger.warning(
                    f"Strict mode: {len(result.computable_factors)} missing factors "
                    "will be treated as errors (use --allow-compute-factors to compute them)"
                )
                logger.warning(f"Missing factors: {missing_names}")
                # Convert computable factors to invalid factors
                for spec in result.computable_factors:
                    # Find affected samples for this factor
                    affected = []
                    for code_info in extracted_codes:
                        strategy_code = code_info.get("strategy_code")
                        if not strategy_code:
                            continue
                        refs = self.extract_factor_references(strategy_code)
                        if spec.name in refs:
                            sample_id = f"{code_info.get('model', 'unknown')}_{code_info.get('query_id', 'unknown')}_{code_info.get('sample_id', 0)}"
                            affected.append(sample_id)

                    inv = InvalidFactor(
                        factor_name=spec.name,
                        reason=f"Factor not in meta_info.json (strict mode)",
                        affected_samples=affected,
                    )
                    result.invalid_factors.append(inv)
                    all_invalid[spec.name] = inv

                # Clear computable factors since we're not computing them
                result.computable_factors = []
            else:
                # Compute missing factors
                logger.info("Computing missing factors...")
                result.computed_factors = await self.compute_missing_factors(
                    result.computable_factors
                )
                logger.info(f"Computed {len(result.computed_factors)} new factors")

        # Step 3: Build invalid samples mapping
        for inv in result.invalid_factors:
            for sample_id in inv.affected_samples:
                if sample_id not in result.invalid_samples:
                    result.invalid_samples[sample_id] = []
                if inv.factor_name not in result.invalid_samples[sample_id]:
                    result.invalid_samples[sample_id].append(inv.factor_name)

        return result

    def save_invalid_report(
        self,
        result: ValidationResult,
        output_path: Path,
    ) -> None:
        """
        Save invalid factors report to JSON file.

        Args:
            result: ValidationResult from validate_and_compute
            output_path: Path to save the report
        """
        report = {
            "summary": {
                "total_invalid_factors": len(result.invalid_factors),
                "total_affected_samples": len(result.invalid_samples),
                "total_computed_factors": len(result.computed_factors),
            },
            "computed_factors": result.computed_factors,
            "invalid_factors": [
                {
                    "factor_name": inv.factor_name,
                    "reason": inv.reason,
                    "affected_samples": inv.affected_samples,
                }
                for inv in result.invalid_factors
            ],
            "affected_samples": [
                {
                    "sample_id": sample_id,
                    "invalid_factors": factors,
                }
                for sample_id, factors in result.invalid_samples.items()
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved invalid factors report to: {output_path}")
