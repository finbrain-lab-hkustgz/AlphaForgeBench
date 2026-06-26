"""
共享工具模块 — 供所有期货策略使用。

包含: 仓位管理、手续费计算、通用辅助函数。
因子计算请使用 src/factor/factors/ 中的因子类。
"""

import json as _json
from typing import Dict, Any, Optional, Literal, List
import numpy as np

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
PRICE_COL = "close"
VOL_COL = "volume"
DEFAULT_TAKER_FEE = 0.0004  # 每侧 0.04%（Binance BNB 折扣等级）


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def safe_float(val, default: float = 0.0) -> float:
    """安全转换为 float，NaN / None 返回默认值。"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# ROI / price conversion helpers
# ---------------------------------------------------------------------------

def price_move_to_roi_pct(entry_price: float, price_move: float, leverage: int) -> float:
    """Convert absolute price move to ROI% (on margin) given leverage.

    ROI% = (price_move / entry_price) * leverage * 100
    """
    if entry_price <= 0 or leverage <= 0:
        return 0.0
    return max(0.0, float(price_move) / float(entry_price) * float(leverage) * 100.0)


def price_target_to_roi_pct(entry_price: float, target_price: float, signal: int, leverage: int) -> float:
    """Convert a TP/SL target price to ROI% given direction.

    signal: +1 (long) or -1 (short)
    """
    if entry_price <= 0 or target_price <= 0 or signal not in (1, -1) or leverage <= 0:
        return 0.0
    if signal == 1:
        move = target_price - entry_price
    else:
        move = entry_price - target_price
    return price_move_to_roi_pct(entry_price, max(move, 0.0), leverage)


# ---------------------------------------------------------------------------
# 手续费模型
# ---------------------------------------------------------------------------

def compute_fee_info(
    price: float,
    quantity: float,
    sl_distance: float,
    tp_distance: float,
    fee_rate: float = DEFAULT_TAKER_FEE,
) -> Dict[str, Any]:
    """计算往返手续费和扣费后的有效风险回报比。"""
    notional = quantity * price
    fee_per_side = notional * fee_rate
    round_trip_fee = fee_per_side * 2

    sl_loss = quantity * sl_distance if sl_distance > 0 else 0.0
    tp_profit = quantity * tp_distance if tp_distance > 0 else 0.0

    effective_sl_loss = sl_loss + round_trip_fee
    effective_tp_profit = tp_profit - round_trip_fee

    if effective_sl_loss > 0 and effective_tp_profit > 0:
        effective_rr = round(effective_tp_profit / effective_sl_loss, 3)
    elif effective_sl_loss > 0:
        effective_rr = 0.0
    else:
        effective_rr = float("inf")

    breakeven_wr = round(1.0 / (1.0 + effective_rr), 3) if effective_rr > 0 else 1.0

    return {
        "fee_per_side": round(fee_per_side, 4),
        "round_trip_fee": round(round_trip_fee, 4),
        "sl_loss_raw": round(sl_loss, 4),
        "tp_profit_raw": round(tp_profit, 4),
        "effective_sl_loss": round(effective_sl_loss, 4),
        "effective_tp_profit": round(effective_tp_profit, 4),
        "effective_rr": effective_rr,
        "breakeven_win_rate": breakeven_wr,
        "fee_pct_of_sl": round(round_trip_fee / sl_loss * 100, 1) if sl_loss > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# 仓位管理
# ---------------------------------------------------------------------------

def calc_position_size(
    equity: float,
    price: float,
    atr: float,
    risk_pct: float = 0.02,
    atr_sl_mult: float = 2.0,
    min_leverage: int = 3,
    max_leverage: int = 10,
    min_notional_pct: float = 0.50,
    max_notional_pct: float = 1.0,
    min_quantity: float = 0.02,
    quantity_precision: int = 3,
) -> Dict[str, float]:
    """通用仓位计算: 固定风险百分比 + 名义价值 / 杠杆硬性限制。

    1. 根据 risk_pct 和 ATR 止损距离计算理论仓位
    2. 用 min/max_notional_pct 和 min/max_leverage 硬性约束

    返回 dict: quantity, leverage, notional, notional_pct,
              risk_usdt, sl_distance, actual_risk_usdt, actual_risk_pct
    """
    sl_distance = atr * atr_sl_mult

    if atr <= 0 or price <= 0 or equity <= 0:
        return {
            "quantity": min_quantity, "leverage": min_leverage,
            "notional": min_quantity * price, "notional_pct": 0.0,
            "risk_usdt": 0.0, "sl_distance": 0.0,
            "actual_risk_usdt": 0.0, "actual_risk_pct": 0.0,
        }

    risk_usdt = equity * risk_pct

    # 理论仓位
    quantity = risk_usdt / sl_distance if sl_distance > 0 else min_quantity
    quantity = max(quantity, min_quantity)
    quantity = round(quantity, quantity_precision)

    notional = quantity * price

    # 名义价值硬性限制
    min_notional = equity * min_notional_pct
    max_notional = equity * max_notional_pct
    notional = max(min_notional, min(max_notional, notional))
    quantity = max(notional / price, min_quantity) if price > 0 else min_quantity
    quantity = round(quantity, quantity_precision)
    notional = quantity * price

    # 杠杆
    leverage = max(1, int(np.ceil(notional / equity)))
    leverage = max(min_leverage, min(max_leverage, leverage))

    actual_risk_usdt = quantity * sl_distance
    actual_risk_pct = actual_risk_usdt / equity if equity > 0 else 0.0

    return {
        "quantity": quantity,
        "leverage": leverage,
        "notional": round(notional, 4),
        "notional_pct": round(notional / equity, 4) if equity > 0 else 0.0,
        "risk_usdt": round(risk_usdt, 4),
        "sl_distance": round(sl_distance, 2),
        "actual_risk_usdt": round(actual_risk_usdt, 4),
        "actual_risk_pct": round(actual_risk_pct, 6),
    }


# ---------------------------------------------------------------------------
# 订单构建辅助
# ---------------------------------------------------------------------------

def build_order(
    symbol: str,
    signal: int,
    current_pos: float,
    sizing: Dict[str, float],
    price: float,
    atr: float,
    atr_sl_multiplier: float,
    atr_tp_multiplier: float = 0.0,
    use_trailing_stop: bool = True,
    flip_mode: Literal["flip", "close_only", "close_then_wait"] = "flip",
    tp_sl_mode: Literal["PRICE", "ROI"] = "PRICE",
    take_profit_price: Optional[float] = None,
    stop_loss_price: Optional[float] = None,
    take_profit_roi_pct: Optional[float] = None,
    stop_loss_roi_pct: Optional[float] = None,
    fee_rate: float = DEFAULT_TAKER_FEE,
    quantity_precision: int = 3,
) -> Dict[str, Any]:
    """根据信号构建 Binance 合约订单（通用版本）。

    返回 {"signal": str, "order": dict|None, "meta": dict}
    """
    cur_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
    cur_amt = abs(current_pos)

    # HOLD: signal=0 → 不操作，仓位不变
    if signal == 0:
        return {"signal": "HOLD", "order": None, "meta": {}}

    meta: Dict[str, Any] = {}

    meta: Dict[str, Any] = {"flip_mode": flip_mode}

    # CLOSE-ONLY: reverse signal but do not flip
    if cur_side != 0 and cur_side != signal and flip_mode in ("close_only", "close_then_wait"):
        close_side = "SELL" if cur_side == 1 else "BUY"
        close_qty = round(cur_amt, quantity_precision)
        order: Dict[str, Any] = {
            "symbol": symbol,
            "side": close_side,
            "type": "MARKET",
            "quantity": close_qty,
            "position_side": "BOTH",
            "reduce_only": True,
        }
        meta.update({
            "mode": "close_only",
            "reduce_only": True,
            "close_qty": close_qty,
        })
        return {"signal": "CLOSE", "order": order, "meta": meta}

    # BUY / SELL (open or flip)
    order_side = "BUY" if signal == 1 else "SELL"
    open_qty = float(sizing["quantity"])

    # 反向持仓: 需要先平旧仓再开新仓 → 实际下单 = 旧仓量 + 新仓量
    # 同向 / 空仓: 直接下新仓量
    if cur_side != 0 and cur_side != signal:
        total_qty = cur_amt + open_qty
    else:
        total_qty = open_qty
    total_qty = round(total_qty, quantity_precision)

    order: Dict[str, Any] = {
        "symbol": symbol,
        "side": order_side,
        "type": "MARKET",
        "quantity": total_qty,
        "position_side": "BOTH",
    }

    sl_distance = 0.0
    tp_distance = 0.0

    if atr > 0 and price > 0:
        sl_distance = float(atr) * float(atr_sl_multiplier)
        sl_price_calc = round(price - sl_distance, 2) if signal == 1 else round(price + sl_distance, 2)
        sl_price_final = float(stop_loss_price) if (stop_loss_price is not None and stop_loss_price > 0) else sl_price_calc

        # TP: fixed only when not trailing; can be overridden by take_profit_price
        tp_price_final: Optional[float] = None
        if not use_trailing_stop:
            if take_profit_price is not None and take_profit_price > 0:
                tp_price_final = float(take_profit_price)
            elif atr_tp_multiplier > 0:
                tp_distance = float(atr) * float(atr_tp_multiplier)
                tp_price_final = round(price + tp_distance, 2) if signal == 1 else round(price - tp_distance, 2)

        # Mode: PRICE vs ROI
        if tp_sl_mode == "ROI":
            lev = int(sizing.get("leverage", 1) or 1)
            order["tp_sl_mode"] = "ROI"
            # Prefer explicit ROI inputs when provided (e.g., fixed ROI TP/SL strategies)
            sl_roi = float(stop_loss_roi_pct) if (stop_loss_roi_pct is not None and stop_loss_roi_pct > 0) else None
            tp_roi = float(take_profit_roi_pct) if (take_profit_roi_pct is not None and take_profit_roi_pct > 0) else None

            if sl_roi is None:
                sl_roi = price_move_to_roi_pct(price, abs(price - sl_price_final), lev)
            order["stop_loss"] = float(sl_roi)
            meta["sl_roi_pct"] = round(float(sl_roi), 4)
            sl_move = price * float(sl_roi) / (max(1, lev) * 100.0)
            sl_price_equiv = (price - sl_move) if signal == 1 else (price + sl_move)
            meta["sl_price_equiv"] = round(sl_price_equiv, 2)
            sl_distance = abs(sl_price_equiv - price)

            if tp_roi is not None:
                order["take_profit"] = float(tp_roi)
                meta["tp_roi_pct"] = round(float(tp_roi), 4)
                tp_move = price * float(tp_roi) / (max(1, lev) * 100.0)
                tp_price_equiv = (price + tp_move) if signal == 1 else (price - tp_move)
                meta["tp_price_equiv"] = round(tp_price_equiv, 2)
                tp_distance = abs(tp_price_equiv - price)
                if sl_distance > 0 and tp_distance > 0:
                    meta["raw_rr"] = round(tp_distance / sl_distance, 2)
            elif tp_price_final is not None:
                tp_roi2 = price_target_to_roi_pct(price, tp_price_final, signal, lev)
                if tp_roi2 > 0:
                    order["take_profit"] = tp_roi2
                    meta["tp_roi_pct"] = round(tp_roi2, 4)
                    meta["tp_price_equiv"] = round(tp_price_final, 2)
                    tp_distance = abs(tp_price_final - price)
                    if sl_distance > 0 and tp_distance > 0:
                        meta["raw_rr"] = round(tp_distance / sl_distance, 2)
        else:
            order["tp_sl_mode"] = "PRICE"
            order["stop_loss"] = round(sl_price_final, 2)
            meta["sl_price"] = round(sl_price_final, 2)
            meta["sl_distance"] = round(sl_distance, 2)

            if tp_price_final is not None:
                order["take_profit"] = round(tp_price_final, 2)
                tp_distance = abs(tp_price_final - price)
                meta["tp_price"] = round(tp_price_final, 2)
                meta["tp_distance"] = round(tp_distance, 2)
                if sl_distance > 0 and tp_distance > 0:
                    meta["raw_rr"] = round(tp_distance / sl_distance, 2)

    # 手续费信息
    # Fee estimate uses executed qty (flip may execute > open_qty)
    meta["fee"] = compute_fee_info(price, total_qty, sl_distance, tp_distance, fee_rate)
    meta["entry_price"] = price
    meta["open_qty"] = open_qty
    meta["mode"] = "trailing_stop" if use_trailing_stop else "fixed_tpsl"
    meta["tp_sl_mode"] = order.get("tp_sl_mode", "PRICE")

    sig_name = "LONG" if signal == 1 else "SHORT"
    return {"signal": sig_name, "order": order, "meta": meta}


# ---------------------------------------------------------------------------
# 追踪止损
# ---------------------------------------------------------------------------

def compute_trailing_stop(
    current_pos: float,
    price: float,
    atr: float,
    atr_sl_multiplier: float,
    entry_price: Optional[float],
    best_price: Optional[float],
) -> tuple:
    """计算追踪止损。返回 (info_dict, updated_best_price)。"""
    if current_pos == 0 or atr <= 0:
        return {
            "active": False, "should_update": False, "sl_price": None,
            "best_price": None, "entry_price": entry_price,
        }, best_price

    sl_distance = atr * atr_sl_multiplier

    if current_pos > 0:
        if best_price is None or price > best_price:
            best_price = price
        trailing_sl = round(best_price - sl_distance, 2)
    else:
        if best_price is None or price < best_price:
            best_price = price
        trailing_sl = round(best_price + sl_distance, 2)

    should_update = False
    if entry_price is not None:
        if current_pos > 0:
            initial_sl = entry_price - sl_distance
            should_update = trailing_sl > initial_sl + sl_distance * 0.1
        else:
            initial_sl = entry_price + sl_distance
            should_update = trailing_sl < initial_sl - sl_distance * 0.1

    info = {
        "active": True,
        "should_update": should_update,
        "sl_price": trailing_sl,
        "best_price": best_price,
        "entry_price": entry_price,
        "sl_distance": round(sl_distance, 2),
    }
    return info, best_price


# ---------------------------------------------------------------------------
# 因子快照工具 — 供所有策略在 __call__ 末尾调用
# ---------------------------------------------------------------------------

def build_factors_json(df, factor_cols: List[str], extra: Optional[Dict[str, Any]] = None) -> str:
    """从 df 最后一行提取指定因子列的值，序列化为 JSON 字符串。

    Args:
        df:          策略 __call__ 传入的 OHLCV + 因子 DataFrame。
        factor_cols: 要记录的列名列表，例如 ["rsi_14", "adx_14", "atr_14"]。
                     不存在或 NaN 的列自动跳过，保持 JSON 紧凑。
        extra:       额外的标量字段（如参数配置、风控状态），直接合并进快照。

    Returns:
        JSON 字符串，失败时返回空字符串。
    """
    try:
        snapshot: Dict[str, Any] = {}

        # 因子数值
        for col in factor_cols:
            if col not in df.columns:
                continue
            v = df[col].iloc[-1]
            try:
                if np.isnan(float(v)):
                    continue
            except (TypeError, ValueError):
                continue
            try:
                snapshot[col] = round(float(v), 6)
            except (TypeError, ValueError):
                pass

        # 额外字段（参数/风控）
        if extra:
            for k, v in extra.items():
                if v is None:
                    continue
                snapshot[k] = v

        return _json.dumps(snapshot, ensure_ascii=False)
    except Exception:
        return ""
