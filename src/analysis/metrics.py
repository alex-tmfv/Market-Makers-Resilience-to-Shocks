"""Метрики из логов симуляции: bid/ask/mid, спреды, P&L, объёмы по типам
агентов, оценка волатильности"""

import numpy as np
import pandas as pd


def best_bid_ask(orderbook_df):
    # отрицательные значения — bid, положительные — ask.
    levels = np.array(orderbook_df.columns, dtype=float)
    bid_levels = orderbook_df.where(orderbook_df < 0).apply(
        lambda row: float(levels[row.notna().values].max()) if row.notna().any() else np.nan,
        axis=1,
    )
    ask_levels = orderbook_df.where(orderbook_df > 0).apply(
        lambda row: float(levels[row.notna().values].min()) if row.notna().any() else np.nan,
        axis=1,
    )
    return pd.DataFrame({"best_bid": bid_levels, "best_ask": ask_levels})


def mid_price(orderbook_df):
    bba = best_bid_ask(orderbook_df)
    return (bba["best_bid"] + bba["best_ask"]) / 2.0


def spread(orderbook_df):
    bba = best_bid_ask(orderbook_df)
    return bba["best_ask"] - bba["best_bid"]


def arb_discrepancy(mid_a, mid_b, mid_fx):
    # P_B − P_A·1000/P_FX в EUR-cents; в равновесии ≈ 0.
    return mid_b - mid_a * 1000 / mid_fx


def holdings_history(agent_log_df, holdings_col="HOLDINGS_SNAPSHOT"):
    rows = agent_log_df[agent_log_df["EventType"] == holdings_col]
    if rows.empty:
        return pd.DataFrame()
    return pd.DataFrame(list(rows["Event"]), index=rows.index).fillna(0)


def trades_history(agent_log_df):
    rows = agent_log_df[agent_log_df["EventType"] == "TRADE"]
    if rows.empty:
        return pd.DataFrame(columns=["symbol", "currency", "qty", "price"])
    return pd.DataFrame(list(rows["Event"]), index=rows.index)


def parse_agent_meta(name):
    import re

    if name == "FxArbAgent":
        return ("FxArbAgent", "multi")
    if name.startswith("BaselineMM_"):
        return ("BaselineMM", name[len("BaselineMM_"):])
    if name.startswith("AvellanedaStoikovMM_"):
        return ("AvellanedaStoikovMM", name[len("AvellanedaStoikovMM_"):])
    if name.startswith("GLFTMM_"):
        return ("GLFTMM", name[len("GLFTMM_"):])
    if name.startswith("CMGLFTMM_"):
        return ("CMGLFTMM", name[len("CMGLFTMM_"):])
    if name.startswith("RLGLFTMM_"):
        return ("RLGLFTMM", name[len("RLGLFTMM_"):])
    if name.startswith("SpreadBasedMM_"):
        return ("SpreadBasedMM", name[len("SpreadBasedMM_"):])
    if name.startswith("AdaptiveMM_"):
        return ("AdaptiveMM", name[len("AdaptiveMM_"):])
    m = re.match(r"^(NoiseAgent|ValueAgent|ZIAgent|MomentumAgent)_(A_USD|B_EUR|FX)_\d+$", name)
    if m:
        type_, market = m.group(1), m.group(2)
        if type_ == "ZIAgent":
            type_ = "ZeroIntelligenceAgent"
        return (type_, market)
    return ("Unknown", "Unknown")


def market_summary_table(load_module, run_name, base=None):
    # По-агентная сводка
    rows = []
    for name in load_module.list_agents(run_name, base=base):
        agent_type, market = parse_agent_meta(name)
        log = load_module.load_agent_log(run_name, name, base=base)

        starting = log.loc[log["EventType"] == "STARTING_CASH", "Event"]
        ending_usd = log.loc[log["EventType"] == "ENDING_CASH_USD_EQUIV", "Event"]
        ending_legacy = log.loc[log["EventType"] == "ENDING_CASH", "Event"]
        starting_cash = float(starting.iloc[0]) if not starting.empty else None

        if starting_cash is not None and not ending_usd.empty:
            pnl = float(ending_usd.iloc[0]) - starting_cash
        elif starting_cash is not None and not ending_legacy.empty:
            pnl = float(ending_legacy.iloc[0]) - starting_cash
        else:
            pnl = None

        trades = log[log["EventType"] == "TRADE"]
        trade_count = len(trades) if not trades.empty else None
        trade_volume_total = (sum(abs(t["qty"]) for t in trades["Event"])
                              if not trades.empty else None)

        rows.append({
            "market": market, "type": agent_type, "name": name,
            "starting_cash": starting_cash, "pnl_usd_cents": pnl,
            "trade_count": trade_count, "trade_volume": trade_volume_total,
        })
    return pd.DataFrame(rows)


def market_summary_aggregated(load_module, run_name, base=None):
    df = market_summary_table(load_module, run_name, base=base)
    df = df[df["type"] != "Unknown"]   # исключаем ExchangeAgent и пр.
    agg = df.groupby(["market", "type"]).agg(
        count=("name", "count"),
        total_starting_cash=("starting_cash", "sum"),
        total_pnl=("pnl_usd_cents", "sum"),
        mean_pnl=("pnl_usd_cents", "mean"),
        total_trades=("trade_count", "sum"),
        total_volume=("trade_volume", "sum"),
    ).reset_index()
    agg["pnl_pct"] = (
        agg["total_pnl"] / agg["total_starting_cash"] * 100
    ).where(agg["total_starting_cash"] > 0)
    return agg


def pnl_pivot_by_market(load_module, run_name, base=None, value="total_pnl"):
    agg = market_summary_aggregated(load_module, run_name, base=base)
    pivot = agg.pivot(index="type", columns="market", values=value).fillna(0)
    cols = [c for c in ("A_USD", "B_EUR", "FX", "multi") if c in pivot.columns]
    return pivot[cols]


def parse_executions(load_module, run_name, base=None):
    """ORDER_EXECUTED события из EXCHANGE_AGENT с обогащением agent_type /
    market / notional. Каждая matched-сделка даёт два event'а (buyer + seller) —
    оба включены, что даёт объём по агенту, а не market volume."""
    import re

    log = load_module.load_agent_log(run_name, "EXCHANGE_AGENT", base=base)
    exec_log = log[log["EventType"] == "ORDER_EXECUTED"]
    if exec_log.empty:
        return pd.DataFrame(columns=[
            "agent_id", "agent_name", "agent_type", "market",
            "symbol", "quantity", "is_buy_order", "fill_price",
        ])

    df = pd.DataFrame(list(exec_log["Event"]), index=exec_log.index)
    df = df[["agent_id", "symbol", "quantity", "is_buy_order", "fill_price"]].copy()
    df["quantity"] = df["quantity"].astype(int)
    df["agent_id"] = df["agent_id"].astype(int)

    agents = load_module.list_agents(run_name, base=base)
    name_by_id = {}
    no_trailing_id = []
    for name in agents:
        m = re.match(r"^(.+)_(\d+)$", name)
        if m:
            name_by_id[int(m.group(2))] = name
        else:
            no_trailing_id.append(name)

    summary = load_module.load_summary(run_name, base=base)
    id_to_strategy = (
        summary.drop_duplicates(subset=["AgentID"])
               .set_index("AgentID")["AgentStrategy"]
               .to_dict()
    )
    for aid, strategy in id_to_strategy.items():
        if aid in name_by_id:
            continue
        for n in no_trailing_id:
            if n.startswith(strategy + "_") or n == strategy:
                name_by_id[aid] = n
                break

    rows = []
    for aid in df["agent_id"].unique():
        name = name_by_id.get(aid)
        t, m = parse_agent_meta(name) if name else ("Unknown", "Unknown")
        rows.append((aid, name, t, m))
    meta_df = pd.DataFrame(rows, columns=["agent_id", "agent_name", "agent_type", "market"])
    # `merge` сбрасывает DatetimeIndex — сохраняем и восстанавливаем.
    saved_index = df.index
    df = df.merge(meta_df, on="agent_id", how="left")
    df.index = saved_index

    def _notional(sym, qty, price):
        if sym == "A_USD":
            return qty * price
        if sym == "B_EUR":
            return int(round(qty * price * 1.1))
        if sym == "FX":
            return int(round(qty * price / 10))
        return qty * price

    df["notional_usd_cents"] = [
        _notional(s, q, p)
        for s, q, p in zip(df["symbol"], df["quantity"], df["fill_price"])
    ]
    return df


def volume_by_agent_type(load_module, run_name, base=None, freq="1min"):
    # Объёмы по (symbol, agent_type) для plot_volume_by_agent_type.
    df = parse_executions(load_module, run_name, base=base)
    if df.empty:
        return pd.DataFrame()
    df = df.assign(bucket=df.index.floor(freq))
    return df.groupby(["symbol", "agent_type", "bucket"])["quantity"].sum().unstack(fill_value=0)


def realized_volatility(mid_series, window="1min"):
    log_returns = np.log(mid_series).diff()
    return log_returns.rolling(window).std()


def estimate_volatility_abs(load_module, run_name, symbol, base=None):
    """σ в единицах цены/√sec (cents для акций, pips для FX). Используется
    для калибровки σ_A в AS / GLFT. С book_freq='1s' std(diff(mid)) = σ."""
    ob = load_module.load_orderbook(run_name, symbol, base=base)
    mid = mid_price(ob).dropna()
    if len(mid) < 2:
        return 0.0
    return float(mid.diff().dropna().std())
