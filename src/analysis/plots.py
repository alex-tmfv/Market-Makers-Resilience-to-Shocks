"""Графики для анализа multi-market симуляции"""

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis import load, metrics


def plot_bid_ask_mid(run_name, base=None, figsize=(12, 9)):
    """Bid / Ask / Mid на каждом из трёх рынков (A_USD, B_EUR, FX)"""
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    symbols = [
        ("A_USD", "USD-cents", "tab:blue"),
        ("B_EUR", "EUR-cents", "tab:orange"),
        ("FX",    "USD-pips/EUR", "tab:green"),
    ]
    for ax, (sym, ylabel, mid_color) in zip(axes, symbols):
        ob = load.load_orderbook(run_name, sym, base)
        bba = metrics.best_bid_ask(ob)
        mid = (bba["best_bid"] + bba["best_ask"]) / 2

        ax.plot(bba.index, bba["best_bid"], color="tab:green",
                linewidth=0.5, alpha=0.5, label="bid")
        ax.plot(bba.index, bba["best_ask"], color="tab:red",
                linewidth=0.5, alpha=0.5, label="ask")
        ax.plot(mid.index, mid, color=mid_color, linewidth=1.2, label="mid")
        ax.set_title(f"{sym}")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Время")
    fig.tight_layout()
    return fig


plot_mid_prices = plot_bid_ask_mid


def plot_bid_ask_mid_with_reference(run_name, ref_run, base=None, figsize=(12, 9),
                                    ref_label="stationary"):
    """График для сравнения шока vs stationary при одинаковом сиде"""
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    symbols = [
        ("A_USD", "USD-cents", "tab:blue"),
        ("B_EUR", "EUR-cents", "tab:orange"),
        ("FX",    "USD-pips/EUR", "tab:green"),
    ]
    for ax, (sym, ylabel, mid_color) in zip(axes, symbols):
        ob = load.load_orderbook(run_name, sym, base)
        bba = metrics.best_bid_ask(ob)
        mid = (bba["best_bid"] + bba["best_ask"]) / 2

        ref_ob = load.load_orderbook(ref_run, sym, base)
        ref_bba = metrics.best_bid_ask(ref_ob)
        ref_mid = (ref_bba["best_bid"] + ref_bba["best_ask"]) / 2
        ax.plot(ref_mid.index, ref_mid, color="gray", linewidth=1.0,
                linestyle="--", alpha=0.7, label=f"mid ({ref_label})")

        ax.plot(bba.index, bba["best_bid"], color="tab:green",
                linewidth=0.5, alpha=0.5, label="bid")
        ax.plot(bba.index, bba["best_ask"], color="tab:red",
                linewidth=0.5, alpha=0.5, label="ask")
        ax.plot(mid.index, mid, color=mid_color, linewidth=1.2, label="mid (shock)")
        ax.set_title(f"{sym}")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Время")
    fig.tight_layout()
    return fig


def plot_arb_check(run_name, base=None, figsize=(12, 5), smooth_window="30s"):
    """График для проверки арбитражной связи между B_EUR и A_USD через FX"""
    ob_a = load.load_orderbook(run_name, "A_USD", base)
    ob_b = load.load_orderbook(run_name, "B_EUR", base)
    ob_fx = load.load_orderbook(run_name, "FX", base)

    mid_a = metrics.mid_price(ob_a)
    mid_b = metrics.mid_price(ob_b)
    mid_fx = metrics.mid_price(ob_fx)

    df = pd.concat({"a": mid_a, "b": mid_b, "fx": mid_fx}, axis=1).dropna()
    implied_b = df["a"] * 1000 / df["fx"]
    implied_b_smoothed = implied_b.rolling(smooth_window).mean()
    discrepancy = df["b"] - implied_b
    rel_pct = discrepancy.abs() / df["b"] * 100

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axes[0].plot(df.index, implied_b, label=f"P_A · 1000 / P_FX (implied, raw)",
                 color="tab:blue", alpha=0.25, linewidth=0.7)
    axes[0].plot(df.index, implied_b_smoothed,
                 label=f"implied (rolling {smooth_window})",
                 color="tab:blue", linewidth=1.5)
    axes[0].plot(df.index, df["b"], label="P_B (actual)",
                 color="tab:orange", linewidth=1.5)
    axes[0].legend(loc="best")
    axes[0].set_ylabel("EUR-cents")
    axes[0].set_title("Арбитражная связь B_EUR: actual vs fair-value (через FX)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df.index, discrepancy, color="tab:red", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("EUR-cents")
    axes[1].set_title(
        f"Discrepancy = P_B − implied  "
        f"(mean={discrepancy.mean():.0f}, std={discrepancy.std():.0f}, "
        f"|max|={discrepancy.abs().max():.0f} = {rel_pct.max():.2f}% от P_B)"
    )
    axes[1].set_xlabel("Время")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_holdings(run_name, agent_name, base=None, figsize=(12, 6)):
    """Динамика инвентаря агента"""
    log = load.load_agent_log(run_name, agent_name, base)
    hh = metrics.holdings_history(log)

    if hh.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5,
                f"{agent_name}: нет сделок\n"
                "(агент не торговал в этом запуске, "
                "либо не multi-currency)",
                ha='center', va='center', fontsize=14,
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Позиции в активах (всё кроме CASH_USD/CASH_EUR)
    asset_cols = [c for c in hh.columns if c not in ("CASH_USD", "CASH_EUR")]
    for c in asset_cols:
        axes[0].plot(hh.index, hh[c], label=c, marker=".", markersize=3, linewidth=1)
    axes[0].legend()
    axes[0].set_ylabel("Позиция (шт)")
    axes[0].set_title(f"{agent_name}: позиции в активах")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].grid(True, alpha=0.3)

    # Cash в двух валютах
    for c in ("CASH_USD", "CASH_EUR"):
        if c in hh.columns:
            axes[1].plot(hh.index, hh[c], label=c, marker=".", markersize=3, linewidth=1)
    axes[1].legend()
    axes[1].set_ylabel("Cash (центы)")
    axes[1].set_title(f"{agent_name}: денежные позиции")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("Время")

    fig.tight_layout()
    return fig


def plot_volume_by_agent_type(run_name, base=None, freq="1min",
                              figsize=(12, 10), as_notional=True):
    """Объёмы торгов по типу агента на каждом рынке"""
    df = metrics.parse_executions(load, run_name, base=base)
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Нет ORDER_EXECUTED событий\n"
                          "(нужен log_orders=True у ExchangeAgent)",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    df = df.copy()
    df["bucket"] = df.index.floor(freq)

    value_col = "notional_usd_cents" if as_notional else "quantity"
    ylabel_unit = "Notional, USD-cents" if as_notional else "Шт"

    symbols = ["A_USD", "B_EUR", "FX"]
    fig, axes = plt.subplots(len(symbols), 1, figsize=figsize, sharex=True)

    for ax, sym in zip(axes, symbols):
        sym_df = df[df["symbol"] == sym]
        if sym_df.empty:
            ax.set_title(f"{sym}: нет сделок")
            ax.set_axis_off()
            continue

        pivot = sym_df.pivot_table(
            index="bucket", columns="agent_type",
            values=value_col, aggfunc="sum", fill_value=0,
        )
        # Фиксированная палитра, чтобы один тип агента имел одинаковый цвет на всех трёх графиках
        type_colors = {
            "NoiseAgent":            "tab:blue",
            "ZeroIntelligenceAgent": "tab:orange",
            "ValueAgent":            "tab:green",
            "MomentumAgent":         "tab:red",
            "FxArbAgent":            "tab:brown",
        }
        mm_color = "tab:purple"
        type_order = [
            "NoiseAgent", "ZeroIntelligenceAgent", "ValueAgent",
            "MomentumAgent", "BaselineMM", "AvellanedaStoikovMM",
            "GLFTMM", "CMGLFTMM", "RLGLFTMM", "SpreadBasedMM",
            "AdaptiveMM", "FxArbAgent",
        ]
        cols = [c for c in type_order if c in pivot.columns]
        pivot = pivot[cols]
        colors = [type_colors.get(c, mm_color) for c in cols]

        ax.stackplot(pivot.index, pivot.T.values, labels=cols,
                     colors=colors, alpha=0.85)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylabel(f"{ylabel_unit}, бин {freq}")
        ax.set_title(f"{sym}: объём торгов по типу агента")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Время")
    fig.tight_layout()
    return fig


def plot_fundamental_vs_market(run_name, base=None, figsize=(12, 8)):
    """Сравнение рыночного mid с истинным fundamental на A_USD и B_EUR.
    Для B_EUR fundamental = r_company · 1000 / P_FX_market."""
    fund = load.load_fundamental(run_name, "A_USD", base)  # r_company
    ob_a = load.load_orderbook(run_name, "A_USD", base)
    ob_b = load.load_orderbook(run_name, "B_EUR", base)
    ob_fx = load.load_orderbook(run_name, "FX", base)

    mid_a = metrics.mid_price(ob_a)
    mid_b = metrics.mid_price(ob_b)
    mid_fx = metrics.mid_price(ob_fx)


    mkt_close = mid_a.index[-1] if len(mid_a) else None
    if mkt_close is not None:
        fund = fund[fund.index <= mkt_close]

    fund_series = fund["FundamentalValue"]
    r_company_aligned = fund_series.reindex(
        fund_series.index.union(mid_fx.index)
    ).ffill().reindex(mid_fx.index)
    fundamental_b = r_company_aligned * 1000 / mid_fx

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=False)

    # --- A_USD ---
    axes[0].plot(fund.index, fund_series, label="r_company (oracle)",
                 color="tab:gray", linewidth=1, alpha=0.8)
    axes[0].plot(mid_a.index, mid_a.values, label="P_A market mid",
                 color="tab:blue", linewidth=1.5)
    axes[0].legend(loc="best")
    axes[0].set_ylabel("USD-cents")
    axes[0].set_title("A_USD: market mid vs fundamental r_company")
    axes[0].grid(True, alpha=0.3)

    # --- B_EUR ---
    axes[1].plot(fundamental_b.index, fundamental_b.values,
                 label="r_company · 1000 / P_FX (formula)",
                 color="tab:gray", linewidth=1, alpha=0.8)
    axes[1].plot(mid_b.index, mid_b.values, label="P_B market mid",
                 color="tab:orange", linewidth=1.5)
    axes[1].legend(loc="best")
    axes[1].set_ylabel("EUR-cents")
    axes[1].set_title("B_EUR: market mid vs fundamental (через FX)")
    axes[1].set_xlabel("Время")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
