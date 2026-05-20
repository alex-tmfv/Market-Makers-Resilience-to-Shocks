"""Базовый класс для агентов, торгующих в двух валютах одновременно"""

from copy import deepcopy

from agent.FinancialAgent import FinancialAgent
from agent.TradingAgent import TradingAgent
from message.Message import Message
from util.util import log_print


# Маппинг symbol → валюта. FX — псевдо-рынок, представляющий обмен валюты.
DEFAULT_CURRENCY_OF_SYMBOL = {
    "A_USD": "USD",
    "B_EUR": "EUR",
    "FX":    "FX",
}


class MultiCurrencyTradingAgent(TradingAgent):

    def __init__(
        self,
        id,
        name,
        type,
        starting_cash_usd,
        starting_cash_eur,
        initial_fx_price,
        get_fx_mid,
        starting_assets=None,
        starting_asset_prices=None,
        currency_of_symbol=None,
        random_state=None,
        log_orders=False,
        log_to_file=True,
        maker_rebate_cents_per_share=0.0,
    ):
        if starting_assets is None:
            starting_assets = {}
        if starting_asset_prices is None:
            starting_asset_prices = {}

        # USD-эквивалент стартового капитала: cash + стартовые активы (через цены).
        # Без `starting_asset_prices` kernel.meanResult пере-оценит PnL по типу
        # агента на стоимость стартовых активов.
        currency_map = (currency_of_symbol if currency_of_symbol is not None
                        else dict(DEFAULT_CURRENCY_OF_SYMBOL))
        assets_value_usd_cents = 0
        for sym, qty in starting_assets.items():
            price = starting_asset_prices.get(sym)
            if price is None:
                continue
            ccy = currency_map.get(sym)
            if ccy == "USD":
                assets_value_usd_cents += qty * price
            elif ccy == "EUR":
                assets_value_usd_cents += int(round(qty * price * initial_fx_price / 1000))

        starting_cash_usd_equiv = int(round(
            starting_cash_usd
            + starting_cash_eur * initial_fx_price / 1000
            + assets_value_usd_cents
        ))

        super().__init__(
            id, name, type,
            starting_cash=starting_cash_usd_equiv,
            random_state=random_state,
            log_orders=log_orders,
            log_to_file=log_to_file,
        )

        # Двухвалютный holdings + стартовые активы (вместо стандартного {'CASH'}).
        self.holdings = {"CASH_USD": starting_cash_usd, "CASH_EUR": starting_cash_eur}
        for sym, qty in starting_assets.items():
            self.holdings[sym] = qty

        self.get_fx_mid = get_fx_mid
        self.initial_fx_price = initial_fx_price
        self.currency_of_symbol = currency_map

        # Отдельные ссылки на стартовые балансы — для kernelStopping.
        self.starting_cash_usd = starting_cash_usd
        self.starting_cash_eur = starting_cash_eur

        # Maker rebate в валюте сделки за каждый passive fill. MM ставят только
        # лимит-ордера, которые обычно становятся maker'ами (marketable limit'ы
        # в нашем сетапе пренебрежимо редки). NASDAQ rebate ~0.2-0.3 c/share.
        # 0 — выключено (academic mode, только spread capture).
        self.maker_rebate_cents_per_share = float(maker_rebate_cents_per_share)

    def orderExecuted(self, order):
        log_print("Received notification of execution for: {}", order)
        if self.log_orders:
            self.logEvent("ORDER_EXECUTED", order.to_dict())

        signed_qty = order.quantity if order.is_buy_order else -order.quantity
        sym = order.symbol
        price = order.fill_price
        currency = self.currency_of_symbol.get(sym)

        if currency == "FX":
            # 1 «акция» FX = 1 EUR = 100 EUR-cents; price в pips, /10 → USD-cents.
            self.holdings["CASH_EUR"] = self.holdings.get("CASH_EUR", 0) + signed_qty * 100
            self.holdings["CASH_USD"] = self.holdings.get("CASH_USD", 0) - int(round(signed_qty * price / 10))

        elif currency == "USD":
            self.holdings[sym] = self.holdings.get(sym, 0) + signed_qty
            if self.holdings[sym] == 0:
                del self.holdings[sym]
            self.holdings["CASH_USD"] = self.holdings.get("CASH_USD", 0) - signed_qty * price

        elif currency == "EUR":
            self.holdings[sym] = self.holdings.get(sym, 0) + signed_qty
            if self.holdings[sym] == 0:
                del self.holdings[sym]
            self.holdings["CASH_EUR"] = self.holdings.get("CASH_EUR", 0) - signed_qty * price

        else:
            raise ValueError(f"symbol {sym!r} не в currency_of_symbol")

        # Maker rebate в валюте сделки (FX исключён — там нет passive MM-логики).
        if self.maker_rebate_cents_per_share > 0.0 and currency != "FX":
            rebate = int(round(abs(signed_qty) * self.maker_rebate_cents_per_share))
            cash_key = "CASH_USD" if currency == "USD" else "CASH_EUR"
            self.holdings[cash_key] = self.holdings.get(cash_key, 0) + rebate
            self.logEvent("MAKER_REBATE", {
                "symbol": sym, "currency": currency,
                "qty": int(abs(signed_qty)), "rebate_cents": rebate,
            })

        # Снять order из открытого списка (копия базовой логики).
        if order.order_id in self.orders:
            o = self.orders[order.order_id]
            if order.quantity >= o.quantity:
                del self.orders[order.order_id]
            else:
                o.quantity -= order.quantity
        else:
            log_print("Execution received for order not in orders list: {}", order)

        # dict(...) — копия; без неё все события ссылаются на один объект.
        self.logEvent("HOLDINGS_SNAPSHOT", dict(self.holdings))
        self.logEvent("TRADE", {
            "symbol": sym, "currency": currency,
            "qty": signed_qty, "price": price,
        })

    def markToMarket(self, holdings, use_midpoint=False):
        # USD-cents через last_trade и текущий FX-mid. Символы без
        # last_trade пропускаются (нормально на старте дня).
        fx = self.get_fx_mid()
        total_usd = (holdings.get("CASH_USD", 0)
                     + int(round(holdings.get("CASH_EUR", 0) * fx / 1000)))

        for symbol, qty in holdings.items():
            if symbol in ("CASH_USD", "CASH_EUR"):
                continue
            currency = self.currency_of_symbol.get(symbol)
            last = self.last_trade.get(symbol)
            if last is None:
                continue
            if currency == "USD":
                total_usd += qty * last
            elif currency == "EUR":
                total_usd += int(round(qty * last * fx / 1000))
            elif currency == "FX":
                # Не должно случаться: holdings['FX'] не ведётся.
                total_usd += int(round(qty * fx / 10)) * last
            self.logEvent(
                "MARK_TO_MARKET",
                f"{qty} {symbol} @ {last} ({currency}) -> total USD={total_usd}",
            )

        self.logEvent("MARKED_TO_MARKET", total_usd)
        return total_usd

    def kernelStopping(self):
        # Скипаем TradingAgent.kernelStopping — он упал бы на holdings['CASH'].
        FinancialAgent.kernelStopping(self)

        self.logEvent("FINAL_HOLDINGS", self.fmtHoldings(self.holdings))
        self.logEvent("FINAL_CASH_USD", self.holdings.get("CASH_USD", 0), True)
        self.logEvent("FINAL_CASH_EUR", self.holdings.get("CASH_EUR", 0), True)

        cash_usd_equiv = self.markToMarket(self.holdings)
        self.logEvent("ENDING_CASH_USD_EQUIV", cash_usd_equiv, True)

        print(
            f"Final holdings for {self.name}: {self.fmtHoldings(self.holdings)}.  "
            f"Marked to market (USD-equiv): {cash_usd_equiv}"
        )

        gain = cash_usd_equiv - self.starting_cash
        mytype = self.type
        if mytype in self.kernel.meanResultByAgentType:
            self.kernel.meanResultByAgentType[mytype] += gain
            self.kernel.agentCountByType[mytype] += 1
        else:
            self.kernel.meanResultByAgentType[mytype] = gain
            self.kernel.agentCountByType[mytype] = 1

    def fmtHoldings(self, holdings):
        parts = [f"CASH_USD: {holdings.get('CASH_USD', 0)}",
                 f"CASH_EUR: {holdings.get('CASH_EUR', 0)}"]
        for k, v in holdings.items():
            if k in ("CASH_USD", "CASH_EUR"):
                continue
            parts.append(f"{k}: {v}")
        return "{ " + ", ".join(parts) + " }"
