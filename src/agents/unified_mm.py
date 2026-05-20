"""MM базовый агент, в который встраивается разная логика торговли из mm_strategies"""

import pandas as pd

from src.multimarket.multi_currency_agent import MultiCurrencyTradingAgent


STATE_AWAITING_WAKEUP = "AWAITING_WAKEUP"
STATE_AWAITING_SPREAD = "AWAITING_SPREAD"


class UnifiedMM(MultiCurrencyTradingAgent):

    def __init__(
        self,
        *,
        symbol,
        strategy,
        wake_up_freq,
        mkt_close,
        shared_state=None,
        **mc_kwargs,
    ):
        super().__init__(**mc_kwargs)
        self.symbol = symbol
        self.strategy = strategy
        self.wake_up_freq = wake_up_freq
        self.mkt_close = mkt_close
        self.shared_state = shared_state
        # Каждый MM-инстанс автоматически регистрирует себя в общем shared_state.
        if shared_state is not None:
            shared_state.register(symbol, self)
        self.state = STATE_AWAITING_WAKEUP

    def wakeup(self, currentTime):
        ready = super().wakeup(currentTime)
        if not ready:
            return
        self._cancel_all_orders()
        self.getCurrentSpread(self.symbol)
        self.state = STATE_AWAITING_SPREAD
        # Poisson wake: exp inter-arrival с mean = wake_up_freq (как у Value / ZI).
        delta_sec = float(self.random_state.exponential(scale=self.wake_up_freq.total_seconds()))
        self.setWakeup(currentTime + pd.Timedelta(seconds=delta_sec))

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if self.state == STATE_AWAITING_SPREAD and msg.body["msg"] == "QUERY_SPREAD":
            self._place_quotes()
            self.state = STATE_AWAITING_WAKEUP

    def _place_quotes(self):
        bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
        if bid is None or ask is None:
            # Стакан пустой — fallback на last_trade с искусственным ±1 спредом.
            if self.symbol in self.last_trade and self.last_trade[self.symbol] is not None:
                mid = float(self.last_trade[self.symbol])
                bid, ask = mid - 1, mid + 1
            else:
                return
        else:
            mid = (bid + ask) / 2.0

        inventory = self.holdings.get(self.symbol, 0)
        time_to_close = (self.mkt_close - self.currentTime).total_seconds()
        equity = self._compute_equity()

        orders = self.strategy.compute_quotes(
            mid=mid, bid=bid, ask=ask,
            inventory=inventory, time_to_close=time_to_close,
            current_time=self.currentTime,
            equity=equity,
            shared_state=self.shared_state,
        )

        for price, size, is_buy in orders:
            if size <= 0:
                continue
            self.placeLimitOrder(
                self.symbol, int(size), is_buy_order=is_buy,
                limit_price=int(price), ignore_risk=True,
            )

        self.logEvent("MM_QUOTES_PLACED", {
            "mid": float(mid),
            "inventory": int(inventory),
            "n_orders": len(orders),
        })

    def _cancel_all_orders(self):
        for order in list(self.orders.values()):
            self.cancelOrder(order)

    def _compute_equity(self):
        # markToMarket() из базового класса пишет лог на каждый wakeup —
        # нам нужен только числовой equity для reward'а RL-агента.
        fx = self.get_fx_mid()
        eq = self.holdings.get("CASH_USD", 0) + int(round(self.holdings.get("CASH_EUR", 0) * fx / 1000))
        for sym, qty in self.holdings.items():
            if sym in ("CASH_USD", "CASH_EUR"):
                continue
            last = self.last_trade.get(sym)
            if last is None:
                continue
            ccy = self.currency_of_symbol.get(sym)
            if ccy == "USD":
                eq += qty * last
            elif ccy == "EUR":
                eq += int(round(qty * last * fx / 1000))
        return eq

    def getWakeFrequency(self):
        return self.wake_up_freq
