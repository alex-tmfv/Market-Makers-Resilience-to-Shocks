"""Одноразовый шок ликвидности: в shock_time очищает одну сторону стакана. Имитирует массовый withdrawal of quotes."""

import pandas as pd

from agent.TradingAgent import TradingAgent


class LiquidityShockAgent(TradingAgent):
    def __init__(
        self,
        id,
        name,
        type,
        symbol,
        shock_time,
        side,
        starting_cash=100_000,
        log_orders=True,
        random_state=None,
    ):
        super().__init__(
            id,
            name,
            type,
            starting_cash=starting_cash,
            log_orders=log_orders,
            random_state=random_state,
        )
        if side not in ("bid", "ask"):
            raise ValueError(f"side must be 'bid' or 'ask', got {side!r}")
        self.symbol = symbol
        self.shock_time = shock_time
        self.side = side
        self.fired = False

    def wakeup(self, currentTime):
        super().wakeup(currentTime)

        if not self.mkt_open or not self.mkt_close:
            return

        if self.fired:
            return

        if currentTime < self.shock_time:
            self.setWakeup(self.shock_time)
            return

        if self.shock_time > self.mkt_close:
            self.fired = True
            return

        # Прямой доступ к стакану через kernel (обход message-протокола).
        exchange = self.kernel.agents[self.exchangeID]
        ob = exchange.order_books[self.symbol]

        cleared_orders = 0
        cleared_volume = 0
        if self.side == "bid":
            for level in ob.bids:
                cleared_orders += len(level)
                cleared_volume += sum(int(o.quantity) for o in level)
            ob.bids = []
        else:  # ask
            for level in ob.asks:
                cleared_orders += len(level)
                cleared_volume += sum(int(o.quantity) for o in level)
            ob.asks = []

        self.logEvent("LIQUIDITY_SHOCK_FIRED", {
            "symbol": self.symbol,
            "side": self.side,
            "cleared_orders": cleared_orders,
            "cleared_volume": cleared_volume,
            "scheduled_time": str(self.shock_time),
            "fired_time": str(currentTime),
        })
        self.fired = True

    def getWakeFrequency(self):
        return pd.Timedelta("1s")
