"""Одноразовый инъектор курсового шока: в shock_time выставляет крупный market-order на FX"""

from agent.TradingAgent import TradingAgent


class FxShockAgent(TradingAgent):
    def __init__(
        self,
        id,
        name,
        type,
        symbol,
        shock_time,
        shock_size,
        is_buy,
        starting_cash,
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
        self.symbol = symbol
        self.shock_time = shock_time
        self.shock_size = int(shock_size)
        self.is_buy = bool(is_buy)
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
        # Защита от шока за границей торгового дня
        if self.shock_time > self.mkt_close:
            self.fired = True
            return

        self.placeMarketOrder(
            self.symbol,
            self.shock_size,
            is_buy_order=self.is_buy,
            ignore_risk=True,
        )
        self.logEvent("FX_SHOCK_FIRED", {
            "symbol": self.symbol,
            "size": self.shock_size,
            "is_buy": self.is_buy,
            "scheduled_time": str(self.shock_time),
            "fired_time": str(currentTime),
        })
        self.fired = True

    def orderExecuted(self, order):
        super().orderExecuted(order)
        self.last_trade[order.symbol] = order.fill_price

    def getWakeFrequency(self):
        import pandas as pd
        return pd.Timedelta("1s")
