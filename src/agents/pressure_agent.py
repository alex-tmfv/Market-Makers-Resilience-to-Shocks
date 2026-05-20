"""Поддерживает одностороннее давление в стакане в течение заданного периода.
 Используется вместе с drift-шоком для имитации bear/bull-режима"""

import pandas as pd

from agent.TradingAgent import TradingAgent


STATE_AWAITING_WAKEUP = "AWAITING_WAKEUP"
STATE_AWAITING_SPREAD = "AWAITING_SPREAD"


class OneSidedPressureAgent(TradingAgent):
    def __init__(
        self,
        id,
        name,
        type,
        symbol,
        start_time,
        end_time,
        side,
        orders_per_wake,
        order_size,
        offset_ticks,
        wake_freq,
        starting_cash=1_000_000_000,
        log_orders=True,
        random_state=None,
    ):
        super().__init__(
            id, name, type,
            starting_cash=starting_cash, log_orders=log_orders, random_state=random_state,
        )
        if side not in ("ask", "bid"):
            raise ValueError(f"side must be 'ask' or 'bid', got {side!r}")
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        self.side = side
        self.orders_per_wake = int(orders_per_wake)
        self.order_size = int(order_size)
        self.offset_ticks = int(offset_ticks)
        self.wake_freq = wake_freq if isinstance(wake_freq, pd.Timedelta) else pd.Timedelta(wake_freq)

        self.state = STATE_AWAITING_WAKEUP
        self.finished = False

    def wakeup(self, currentTime):
        super().wakeup(currentTime)

        # Ждём пока TradingAgent узнает mkt_open/mkt_close
        if not self.mkt_open or not self.mkt_close:
            return

        if self.finished:
            return

        if currentTime < self.start_time:
            self.setWakeup(self.start_time)
            return

        if currentTime > self.end_time:
            self._cancel_all_orders()
            self.finished = True
            return

        # Внутри окна: отменяем старые ордера, запрашиваем spread
        self._cancel_all_orders()
        self.getCurrentSpread(self.symbol)
        self.state = STATE_AWAITING_SPREAD

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if self.state == STATE_AWAITING_SPREAD and msg.body["msg"] == "QUERY_SPREAD":
            self._place_pressure_orders()
            self.state = STATE_AWAITING_WAKEUP
            # Запланировать следующий wake
            next_t = currentTime + self.wake_freq
            if next_t <= self.end_time:
                self.setWakeup(next_t)
            else:
                self.finished = True

    def _place_pressure_orders(self):
        # N лимиток на одну цену (строят queue). Floor-cap не даёт цене пересечь противоположную сторону стакана
        bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
        if bid is None or ask is None:
            return   # стакан пустой — пропускаем

        if self.side == "ask":
            target = int(ask) - self.offset_ticks
            floor = int(bid) + 1                # не уходим в спред-cross
            price = max(target, floor)
            is_buy = False
        else:  # bid
            target = int(bid) + self.offset_ticks
            floor = int(ask) - 1
            price = min(target, floor)
            is_buy = True

        for _ in range(self.orders_per_wake):
            self.placeLimitOrder(
                self.symbol, self.order_size, is_buy_order=is_buy,
                limit_price=price, ignore_risk=True,
            )

        self.logEvent("PRESSURE_PLACED", {
            "symbol": self.symbol, "side": self.side,
            "orders": self.orders_per_wake, "price": price,
            "best_bid": int(bid), "best_ask": int(ask),
        })

    def _cancel_all_orders(self):
        for order in list(self.orders.values()):
            self.cancelOrder(order)

    def orderExecuted(self, order):
        super().orderExecuted(order)
        self.last_trade[order.symbol] = order.fill_price

    def getWakeFrequency(self):
        return self.wake_freq
