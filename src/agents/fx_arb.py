"""Арбитражёр между A_USD, B_EUR и FX. Открывает позицию, когда |P_B − P_A·1000/P_FX| превышает open_threshold, и закрывает её при возврате под close_threshold"""

from src.multimarket.multi_currency_agent import MultiCurrencyTradingAgent


STATE_AWAITING_WAKEUP = "AWAITING_WAKEUP"
STATE_AWAITING_SPREADS = "AWAITING_SPREADS"


class FxArbAgent(MultiCurrencyTradingAgent):
    def __init__(
        self,
        *,
        symbol_a="A_USD",
        symbol_b="B_EUR",
        symbol_fx="FX",
        open_threshold_pct=2.0,
        close_threshold_pct=1.0,
        order_size=10,
        wake_up_freq,
        allow_short=True,
        starting_inventory_a=0,
        starting_inventory_b=0,
        max_exposure_per_side=100,
        **mc_kwargs,
    ):
        # Стартовые активы — для корректного USD-eq учёта starting_cash
        mc_kwargs.setdefault("starting_assets", {})
        mc_kwargs["starting_assets"][symbol_a] = starting_inventory_a
        mc_kwargs["starting_assets"][symbol_b] = starting_inventory_b

        super().__init__(**mc_kwargs)

        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.symbol_fx = symbol_fx
        self.open_threshold_pct = open_threshold_pct
        self.close_threshold_pct = close_threshold_pct
        self.order_size = order_size
        self.wake_up_freq = wake_up_freq
        self.allow_short = allow_short
        self.starting_inventory_a = starting_inventory_a
        self.starting_inventory_b = starting_inventory_b
        self.max_exposure_per_side = max_exposure_per_side

        if close_threshold_pct >= open_threshold_pct:
            raise ValueError(
                "close_threshold_pct должен быть строго меньше open_threshold_pct"
            )

        self.state = STATE_AWAITING_WAKEUP
        self._spreads_received = 0

    def wakeup(self, currentTime):
        ready = super().wakeup(currentTime)
        if not ready:
            return
        self._spreads_received = 0
        self.getCurrentSpread(self.symbol_a)
        self.getCurrentSpread(self.symbol_b)
        self.getCurrentSpread(self.symbol_fx)
        self.state = STATE_AWAITING_SPREADS
        self.setWakeup(currentTime + self.wake_up_freq)

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if self.state != STATE_AWAITING_SPREADS:
            return
        if msg.body["msg"] != "QUERY_SPREAD":
            return
        self._spreads_received += 1
        if self._spreads_received < 3:
            return
        self._maybe_trade()
        self.state = STATE_AWAITING_WAKEUP

    # --- Логика ---
    def _mid(self, symbol):
        bid, _, ask, _ = self.getKnownBidAsk(symbol)
        if bid is None or ask is None:
            return None
        return (bid + ask) // 2

    def _maybe_trade(self):
        p_a = self._mid(self.symbol_a)
        p_b = self._mid(self.symbol_b)
        p_fx = self._mid(self.symbol_fx)
        if p_a is None or p_b is None or p_fx is None or p_fx <= 0 or p_b <= 0:
            return

        fair_p_b = int(round(p_a * 1000 / p_fx))
        discrepancy = p_b - fair_p_b
        discr_pct = abs(discrepancy) / p_b * 100
        # delta_b > 0 — long (купили доп. B); delta_b < 0 — short
        delta_b = self.holdings.get(self.symbol_b, 0) - self.starting_inventory_b

        action = self._decide_action(discrepancy, discr_pct, delta_b)
        self.logEvent("ARB_CHECK", {
            "p_a": int(p_a), "p_b": int(p_b), "p_fx": int(p_fx),
            "fair_p_b": fair_p_b, "discrepancy": int(discrepancy),
            "discr_pct": round(discr_pct, 4),
            "delta_b": int(delta_b), "action": action,
        })
        if action is None:
            return

        bid_a, _, ask_a, _ = self.getKnownBidAsk(self.symbol_a)
        bid_b, _, ask_b, _ = self.getKnownBidAsk(self.symbol_b)
        bid_fx, _, ask_fx, _ = self.getKnownBidAsk(self.symbol_fx)

        if action == "sell_b":
            if bid_b is None or ask_a is None or bid_fx is None:
                return
            if not self._can_sell_b(bid_b):
                return
            self._execute(b_is_buy=False, b_price=bid_b,
                          a_is_buy=True,  a_price=ask_a,
                          fx_is_buy=False, fx_price=bid_fx,
                          eur_proceeds=self.order_size * bid_b)
        else:  # buy_b
            if ask_b is None or bid_a is None or ask_fx is None:
                return
            if not self._can_buy_b(ask_a, ask_b, ask_fx):
                return
            self._execute(b_is_buy=True,  b_price=ask_b,
                          a_is_buy=False, a_price=bid_a,
                          fx_is_buy=True, fx_price=ask_fx,
                          eur_proceeds=self.order_size * ask_b)

    def _decide_action(self, discrepancy, discr_pct, delta_b):
        if discr_pct >= self.open_threshold_pct:
            if discrepancy > 0 and delta_b > -self.max_exposure_per_side:
                return "sell_b"        # B переоценён, не уходим в глубокий short
            if discrepancy < 0 and delta_b < self.max_exposure_per_side:
                return "buy_b"         # B недооценён, не уходим в глубокий long
        elif discr_pct <= self.close_threshold_pct:
            if delta_b > 0:
                return "sell_b"        # закрываем long
            if delta_b < 0:
                return "buy_b"         # закрываем short
        return None

    # --- No-short проверки ---
    def _can_sell_b(self, bid_b):
        if not self.allow_short and self.holdings.get(self.symbol_b, 0) < self.order_size:
            return False
        return True

    def _can_buy_b(self, ask_a, ask_b, ask_fx):
        if not self.allow_short and self.holdings.get(self.symbol_a, 0) < self.order_size:
            return False
        return True

    # --- Размещение трёх ног (B / A / FX) ---
    def _execute(self, b_is_buy, b_price, a_is_buy, a_price,
                 fx_is_buy, fx_price, eur_proceeds):
        # Три ноги: B-order, зеркальная A-order, FX-конверсия EUR-выручки.
        qty_fx = max(1, int(round(eur_proceeds / 100)))
        self.logEvent("ARB_EXECUTE", {
            "b_is_buy": b_is_buy, "b_price": int(b_price), "b_qty": self.order_size,
            "a_is_buy": a_is_buy, "a_price": int(a_price), "a_qty": self.order_size,
            "fx_is_buy": fx_is_buy, "fx_price": int(fx_price), "fx_qty": qty_fx,
        })
        self.placeLimitOrder(self.symbol_b, self.order_size,
                             is_buy_order=b_is_buy, limit_price=b_price,
                             ignore_risk=True)
        self.placeLimitOrder(self.symbol_a, self.order_size,
                             is_buy_order=a_is_buy, limit_price=a_price,
                             ignore_risk=True)
        self.placeLimitOrder(self.symbol_fx, qty_fx,
                             is_buy_order=fx_is_buy, limit_price=fx_price,
                             ignore_risk=True)

    def getWakeFrequency(self):
        return self.wake_up_freq
