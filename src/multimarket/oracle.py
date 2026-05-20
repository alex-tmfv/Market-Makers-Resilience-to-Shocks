"""Oracle для трёх связанных рынков"""

from math import sqrt

from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle


R_COMPANY_KEY = "A_USD"


class MultiMarketOracle(SparseMeanRevertingOracle):

    A_USD = "A_USD"
    B_EUR = "B_EUR"
    FX = "FX"

    def __init__(
        self,
        mkt_open,
        mkt_close,
        r_company_cfg,
        get_fx_mid,
        initial_fx_price,
        manual_shocks=None,
    ):

        super().__init__(mkt_open, mkt_close, {R_COMPANY_KEY: r_company_cfg})

        self.get_fx_mid = get_fx_mid
        self.initial_fx_price = initial_fx_price
        self._r_company_bar = r_company_cfg["r_bar"]


        self._manual_shocks = sorted(manual_shocks or [], key=lambda x: x[0])

    def getDailyOpenPrice(self, symbol, mkt_open=None):
        if symbol == self.A_USD:
            return self._r_company_bar
        if symbol == self.B_EUR:
            return int(round(self._r_company_bar * 1000 / self.initial_fx_price))
        if symbol == self.FX:
            return self.initial_fx_price
        raise ValueError(f"Unknown symbol: {symbol}")

    def advance_fundamental_value_series(self, currentTime, symbol):
        if symbol == R_COMPANY_KEY and self._manual_shocks:
            while self._manual_shocks and self._manual_shocks[0][0] <= currentTime:
                t_shock, magnitude = self._manual_shocks.pop(0)
                if t_shock <= self.r[symbol][0]:
                    continue   # шок «в прошлом» — игнорируем
                v_pre = super().advance_fundamental_value_series(t_shock, symbol)
                v_post = max(0, int(round(v_pre + magnitude)))
                self.r[symbol] = (t_shock, v_post)
                last = self.f_log[symbol][-1] if self.f_log[symbol] else None
                entry = {"FundamentalTime": t_shock, "FundamentalValue": v_post}
                if last is not None and last.get("FundamentalTime") == t_shock:
                    self.f_log[symbol][-1] = entry
                else:
                    self.f_log[symbol].append(entry)
        return super().advance_fundamental_value_series(currentTime, symbol)

    def observePrice(self, symbol, currentTime, sigma_n=1000, random_state=None):
        if symbol == self.A_USD:
            return super().observePrice(symbol, currentTime,
                                        sigma_n=sigma_n, random_state=random_state)
        if symbol == self.B_EUR:
            r_company = self.advance_fundamental_value_series(currentTime, R_COMPANY_KEY)
            r_b_clean = int(round(r_company * 1000 / self.get_fx_mid()))
            return self._add_noise(r_b_clean, sigma_n, random_state)
        if symbol == self.FX:
            return self._add_noise(self.get_fx_mid(), sigma_n, random_state)
        raise ValueError(f"Unknown symbol: {symbol}")

    @staticmethod
    def _add_noise(value, sigma_n, random_state):
        if sigma_n == 0 or random_state is None:
            return value
        return int(round(random_state.normal(loc=value, scale=sqrt(sigma_n))))
