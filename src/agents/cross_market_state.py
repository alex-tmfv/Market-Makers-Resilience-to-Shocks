class CrossMarketState:
    """Общий объект, через который два UnifiedMM (на A_USD и B_EUR) видят
    инвентарь друг друга и текущий FX-курс. Нужен Cross-Market AS и GLFT
    для расчёта combined exposure."""
    def __init__(self, fx_callback):
        # fx_callback() возвращает FX-mid в pips (1100 = $1.10/EUR).
        self.fx_callback = fx_callback
        self._agents = {}

    def register(self, symbol, agent):
        self._agents[symbol] = agent

    def get_inventory(self, symbol):
        ag = self._agents.get(symbol)
        return 0 if ag is None else ag.holdings.get(symbol, 0)

    def fx_rate(self):
        return self.fx_callback()
