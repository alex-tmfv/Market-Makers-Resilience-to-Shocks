"""Стратегии торговли для маркет-мейкера"""

import math
from abc import ABC, abstractmethod
from collections import deque

import numpy as np


class MMStrategy(ABC):
    """Базовый интерфейс"""

    @abstractmethod
    def compute_quotes(self, *, mid, bid, ask, inventory, time_to_close,
                       current_time=None, equity=None, shared_state=None):
        raise NotImplementedError


# ============================================================================
# 1. FixedSpread (baseline)
# ============================================================================

class FixedSpreadStrategy(MMStrategy):
    """BaselineMM: симметричный фикс-spread без inventory mgmt."""
    def __init__(self, half_spread, order_size):
        self.half_spread = int(half_spread)
        self.order_size = int(order_size)

    def compute_quotes(self, *, mid, bid, ask, inventory, time_to_close,
                       current_time=None, equity=None, shared_state=None):
        return [
            (int(round(mid - self.half_spread)), self.order_size, True),
            (int(round(mid + self.half_spread)), self.order_size, False),
        ]


# ============================================================================
# 2. Cross-Market AvellanedaStoikov (Bergault, Evangelista, Guéant, Sansavini 2021)
# ============================================================================

class CrossMarketAvellanedaStoikovStrategy(MMStrategy):
    """Cross-Market AS: обобщение AS-2008 на два
    коррелированных актива через combined inventory q_A + c·q_B"""

    SYM_A = 'A_USD'
    SYM_B = 'B_EUR'

    def __init__(self, gamma, sigma_a, k, order_size, symbol):
        self.gamma = float(gamma)
        self.sigma_a = float(sigma_a)
        self.k = float(k)
        self.order_size = int(order_size)
        if symbol not in (self.SYM_A, self.SYM_B):
            raise ValueError(f"symbol must be {self.SYM_A!r} or {self.SYM_B!r}, got {symbol!r}")
        self.symbol = symbol

    def compute_quotes(self, *, mid, bid, ask, inventory, time_to_close,
                       current_time=None, equity=None, shared_state=None):
        if time_to_close <= 0:
            return []

        # Общиай инвентарь
        if shared_state is not None:
            q_a = shared_state.get_inventory(self.SYM_A)
            q_b = shared_state.get_inventory(self.SYM_B)
            fx = shared_state.fx_rate()
            c = 1000.0 / fx
        else:
            q_a = inventory if self.symbol == self.SYM_A else 0
            q_b = inventory if self.symbol == self.SYM_B else 0
            c = 1.0

        sigma_a2 = self.sigma_a ** 2

        # Skew_i = γ · (Σq)_i · (T-t); для A в USD-cents, для B в EUR-cents.
        skew_in_a = self.gamma * sigma_a2 * (q_a + c * q_b) * time_to_close
        if self.symbol == self.SYM_A:
            inv_skew = skew_in_a
            local_sigma2 = sigma_a2
        else:  # B_EUR
            inv_skew = c * skew_in_a
            local_sigma2 = (c * self.sigma_a) ** 2

        r = mid - inv_skew
        spread = (
            self.gamma * local_sigma2 * time_to_close
            + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)
        )
        half = spread / 2.0
        bid_price = int(round(r - half))
        ask_price = int(round(r + half))
        if bid_price >= ask_price:
            return []
        return [
            (bid_price, self.order_size, True),
            (ask_price, self.order_size, False),
        ]


# ============================================================================
# 2b. GLFT (Guéant-Lehalle-Fernandez-Tapia 2013) — single-asset, stationary
# ============================================================================

class GLFTStrategy(MMStrategy):
    """GLFT. Сохранён для исторического сравнения, в работе используется Cross-Market вариант ниже."""

    def __init__(self, gamma, sigma, k, A, order_size):
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.k = float(k)
        self.A = float(A)
        self.order_size = int(order_size)

        self.psi = math.log(1.0 + gamma / k) / gamma
        factor = (1.0 + gamma / k) ** (1.0 + k / gamma)
        self.eta = math.sqrt(sigma ** 2 * gamma / (2.0 * A * k) * factor)

    def compute_quotes(self, *, mid, bid, ask, inventory, time_to_close,
                       current_time=None, equity=None, shared_state=None):
        q = inventory
        r = mid - self.eta * q
        half_spread = self.psi + self.eta / 2.0

        bid_price = int(round(r - half_spread))
        ask_price = int(round(r + half_spread))
        if bid_price >= ask_price:
            return []
        return [
            (bid_price, self.order_size, True),
            (ask_price, self.order_size, False),
        ]


# ============================================================================
# 2c. Cross-Market GLFT — GLFT + cross-market coupling, без (T-t) scaling
# ============================================================================

class CrossMarketGLFTStrategy(MMStrategy):
    """GLFT + cross-market inventory coupling"""

    SYM_A = 'A_USD'
    SYM_B = 'B_EUR'

    def __init__(self, gamma, sigma_a, k, A, order_size, symbol):
        self.gamma = float(gamma)
        self.sigma_a = float(sigma_a)
        self.k = float(k)
        self.A_param = float(A)
        self.order_size = int(order_size)
        if symbol not in (self.SYM_A, self.SYM_B):
            raise ValueError(f"symbol must be {self.SYM_A!r} or {self.SYM_B!r}, got {symbol!r}")
        self.symbol = symbol

        self.psi = math.log(1.0 + gamma / k) / gamma
        factor = (1.0 + gamma / k) ** (1.0 + k / gamma)
        self.eta_a = math.sqrt(sigma_a ** 2 * gamma / (2.0 * A * k) * factor)

    def compute_quotes(self, *, mid, bid, ask, inventory, time_to_close,
                       current_time=None, equity=None, shared_state=None):
        if shared_state is not None:
            q_a = shared_state.get_inventory(self.SYM_A)
            q_b = shared_state.get_inventory(self.SYM_B)
            c = 1000.0 / shared_state.fx_rate()
        else:
            q_a = inventory if self.symbol == self.SYM_A else 0
            q_b = inventory if self.symbol == self.SYM_B else 0
            c = 1.0

        combined = q_a + c * q_b   # в A-эквиваленте

        if self.symbol == self.SYM_A:
            inv_skew  = self.eta_a * combined        # USD-cents
            eta_local = self.eta_a
        else:  # B_EUR
            inv_skew  = c * self.eta_a * combined    # EUR-cents
            eta_local = c * self.eta_a               # η_B = c·η_A, поскольку σ_B = c·σ_A

        r = mid - inv_skew
        half_spread = self.psi + eta_local / 2.0

        bid_price = int(round(r - half_spread))
        ask_price = int(round(r + half_spread))
        if bid_price >= ask_price:
            return []
        return [
            (bid_price, self.order_size, True),
            (ask_price, self.order_size, False),
        ]


# ============================================================================
# 3. FixedLadder (Chakraborty-Kearns)
# ============================================================================

class FixedLadderStrategy(MMStrategy):
    """Ladder Chakraborty-Kearns"""
    def __init__(self, window_size, num_ticks, order_size, anchor='middle'):
        self.window_size = int(window_size)
        self.num_ticks = int(num_ticks)
        self.order_size = int(order_size)
        if anchor not in ('top', 'middle', 'bottom'):
            raise ValueError(f"anchor must be top/middle/bottom, got {anchor!r}")
        self.anchor = anchor

    def compute_quotes(self, *, mid, bid, ask, inventory, time_to_close,
                       current_time=None, equity=None, shared_state=None):
        if self.anchor == 'middle':
            highest_bid = int(mid - self.window_size // 2)
            lowest_ask = int(mid + (self.window_size + 1) // 2)
        elif self.anchor == 'bottom':
            highest_bid = int(mid - 1)
            lowest_ask = int(mid + self.window_size)
        else:  # 'top'
            highest_bid = int(mid - self.window_size)
            lowest_ask = int(mid + 1)

        orders = []
        for i in range(self.num_ticks):
            orders.append((highest_bid - i, self.order_size, True))
            orders.append((lowest_ask + i, self.order_size, False))
        return orders


# ============================================================================
# 4. SizeSkewLadder (Adaptive с sigmoid skew по инвентарю)
# ============================================================================

class SizeSkewLadderStrategy(MMStrategy):
    """AdaptiveMM: ladder с inventory-skew через размеры ордеров"""
    def __init__(self, window_size, num_ticks, base_size, skew_beta, anchor='middle'):
        self.window_size = int(window_size)
        self.num_ticks = int(num_ticks)
        self.base_size = int(base_size)
        self.skew_beta = float(skew_beta)
        if anchor not in ('top', 'middle', 'bottom'):
            raise ValueError(f"anchor must be top/middle/bottom, got {anchor!r}")
        self.anchor = anchor

    def compute_quotes(self, *, mid, bid, ask, inventory, time_to_close,
                       current_time=None, equity=None, shared_state=None):
        x = inventory * self.skew_beta
        if x > 50:
            prop_sell = 1.0
        elif x < -50:
            prop_sell = 0.0
        else:
            prop_sell = 1.0 / (1.0 + math.exp(-x))

        sell_size = max(1, int(round(2 * prop_sell * self.base_size)))
        buy_size = max(1, int(round(2 * (1 - prop_sell) * self.base_size)))

        if self.anchor == 'middle':
            highest_bid = int(mid - self.window_size // 2)
            lowest_ask = int(mid + (self.window_size + 1) // 2)
        elif self.anchor == 'bottom':
            highest_bid = int(mid - 1)
            lowest_ask = int(mid + self.window_size)
        else:
            highest_bid = int(mid - self.window_size)
            lowest_ask = int(mid + 1)

        orders = []
        for i in range(self.num_ticks):
            orders.append((highest_bid - i, buy_size, True))
            orders.append((lowest_ask + i, sell_size, False))
        return orders


# ============================================================================
# 5. RLTunedGLFT — GLFT с RL-policy, выбирающей γ из дискретного набора
# ============================================================================

class RLTunedGLFTStrategy(MMStrategy):
    """GLFT с динамически выбираемой γ от RL-policy"""

    STATE_FEATURES = [
        "inventory_norm",       # q / mm_capacity
        "time_to_close_norm",   # (T-t) / T_total
        "mid_drift_60s",        # (mid_now - mid_60s) / 100 (USD)
        "net_fill_60s",         # (q_now - q_60s) / 200
        "spread_rel",           # (ask - bid) / mid
        "realized_vol_60s",     # std(diff(mid)) over 60s / 10
        "mid_drift_10s",        # (mid_now - mid_10s) / 100
        "mispricing",           # (mid - r_company) / 100
    ]
    STATE_DIM = len(STATE_FEATURES)

    def __init__(
        self,
        gamma_set,
        sigma,
        k,
        A,
        order_size,
        oracle=None,
        symbol=None,
        mm_capacity=1000,
        policy=None,
        history_window_sec=60.0,
    ):
        self.gamma_set = [float(g) for g in gamma_set]
        if not self.gamma_set:
            raise ValueError("gamma_set must be non-empty")
        self.sigma = float(sigma)
        self.k = float(k)
        self.A = float(A)
        self.order_size = int(order_size)
        self.oracle = oracle
        self.symbol = symbol
        self.mm_capacity = float(mm_capacity)
        self.policy = policy
        self.history_window_sec = float(history_window_sec)

        self._rolling = deque()
        self._T_total = None
        self.history_log = []

        self._oracle_rs = np.random.RandomState(0)

    def reset_history(self):
        """Сбросить trajectory log и rolling buffer в начале нового эпизода."""
        self.history_log = []
        self._rolling.clear()
        self._T_total = None

    def get_history(self):
        return self.history_log

    def set_policy(self, policy_fn):
        """policy_fn(state: np.ndarray[STATE_DIM]) -> int (action index in 0..n_actions-1)."""
        self.policy = policy_fn

    @property
    def n_actions(self):
        return len(self.gamma_set)


    def _lookup_past(self, target_ttc, mid_now, inv_now):
        """Ближайшая запись с `ttc ≥ target_ttc`, либо (mid_now, inv_now)."""
        for ttc, mid_h, inv_h in self._rolling:
            if ttc >= target_ttc:
                return mid_h, inv_h
        return mid_now, inv_now

    def _push_rolling(self, ttc, mid, inventory):
        self._rolling.append((ttc, mid, inventory))
        cutoff = ttc + self.history_window_sec * 1.5
        while self._rolling and self._rolling[0][0] > cutoff:
            self._rolling.popleft()

    def _build_state(self, *, mid, bid, ask, inventory, time_to_close, current_time):
        if self._T_total is None or time_to_close > self._T_total:
            self._T_total = max(time_to_close, 1.0)

        inv_norm   = inventory / self.mm_capacity
        ttc_norm   = max(0.0, min(1.0, time_to_close / self._T_total))
        m60, i60   = self._lookup_past(time_to_close + 60.0, mid, inventory)
        m10, _     = self._lookup_past(time_to_close + 10.0, mid, inventory)
        drift60    = (mid - m60) / 100.0
        drift10    = (mid - m10) / 100.0
        net_fill   = (inventory - i60) / 200.0
        spread_rel = (ask - bid) / max(mid, 1.0)

        recent_mids = [mh for ttc, mh, _ in self._rolling if ttc <= time_to_close + 60.0]
        vol = float(np.std(np.diff(recent_mids))) / 10.0 if len(recent_mids) >= 2 else 0.0

        # mispricing = (mid - true fundamental) / 100; нужен oracle и current_time.
        mispricing = 0.0
        if self.oracle is not None and current_time is not None and self.symbol is not None:
            try:
                fund = float(self.oracle.observePrice(
                    self.symbol, current_time, sigma_n=0, random_state=self._oracle_rs,
                ))
                mispricing = (mid - fund) / 100.0
            except Exception:
                pass

        return np.array([
            inv_norm, ttc_norm, drift60, net_fill,
            spread_rel, vol, drift10, mispricing,
        ], dtype=np.float32)

    def _glft_constants(self, gamma):
        psi = math.log(1.0 + gamma / self.k) / gamma
        factor = (1.0 + gamma / self.k) ** (1.0 + self.k / gamma)
        eta = math.sqrt(self.sigma ** 2 * gamma / (2.0 * self.A * self.k) * factor)
        return psi, eta

    def compute_quotes(self, *, mid, bid, ask, inventory, time_to_close,
                       current_time=None, equity=None, shared_state=None):
        state = self._build_state(
            mid=mid, bid=bid, ask=ask, inventory=inventory,
            time_to_close=time_to_close, current_time=current_time,
        )

        if self.policy is not None:
            action = int(self.policy(state))
        else:
            action = 0
        action = max(0, min(action, len(self.gamma_set) - 1))
        gamma = self.gamma_set[action]

        psi, eta = self._glft_constants(gamma)

        q = inventory
        r = mid - eta * q
        half = psi + eta / 2.0
        bid_price = int(round(r - half))
        ask_price = int(round(r + half))

        self.history_log.append({
            "state": state,
            "action": action,
            "gamma": gamma,
            "mid": float(mid),
            "inventory": int(inventory),
            "time_to_close": float(time_to_close),
            "equity": float(equity) if equity is not None else None,
        })
        self._push_rolling(time_to_close, mid, inventory)

        if bid_price >= ask_price:
            return []
        return [
            (bid_price, self.order_size, True),
            (ask_price, self.order_size, False),
        ]