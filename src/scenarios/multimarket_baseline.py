"""Главный сценарий: ExchangeAgent с тремя стаканами (A_USD, B_EUR, FX),
MultiMarketOracle с OU-процессом для r_company, набор фоновых агентов
(Noise/Value/ZI/Momentum + FxArbAgent) и MM-агент с выбираемой через
`--mm-type` стратегией. Шоки и параметры задаются CLI.

Запуск:
    cd experiments
    PYTHONPATH=../vendor/abides:.. python ../src/scenarios/multimarket_baseline.py \\
        --mm-type glft --log-dir my_run"""

import argparse
import numpy as np
import pandas as pd

from Kernel import Kernel
from agent.ExchangeAgent import ExchangeAgent
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from agent.ZeroIntelligenceAgent import ZeroIntelligenceAgent
from agent.examples.MomentumAgent import MomentumAgent
from util import util

from src.multimarket.oracle import MultiMarketOracle
from src.agents.fx_arb import FxArbAgent
from src.agents.fx_shock_agent import FxShockAgent
from src.agents.liquidity_shock_agent import LiquidityShockAgent
from src.agents.pressure_agent import OneSidedPressureAgent

# UnifiedMM + plug-in стратегия — все типы MM построены поверх одного класса.
from src.agents.unified_mm import UnifiedMM
from src.agents.mm_strategies import (
    FixedSpreadStrategy,
    CrossMarketAvellanedaStoikovStrategy,
    GLFTStrategy,
    CrossMarketGLFTStrategy,
    FixedLadderStrategy,
    SizeSkewLadderStrategy,
    RLTunedGLFTStrategy,
)
from src.agents.cross_market_state import CrossMarketState


# --- Базовые константы рыночной среды ---

SYM_A  = "A_USD"
SYM_B  = "B_EUR"
SYM_FX = "FX"

# Уменьшенная шкала акций (×1/100 от типичной) → notional A/B сопоставим с FX.
R_BAR = 1_000               # r_company = $10.00 (USD-cents)
# FX в pips/EUR: tick = 0.09% mid, близко к реальной FX-микроструктуре.
INITIAL_FX = 1100           # = $1.1000/EUR


def make_fx_mid_callback(exchange_agent):
    # FX-mid с fallback'ом: (bid+ask)//2 → last_trade → INITIAL_FX.
    def _get_fx_mid():
        ob = exchange_agent.order_books[SYM_FX]
        best_bid = ob.bids[0][0].limit_price if ob.bids else None
        best_ask = ob.asks[0][0].limit_price if ob.asks else None
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) // 2
        if ob.last_trade is not None:
            return ob.last_trade
        return INITIAL_FX
    return _get_fx_mid


def fresh_random_state():
    # Независимый RandomState из глобального np.random (должен быть seed'нут заранее).
    return np.random.RandomState(seed=np.random.randint(0, 2**32, dtype="uint64"))


def _parse_shock_time(raw, mkt_open, mkt_close, margin_minutes=10):
    """`'HH:MM'` или `'random'` → pd.Timestamp. Для `'random'` сэмпл uniform
    в `[mkt_open + margin, mkt_close - margin]` (через глобальный np.random,
    предварительно засидированный)."""
    if raw == "random":
        total = (mkt_close - mkt_open).total_seconds()
        m = margin_minutes * 60
        offset = np.random.uniform(m, total - m)
        return mkt_open + pd.Timedelta(seconds=float(offset))
    hh, mm = raw.split(":")
    return mkt_open.normalize() + pd.Timedelta(f"{int(hh)}:{int(mm)}:00")


def _parse_duration(raw):
    """`'5min'` / `'30s'` → `pd.Timedelta`."""
    return pd.to_timedelta(raw)


# 7 log-uniform значений в диапазоне [1e-8, 1e-5] — action space DQN'а.
DEFAULT_RL_GAMMA_SET = (1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5)


def _parse_gamma_set(raw):
    """`'1e-8,5e-8,...'` → `tuple[float]`; пусто / None → `DEFAULT_RL_GAMMA_SET`."""
    if not raw:
        return tuple(DEFAULT_RL_GAMMA_SET)
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def build(seed, num_noise, num_value, num_momentum, num_zi,
          value_lambda_a_override, value_sigma_n_override,
          mm_wake_freq_override,
          num_noise_fx, num_zi_fx, num_momentum_fx,
          arb_order_size, arb_open_pct, arb_close_pct,
          arb_wake_freq, arb_starting_inventory, arb_max_exposure,
          mm_starting_cash, arb_starting_cash, maker_rebate,
          mm_type, as_gamma, as_k, as_sigma_a, as_sigma_b,
          glft_gamma, glft_k, glft_A, glft_sigma,
          rl_gamma_set, rl_sigma, rl_k, rl_A, rl_policy_path,
          sb_order_size, sb_window_size, sb_num_ticks, sb_anchor,
          adp_min_order_size, adp_window_size, adp_num_ticks,
          adp_anchor, adp_skew_beta,
          fast_mode,
          stress_multiplier, megashock_mean, megashock_var,
          fx_shock_specs, news_shock_specs, drift_shock_specs,
          liquidity_shock_specs, pressure_shock_specs):
    # Возвращает (agents, oracle, kernel_start, mkt_close, kernel_seed).
    # Shock-spec'и — raw-строки из CLI; парсятся уже после того, как
    # известны mkt_open/mkt_close (для 'random'-времён).
    np.random.seed(seed)

    # kernel_seed фиксируется СРАЗУ после np.random.seed: иначе любое изменение
    # числа агентов или 'random'-шоков сдвигает np.random sequence, kernel
    # получает другой seed, и latency-noise отличается с t=0 → траектории
    # разъезжаются ещё до момента «выстрела» шока.
    kernel_seed = int(np.random.randint(0, 2**32, dtype="uint64"))

    historical_date = pd.to_datetime("2024-01-15")
    mkt_open  = historical_date + pd.to_timedelta("09:30:00")
    mkt_close = historical_date + pd.to_timedelta("10:30:00")

    # --- Парсинг шоков (использует mkt_open/mkt_close для 'random'-времён) ---

    # news_shocks: [(t, magnitude_cents)]
    parsed_news_shocks = []
    for raw in news_shock_specs:
        # формат: "HH:MM:+300" или "random:+300" (знак обязателен)
        parts = raw.split(":")
        if parts[0] == "random":
            t = _parse_shock_time("random", mkt_open, mkt_close)
            mag = int(parts[1])
        else:
            t = _parse_shock_time(f"{parts[0]}:{parts[1]}", mkt_open, mkt_close)
            mag = int(parts[2])
        parsed_news_shocks.append((t, mag))

    # drift_shocks: непрерывный тренд дискретизируется в серию мелких jumps.
    # Формат: "HH:MM-HH:MM:±N" (окно + amplitude) или "random:5min:±N".
    parsed_drift_jumps = []
    drift_granularity = pd.Timedelta("5s")
    for raw in drift_shock_specs:
        parts = raw.split(":")
        if parts[0] == "random":
            duration = _parse_duration(parts[1])
            mag = int(parts[2])
            margin_secs = 10 * 60
            total = (mkt_close - mkt_open).total_seconds()
            free = total - margin_secs * 2 - duration.total_seconds()
            if free <= 0:
                raise ValueError(f"drift-shock duration {duration} too long")
            offset = np.random.uniform(margin_secs, margin_secs + free)
            t_start = mkt_open + pd.Timedelta(seconds=float(offset))
            t_end = t_start + duration
        else:
            # split → ['HH','MM-HH','MM','±N'] → time-window: parts[:3]
            time_window = ":".join(parts[:3])
            t_start_str, t_end_str = time_window.split("-")
            t_start = _parse_shock_time(t_start_str, mkt_open, mkt_close)
            t_end   = _parse_shock_time(t_end_str,   mkt_open, mkt_close)
            mag = int(parts[3])
        n_steps = max(int((t_end - t_start) / drift_granularity), 1)
        per_step = mag / n_steps
        for i in range(n_steps):
            parsed_drift_jumps.append(
                (t_start + drift_granularity * i, int(round(per_step)))
            )

    manual_shocks_for_oracle = parsed_news_shocks + parsed_drift_jumps

    # fx_shocks: [{time, size, is_buy}]
    parsed_fx_shocks = []
    for raw in fx_shock_specs:
        # формат: "HH:MM:size:direction" или "random:size:direction"
        parts = raw.split(":")
        if parts[0] == "random":
            t = _parse_shock_time("random", mkt_open, mkt_close)
            size, direction = parts[1], parts[2]
        else:
            t = _parse_shock_time(f"{parts[0]}:{parts[1]}", mkt_open, mkt_close)
            size, direction = parts[2], parts[3]
        if direction not in ("buy", "sell"):
            raise ValueError(f"fx-shock direction must be buy|sell, got {direction!r}")
        parsed_fx_shocks.append({
            "time": t, "size": int(size), "is_buy": direction == "buy"
        })

    # liquidity_shocks: [{symbol, time, side}]
    parsed_liquidity_shocks = []
    for raw in liquidity_shock_specs:
        # формат: "symbol:HH:MM:side" или "symbol:random:side"
        parts = raw.split(":")
        symbol = parts[0]
        if symbol not in (SYM_A, SYM_B, SYM_FX):
            raise ValueError(f"liquidity-shock symbol must be A_USD|B_EUR|FX, got {symbol!r}")
        if parts[1] == "random":
            t = _parse_shock_time("random", mkt_open, mkt_close)
            side = parts[2]
        else:
            t = _parse_shock_time(f"{parts[1]}:{parts[2]}", mkt_open, mkt_close)
            side = parts[3]
        if side not in ("bid", "ask"):
            raise ValueError(f"liquidity-shock side must be bid|ask, got {side!r}")
        parsed_liquidity_shocks.append({"symbol": symbol, "time": t, "side": side})

    # pressure-shocks: "symbol:HH:MM-HH:MM:side:orders_per_wake:order_size:wake_freq".
    # Пример: "A_USD:10:00-10:15:ask:50:5:2s" — каждые 2 сек 50 sell-orders × 5 шт.
    parsed_pressure_shocks = []
    for raw in pressure_shock_specs:
        parts = raw.split(":")
        symbol = parts[0]
        if symbol not in (SYM_A, SYM_B, SYM_FX):
            raise ValueError(f"pressure-shock symbol must be A_USD|B_EUR|FX, got {symbol!r}")
        # symbol + ['HH','MM-HH','MM'] → time-window parts[1:4]
        time_window = ":".join(parts[1:4])
        t_start_str, t_end_str = time_window.split("-")
        t_start = _parse_shock_time(t_start_str, mkt_open, mkt_close)
        t_end   = _parse_shock_time(t_end_str,   mkt_open, mkt_close)
        side = parts[4]
        if side not in ("bid", "ask"):
            raise ValueError(f"pressure-shock side must be bid|ask, got {side!r}")
        orders_per_wake = int(parts[5])
        order_size = int(parts[6])
        wake_freq = parts[7]
        parsed_pressure_shocks.append({
            "symbol": symbol, "start": t_start, "end": t_end, "side": side,
            "orders_per_wake": orders_per_wake, "order_size": order_size,
            "wake_freq": wake_freq,
        })

    # --- 1. ExchangeAgent (создаём первым: oracle получает callback к нему) ---
    agents = []
    agent_id = 0

    # В `--fast-mode` отключены тяжёлые логи exchange'а (per-trade ORDER_EXECUTED
    # и orderbook-snapshots) — нужны только в analysis.ipynb /
    # scenarios_overview.ipynb. Экономит ~25-30% wall-time на прогон.
    exchange = ExchangeAgent(
        id=agent_id,
        name="EXCHANGE_AGENT",
        type="ExchangeAgent",
        mkt_open=mkt_open,
        mkt_close=mkt_close,
        symbols=[SYM_A, SYM_B, SYM_FX],
        log_orders=(not fast_mode),
        pipeline_delay=0,
        computation_delay=0,
        stream_history=10,
        book_freq=(None if fast_mode else "1s"),
        wide_book=True,
        random_state=fresh_random_state(),
    )
    agents.append(exchange)
    agent_id += 1

    # --- 2. Oracle (callback к FX-стакану) ---
    # Stress-режим: повышаем частоту megashock'ов через `megashock_lambda_a`,
    # амплитуду оставляем — она задаёт физический смысл шока (~0.2-0.5% от r_bar).
    # Baseline λ=2.78e-18 → ~0 шоков/час; stress=1e5 → ~1/час; 1e6 → ~10/час.
    r_company_cfg = {
        "r_bar": R_BAR,
        "kappa": 1.67e-16,
        "sigma_s": 0,
        "fund_vol": 1e-8,
        "megashock_lambda_a": 2.77778e-18 * stress_multiplier,
        "megashock_mean": megashock_mean,
        "megashock_var": megashock_var,
        "random_state": fresh_random_state(),
    }
    oracle = MultiMarketOracle(
        mkt_open=mkt_open,
        mkt_close=mkt_close,
        r_company_cfg=r_company_cfg,
        get_fx_mid=make_fx_mid_callback(exchange),
        initial_fx_price=INITIAL_FX,
        manual_shocks=manual_shocks_for_oracle,
    )

    # --- 3. Single-market агенты на A_USD и B_EUR ---
    # Стартовый кэш по $100k/€100k на каждого. При цене $10 это даёт ~10k
    # capacity по штукам; реальный лимит задаётся q_max или другой логикой
    # агента, кэш в основном служит буфером для MTM-колебаний.
    starting_cash = 10_000_000   # $100k / €100k в центах
    # Noise-окно = sim-окну: иначе ABIDES U-quadratic wakeup даёт горб в начале
    # и спад в конце; короткая симуляция требует РАВНОМЕРНОГО noise flow.
    noise_open  = mkt_open
    noise_close = mkt_close

    def _uniform_noise_wakeup():
        """UNIFORM (а не U-quadratic из `util.get_wake_time`) во всём окне."""
        span_ns = int((noise_close - noise_open).total_seconds() * 1e9)
        offset_ns = int(np.random.randint(0, span_ns + 1))
        return noise_open + pd.Timedelta(nanoseconds=offset_ns)

    # ValueAgent: lambda_a=1e-10 (~10 сек) и sigma_n=R_BAR/100 (низкий obs-noise).
    # CLI-overrides (`--value-lambda-a`, `--value-sigma-n`) позволяют усилить
    # informed flow — нужно для штрафа MM-стратегий за тесные котировки.
    value_kappa    = 1.67e-15
    value_sigma_n  = value_sigma_n_override  if value_sigma_n_override  is not None else (R_BAR // 100)
    value_lambda_a = value_lambda_a_override if value_lambda_a_override is not None else 1e-10
    value_sigma_s  = r_company_cfg["fund_vol"]

    # Prior r_bar: A_USD = r_company; B_EUR = r_company·1000/FX. Без поправки
    # value-агенты B_EUR «учатся» ~20 минут вместо моментального tracking'а.
    r_bar_for = {
        SYM_A: R_BAR,
        SYM_B: int(round(R_BAR * 1000 / INITIAL_FX)),
    }

    # Momentum: крупные ордера + частые wakeups (для заметного объёма в графиках).
    momentum_min_size, momentum_max_size = 20, 50
    momentum_wake_freq = "10s"

    for symbol in (SYM_A, SYM_B):
        for _ in range(num_noise):
            agents.append(NoiseAgent(
                id=agent_id, name=f"NoiseAgent_{symbol}_{agent_id}",
                type="NoiseAgent", symbol=symbol,
                starting_cash=starting_cash,
                wakeup_time=_uniform_noise_wakeup(),
                log_orders=False,
                random_state=fresh_random_state(),
            ))
            agent_id += 1

        for _ in range(num_value):
            agents.append(ValueAgent(
                id=agent_id, name=f"ValueAgent_{symbol}_{agent_id}",
                type="ValueAgent", symbol=symbol,
                starting_cash=starting_cash,
                r_bar=r_bar_for[symbol],
                kappa=value_kappa,
                sigma_s=value_sigma_s,
                sigma_n=value_sigma_n,
                lambda_a=value_lambda_a,
                log_orders=False,
                random_state=fresh_random_state(),
            ))
            agent_id += 1

        # ZI добавляют ликвидность (ставят лимиты даже в пустой стакан, в
        # отличие от NoiseAgent) и тянут цену к fundamental. q_max=600 —
        # большой запас, чтобы ZI не упирались в лимит и оставались активными
        # всю сессию.
        for _ in range(num_zi):
            agents.append(ZeroIntelligenceAgent(
                id=agent_id, name=f"ZIAgent_{symbol}_{agent_id}",
                type="ZeroIntelligenceAgent", symbol=symbol,
                starting_cash=starting_cash,
                sigma_n=value_sigma_n,
                r_bar=r_bar_for[symbol],
                kappa=value_kappa,
                sigma_s=value_sigma_s,
                # ZI на A/B: q_max=600 даёт длину theta-массива 1200 — ZI медленно
                # «выдыхается», активен почти всю симуляцию. sigma_pv=2M, R_min=20
                # — компромисс между объёмом и плавностью activity-distribution.
                q_max=600,
                sigma_pv=2_000_000,
                R_min=20, R_max=250,
                eta=1.0,
                lambda_a=7e-11,
                log_orders=False,
                random_state=fresh_random_state(),
            ))
            agent_id += 1

        for _ in range(num_momentum):
            agents.append(MomentumAgent(
                id=agent_id, name=f"MomentumAgent_{symbol}_{agent_id}",
                type="MomentumAgent", symbol=symbol,
                starting_cash=starting_cash,
                min_size=momentum_min_size, max_size=momentum_max_size,
                wake_up_freq=momentum_wake_freq,
                log_orders=False,
                random_state=fresh_random_state(),
            ))
            agent_id += 1

    # --- 4. FX-рынок: NoiseAgent + ZI + Momentum (без ValueAgent — у FX
    # нет собственного fundamental'а; см. oracle.py). FX делается самым
    # ликвидным: одна арб-нога = ~9000 шт, иначе P_FX «замерзает».
    for _ in range(num_noise_fx):
        agents.append(NoiseAgent(
            id=agent_id, name=f"NoiseAgent_{SYM_FX}_{agent_id}",
            type="NoiseAgent", symbol=SYM_FX,
            starting_cash=starting_cash,
            wakeup_time=_uniform_noise_wakeup(),
            log_orders=False,
            random_state=fresh_random_state(),
        ))
        agent_id += 1

    # ZI на FX: реалистичный bid-ask 2-10 pips (0.02-0.1%). sigma_pv=20k →
    # theta ≈ ±140 pips (±1.3% от 1100); R_max=50 → markup ≤ 0.05%.
    # q_max=200: запас под крупные арб-ноги, иначе IndexError в kernelStopping.
    for _ in range(num_zi_fx):
        agents.append(ZeroIntelligenceAgent(
            id=agent_id, name=f"ZIAgent_{SYM_FX}_{agent_id}",
            type="ZeroIntelligenceAgent", symbol=SYM_FX,
            starting_cash=starting_cash,
            sigma_n=100,
            r_bar=INITIAL_FX,
            kappa=0.05,
            sigma_s=100,
            q_max=200,
            sigma_pv=20_000,
            R_min=0, R_max=50,
            eta=1.0,
            lambda_a=7e-11,
            log_orders=False,
            random_state=fresh_random_state(),
        ))
        agent_id += 1

    # MomentumAgent на FX — даёт динамику микро-трендов и заметный объём.
    # Параметры синхронизированы с моментум на A/B (раньше были крошечные —
    # не были видны на volume-графиках).
    for _ in range(num_momentum_fx):
        agents.append(MomentumAgent(
            id=agent_id, name=f"MomentumAgent_{SYM_FX}_{agent_id}",
            type="MomentumAgent", symbol=SYM_FX,
            starting_cash=starting_cash,
            min_size=momentum_min_size, max_size=momentum_max_size,
            wake_up_freq=momentum_wake_freq,
            log_orders=False,
            random_state=fresh_random_state(),
        ))
        agent_id += 1

    # --- 5. Multi-currency агенты (UnifiedMM + FxArbAgent) ---
    # starting_asset_prices необходимы для корректного USD-eq расчёта
    # starting_cash в `MultiCurrencyTradingAgent.__init__`.
    initial_p_b = int(round(R_BAR * 1000 / INITIAL_FX))
    mc_kwargs_common = dict(
        starting_cash_usd=starting_cash,
        starting_cash_eur=starting_cash,
        initial_fx_price=INITIAL_FX,
        get_fx_mid=make_fx_mid_callback(exchange),
        starting_asset_prices={SYM_A: R_BAR, SYM_B: initial_p_b},
    )
    # MM получает повышенный starting_cash (двусторонняя котировка должна
    # переживать inventory swings без банкротства) и опциональный maker rebate.
    mc_kwargs_mm = dict(mc_kwargs_common)
    mc_kwargs_mm["starting_cash_usd"] = mm_starting_cash
    mc_kwargs_mm["starting_cash_eur"] = mm_starting_cash
    mc_kwargs_mm["maker_rebate_cents_per_share"] = float(maker_rebate)
    # FxArbAgent — отдельный starting_cash, rebate не применяется (это taker).
    mc_kwargs_arb = dict(mc_kwargs_common)
    mc_kwargs_arb["starting_cash_usd"] = arb_starting_cash
    mc_kwargs_arb["starting_cash_eur"] = arb_starting_cash

    # Order size и wake-freq одинаковые для всех MM-типов — apples-to-apples
    # сравнение, в котором отличие сводится только к логике котировок.
    mm_order_size = 50
    mm_wake_freq = pd.Timedelta(mm_wake_freq_override) if mm_wake_freq_override else pd.Timedelta("5s")

    # Cross-market shared state — для AS и GLFT (обе видят combined inventory).
    mm_shared_state = None
    if mm_type in ("as", "glft"):
        mm_shared_state = CrossMarketState(fx_callback=make_fx_mid_callback(exchange))

    def make_mm(symbol, name_suffix, sigma_val, half_spread_baseline):
        if mm_type == "baseline":
            strategy = FixedSpreadStrategy(
                half_spread=half_spread_baseline,
                order_size=mm_order_size,
            )
            mm_name = f"BaselineMM_{name_suffix}"
            mm_type_str = "BaselineMM"
        elif mm_type == "as":
            # σ_A передаётся обоим инстансам; σ_B = c·σ_A вычисляется в стратегии.
            strategy = CrossMarketAvellanedaStoikovStrategy(
                gamma=as_gamma, sigma_a=sigma_val, k=as_k,
                order_size=mm_order_size, symbol=symbol,
            )
            mm_name = f"AvellanedaStoikovMM_{name_suffix}"
            mm_type_str = "AvellanedaStoikovMM"
        elif mm_type == "glft":
            # Cross-Market GLFT: stationary, но `skew = η·(q_A + c·q_B)` как у AS.
            # Single-asset GLFT (`GLFTStrategy`) сохранён в `mm_strategies.py`
            # для исторических сравнений; фабрикой не используется.
            strategy = CrossMarketGLFTStrategy(
                gamma=glft_gamma, sigma_a=glft_sigma, k=glft_k, A=glft_A,
                order_size=mm_order_size, symbol=symbol,
            )
            mm_name = f"GLFTMM_{name_suffix}"
            mm_type_str = "GLFTMM"
        elif mm_type == "rl_glft":
            # Policy подгружается через rl_policy_path; без неё fallback на
            # gamma_set[0] (нужно для инициализации до старта тренировки).
            strategy = RLTunedGLFTStrategy(
                gamma_set=rl_gamma_set,
                sigma=rl_sigma, k=rl_k, A=rl_A,
                order_size=mm_order_size,
                oracle=oracle, symbol=symbol,
                mm_capacity=1000.0,
                policy=None,  # train-script подсунет set_policy() ПОСЛЕ build()
            )
            if rl_policy_path:
                from src.rl.dqn import load_policy_for_inference
                strategy.set_policy(load_policy_for_inference(rl_policy_path))
            mm_name = f"RLGLFTMM_{name_suffix}"
            mm_type_str = "RLGLFTMM"
        elif mm_type == "spread_based":
            strategy = FixedLadderStrategy(
                window_size=int(sb_window_size),
                num_ticks=sb_num_ticks,
                order_size=sb_order_size,
                anchor=sb_anchor,
            )
            mm_name = f"SpreadBasedMM_{name_suffix}"
            mm_type_str = "SpreadBasedMM"
        elif mm_type == "adaptive":
            ws = int(adp_window_size) if adp_window_size != "adaptive" else 2
            strategy = SizeSkewLadderStrategy(
                window_size=ws,
                num_ticks=adp_num_ticks,
                base_size=adp_min_order_size,
                skew_beta=adp_skew_beta,
                anchor=adp_anchor,
            )
            mm_name = f"AdaptiveMM_{name_suffix}"
            mm_type_str = "AdaptiveMM"
        else:
            raise ValueError(f"Unknown mm_type: {mm_type!r}")

        return UnifiedMM(
            id=agent_id_holder[0],
            name=mm_name,
            type=mm_type_str,
            symbol=symbol,
            strategy=strategy,
            wake_up_freq=mm_wake_freq,
            mkt_close=mkt_close,
            shared_state=mm_shared_state,
            random_state=fresh_random_state(),
            **mc_kwargs_mm,
        )

    # list-обёртка для agent_id, чтобы инкрементить из замыкания make_mm.
    # Cross-Market AS использует одну σ_A для обоих рынков (σ_B = c·σ_A
    # вычисляется в стратегии); параметр `as_sigma_b` оставлен для совместимости.
    agent_id_holder = [agent_id]
    agents.append(make_mm(SYM_A, SYM_A, sigma_val=as_sigma_a, half_spread_baseline=1))
    agent_id_holder[0] += 1
    agents.append(make_mm(SYM_B, SYM_B, sigma_val=as_sigma_a, half_spread_baseline=1))
    agent_id_holder[0] += 1
    agent_id = agent_id_holder[0]

    # FxArbAgent: hysteresis open/close при |discrepancy/P_B| ≥/≤ порога.
    agents.append(FxArbAgent(
        id=agent_id, name="FxArbAgent",
        type="FxArbAgent",
        open_threshold_pct=arb_open_pct,
        close_threshold_pct=arb_close_pct,
        order_size=arb_order_size,
        wake_up_freq=pd.Timedelta(arb_wake_freq),
        allow_short=True,
        starting_inventory_a=arb_starting_inventory,
        starting_inventory_b=arb_starting_inventory,
        max_exposure_per_side=arb_max_exposure,
        random_state=fresh_random_state(),
        **mc_kwargs_arb,
    ))
    agent_id += 1

    # --- 6. Shock-агенты ---
    for i, sh in enumerate(parsed_fx_shocks):
        # Cash-буфер для buy на size штук: size·INITIAL_FX/10 центов × 10 (запас).
        shock_cash = max(int(sh["size"] * INITIAL_FX / 10 * 10), 100_000_000)
        agents.append(FxShockAgent(
            id=agent_id,
            name=f"FxShockAgent_{i}",
            type="FxShockAgent",
            symbol=SYM_FX,
            shock_time=sh["time"],
            shock_size=sh["size"],
            is_buy=sh["is_buy"],
            starting_cash=shock_cash,
            log_orders=True,
            random_state=fresh_random_state(),
        ))
        agent_id += 1

    # LiquidityShockAgent чистит сторону стакана напрямую через `kernel.agents
    # [exchangeID].order_books`, в обход ABIDES message-протокола. Корректное
    # отзыв заявок «от имени чужого агента» через сеть невозможен (exchange
    # проверяет ownership) — поэтому шок работает на уровне симуляции.
    for i, sh in enumerate(parsed_liquidity_shocks):
        agents.append(LiquidityShockAgent(
            id=agent_id,
            name=f"LiquidityShockAgent_{i}",
            type="LiquidityShockAgent",
            symbol=sh["symbol"],
            shock_time=sh["time"],
            side=sh["side"],
            log_orders=True,
            random_state=fresh_random_state(),
        ))
        agent_id += 1

    # OneSidedPressureAgent: sustained supply/demand pressure через периодические
    # limit-orders. Комбинируется с drift-shock для реалистичного bear market
    # (drift = падение фунда; pressure = навес предложения от informed sellers).
    for i, sh in enumerate(parsed_pressure_shocks):
        agents.append(OneSidedPressureAgent(
            id=agent_id,
            name=f"PressureAgent_{i}_{sh['symbol']}_{sh['side']}",
            type="OneSidedPressureAgent",
            symbol=sh["symbol"],
            start_time=sh["start"],
            end_time=sh["end"],
            side=sh["side"],
            orders_per_wake=sh["orders_per_wake"],
            order_size=sh["order_size"],
            offset_ticks=1,  # хардкод: ставить за best+1 (joining inside)
            wake_freq=sh["wake_freq"],
            log_orders=True,
            random_state=fresh_random_state(),
        ))
        agent_id += 1

    # Re-seed global np.random в конце build: vendor-агенты ABIDES (NoiseAgent,
    # ZI, MomentumAgent, util.get_wake_time) читают глобальное np.random во время
    # симуляции, а не свой self.random_state. Каждый шок-агент потребил по
    # randint → последовательность сдвинута; детерминированный re-seed фиксирует
    # sim-runtime np.random независимо от числа шок-агентов.
    np.random.seed(seed + 1)

    return agents, oracle, historical_date, mkt_close, kernel_seed


def main():
    parser = argparse.ArgumentParser(description="multimarket_baseline scenario")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--num-noise", type=int, default=3000,
                        help="NoiseAgent на A_USD/B_EUR. Каждый делает 1 трейд, "
                             "поэтому нужны сотни для заметного шумового потока.")
    parser.add_argument("--num-value", type=int, default=10,
                        help="число ValueAgent на каждом из A_USD/B_EUR")
    parser.add_argument("--value-lambda-a", type=float, default=None,
                        help="ValueAgent wake rate (1/ns). Default=1e-10 (~10 сек). "
                             "Меньше → реже просыпаются. Больше → агрессивнее "
                             "обновляют квоты, быстрее снимают stale-котировки MM.")
    parser.add_argument("--value-sigma-n", type=float, default=None,
                        help="ValueAgent observation noise (variance of price observations). "
                             "Default=R_BAR/100=10. Меньше → точнее видят fundamental → "
                             "более информированный поток.")
    parser.add_argument("--mm-wake-freq", type=str, default=None,
                        help="Wake frequency для BaselineMM и AS-MM (наши). Default='5s'. "
                             "Больше → MM реже обновляет квоты → его котировки становятся "
                             "stale → быстрые информированные агенты успевают их 'снять' "
                             "после шока → MM накапливает adverse inventory. Нужно для "
                             "реалистичного теста inventory-management алгоритмов.")
    parser.add_argument("--num-momentum", type=int, default=5,
                        help="число MomentumAgent на каждом из A_USD/B_EUR")
    parser.add_argument("--num-zi", type=int, default=50,
                        help="число ZeroIntelligenceAgent на каждом из A_USD/B_EUR")
    parser.add_argument("--num-noise-fx", type=int, default=3000,
                        help="число NoiseAgent на FX (FX — самый ликвидный)")
    parser.add_argument("--num-zi-fx", type=int, default=200,
                        help="число ZeroIntelligenceAgent на FX — даёт глубину стакана. "
                             "Чем больше, тем меньше price-impact арба")
    parser.add_argument("--num-momentum-fx", type=int, default=10,
                        help="число MomentumAgent на FX — даёт видимую динамику")
    parser.add_argument("--arb-order-size", type=int, default=1,
                        help="размер ноги FxArbAgent на A_USD и B_EUR в штуках. "
                             "FX-нога автоматически = arb_order_size*P_B/100. "
                             "При размере 1 арб виден только на FX как поставщик "
                             "арбитражной ликвидности; это и есть реалистичное поведение")
    parser.add_argument("--arb-open-pct", type=float, default=0.3,
                        help="% от P_B, выше которого арб ОТКРЫВАЕТ/расширяет позицию. "
                             "Должен быть > transaction cost (~0.15% за 3-leg). "
                             "Иначе арб гарантированно теряет на spread-costs.")
    parser.add_argument("--arb-close-pct", type=float, default=0.15,
                        help="% от P_B, ниже которого арб ЗАКРЫВАЕТ позицию (hysteresis)")
    parser.add_argument("--arb-wake-freq", type=str, default="10s",
                        help="как часто арб просыпается. Реже = менее агрессивно")
    parser.add_argument("--arb-starting-inventory", type=int, default=0,
                        help="стартовый инвентарь A_USD и B_EUR у арба. "
                             "Default=0: арб стартует только с CASH в двух валютах")
    parser.add_argument("--arb-max-exposure", type=int, default=1000,
                        help="максимальное отклонение позиции от стартовой. "
                             "При start=0, max=1000, allow_short=True: позиция в [-1000, +1000]. "
                             "Скейлировано под цены $10/акция (раньше было 100 при $1000/акция).")
    parser.add_argument("--mm-starting-cash", type=int, default=50_000_000,
                        help="стартовый капитал у каждого BaselineMM (в каждой валюте, центы). "
                             "Default=50M = $500k USD и €500k EUR. Больше чем у обычных агентов, "
                             "т.к. MM держит двустороннюю котировку и переживает inventory swings.")
    parser.add_argument("--arb-starting-cash", type=int, default=20_000_000,
                        help="стартовый капитал у FxArbAgent (в каждой валюте, центы). "
                             "Default=20M = $200k USD и €200k EUR. Больше чем у обычных, "
                             "потому что арб торгует на 3 рынках одновременно.")
    parser.add_argument("--maker-rebate", type=float, default=0.0,
                        help="Maker rebate в cents per share для MM-агентов на КАЖДЫЙ fill "
                             "на A_USD/B_EUR. Default=0 (academic mode, MM зарабатывает только "
                             "spread capture). Реалистичные значения: NASDAQ ~0.2-0.3 cents/share, "
                             "наш сетап с высоким adverse selection требует ~1-2 cents/share для "
                             "положительного PnL.")
    # --- MM-стратегия ---
    parser.add_argument("--mm-type", choices=["baseline", "as", "glft", "rl_glft", "spread_based", "adaptive"],
                        default="baseline",
                        help="Тип маркет-мейкера. Все типы используют UnifiedMM, "
                             "отличаются только стратегией (см. src/agents/mm_strategies.py): "
                             "baseline (FixedSpreadStrategy), "
                             "as (CrossMarketAvellanedaStoikovStrategy), "
                             "glft (CrossMarketGLFTStrategy — Cross-Market GLFT, stationary, "
                             "после ablation шаг 22 это default GLFT), "
                             "rl_glft (RLTunedGLFTStrategy — γ выбирается DQN-policy), "
                             "spread_based (FixedLadderStrategy), "
                             "adaptive (SizeSkewLadderStrategy).")
    parser.add_argument("--as-gamma", type=float, default=5e-5,
                        help="AS-MM: risk aversion γ. Подобран под наш масштаб (R_BAR=1000).")
    parser.add_argument("--as-k", type=float, default=1.5,
                        help="AS-MM: order intensity decay k. Стандарт.")
    parser.add_argument("--as-sigma-a", type=float, default=4.4,
                        help="AS-MM: оценка abs volatility A_USD mid (cents per √sec). "
                             "Калибруется из baseline-run через metrics.estimate_volatility_abs.")
    parser.add_argument("--as-sigma-b", type=float, default=3.2,
                        help="AS-MM: оценка abs volatility B_EUR mid (EUR-cents per √sec). "
                             "Cross-Market AS не использует — σ_B вычисляется через FX.")

    # --- GLFT (стационарное приближение AS) ---
    parser.add_argument("--glft-gamma", type=float, default=5e-7,
                        help="GLFT: risk aversion γ. Аналогично --as-gamma.")
    parser.add_argument("--glft-k", type=float, default=1.5,
                        help="GLFT: intensity decay k в λ(δ) = A·exp(-k·δ).")
    parser.add_argument("--glft-A", type=float, default=1.0,
                        help="GLFT: intensity scale A. Чем меньше A, тем СИЛЬНЕЕ inventory penalty "
                             "(η растёт). Default A=1.0 даёт subtle skew, можно уменьшить до 0.1 "
                             "для более агрессивного inventory mgmt.")
    parser.add_argument("--glft-sigma", type=float, default=4.4,
                        help="GLFT: σ mid в cents/√sec (используется одинаково для A_USD и B_EUR; "
                             "GLFT — single-asset стратегия в отличие от Cross-Market AS).")

    # --- RL-Tuned GLFT (DQN выбирает γ из дискретного gamma_set) ---
    parser.add_argument("--rl-gamma-set", type=str, default=None,
                        help="Comma-separated значения γ для RL action space, "
                             "например '1e-8,5e-8,1e-7,5e-7,1e-6,5e-6,1e-5'. "
                             "Default — 7 log-uniform значений.")
    parser.add_argument("--rl-sigma", type=float, default=4.4,
                        help="RL-GLFT: σ mid в cents/√sec. Default = glft default.")
    parser.add_argument("--rl-k", type=float, default=1.5,
                        help="RL-GLFT: k intensity decay. Default = glft default.")
    parser.add_argument("--rl-A", type=float, default=1.0,
                        help="RL-GLFT: A intensity scale. Default = glft default.")
    parser.add_argument("--rl-policy-path", type=str, default=None,
                        help="Путь к .pt с обученной DQN-policy. Если None — policy "
                             "не загружается, стратегия использует gamma_set[0] (fallback). "
                             "Это полезно для baseline-сравнения 'RL без обучения'.")

    # --- SpreadBasedMM (FixedLadderStrategy в UnifiedMM) ---
    parser.add_argument("--sb-order-size", type=int, default=1,
                        help="SpreadBasedMM: размер ордера на каждом тике лестницы.")
    parser.add_argument("--sb-window-size", type=int, default=5,
                        help="SpreadBasedMM: ширина окна вокруг mid в тиках.")
    parser.add_argument("--sb-num-ticks", type=int, default=20,
                        help="SpreadBasedMM: число тиков-уровней с каждой стороны.")
    parser.add_argument("--sb-anchor", choices=["top", "middle", "bottom"], default="bottom",
                        help="SpreadBasedMM: куда якорится window относительно mid.")

    # --- AdaptiveMM (SizeSkewLadderStrategy в UnifiedMM) ---
    parser.add_argument("--adp-min-order-size", type=int, default=20,
                        help="AdaptiveMM: base size — размер ордера при q=0 (симметрии).")
    parser.add_argument("--adp-window-size", type=str, default="2",
                        help="AdaptiveMM: ширина окна вокруг mid в тиках.")
    parser.add_argument("--adp-num-ticks", type=int, default=20,
                        help="AdaptiveMM: число тиков-уровней с каждой стороны.")
    parser.add_argument("--adp-anchor", choices=["top", "middle", "bottom"], default="middle",
                        help="AdaptiveMM: якорь window относительно mid.")
    parser.add_argument("--adp-skew-beta", type=float, default=10.0,
                        help="AdaptiveMM: параметр sigmoid'а size-skew. 0 = без skew, "
                             "→∞ = весь объём на одной стороне при q≠0.")

    # --- Fast mode (lightweight logging) ---
    parser.add_argument("--fast-mode", action="store_true",
                        help="Отключает тяжёлые логи exchange'а: per-trade ORDER_EXECUTED "
                             "и orderbook snapshots. Экономит ~25-30%% wall-time. "
                             "Подходит когда нужен только агрегированный PnL агентов (например "
                             "для comparison.ipynb матрицы). НЕ ИСПОЛЬЗОВАТЬ если нужны "
                             "bid/ask/mid графики или volume-by-agent-type разбивка.")
    parser.add_argument("--stress-multiplier", type=float, default=1.0,
                        help="Множитель частоты megashock'ов фундаментала. "
                             "1.0 = baseline (стационар, ~0 шоков/час). 1e6 → ~10 шоков/час.")
    parser.add_argument("--megashock-mean", type=float, default=10.0,
                        help="Средняя амплитуда megashock'а r_company в центах. "
                             "Default=10 (≈1% от r_bar=1000). Увеличить (например до 50) "
                             "чтобы megashock'и были визуально заметнее на графиках mid.")
    parser.add_argument("--megashock-var", type=float, default=5.0,
                        help="Дисперсия амплитуды megashock'а. Default=5 → std≈2.2 центов.")
    parser.add_argument("--fx-shock", action="append", default=[],
                        help="Шок на FX-рынке. Формат: 'HH:MM:size:direction' либо "
                             "'random:size:direction'. direction = buy|sell. Например "
                             "'10:00:2000:buy' = в 10:00 купить 2000 шт FX по market price. "
                             "Random берёт случайное время из [open+10min, close-10min]. "
                             "Повторяемый: можно задать несколько шоков.")
    parser.add_argument("--news-shock", action="append", default=[],
                        help="Одноразовый scheduled jump в r_company (fundamental). "
                             "Формат: 'HH:MM:+magnitude' или 'random:+magnitude' (знак "
                             "обязателен: +300 или -250). Магнитуда в центах. Применяется "
                             "ПОВЕРХ обычного OU+megashock процесса. Повторяемый.")
    parser.add_argument("--drift-shock", action="append", default=[],
                        help="Устойчивый тренд (drift) в r_company на окне. Формат: "
                             "'HH:MM-HH:MM:+magnitude' или 'random:5min:+magnitude'. "
                             "magnitude — суммарное изменение цены за всё окно. Внутри "
                             "разбивается на мелкие jump'ы с шагом 5 сек. Повторяемый.")
    parser.add_argument("--liquidity-shock", action="append", default=[],
                        help="Очистка стороны стакана. Формат: 'symbol:HH:MM:side' или "
                             "'symbol:random:side'. symbol ∈ {A_USD, B_EUR, FX}, "
                             "side ∈ {bid, ask}. В заданное время все limit-orders на "
                             "указанной стороне удаляются (прямая модификация order book). "
                             "Повторяемый.")
    parser.add_argument("--pressure-shock", action="append", default=[],
                        help="Sustained one-sided pressure через периодические лимит-заявки. "
                             "Формат: 'symbol:HH:MM-HH:MM:side:orders_per_wake:order_size:wake_freq'. "
                             "Пример: 'A_USD:10:00-10:15:ask:50:5:2s' — на A_USD с 10:00 до "
                             "10:15 каждые 2 сек ставит 50 sell-orders по 5 шт каждый. Имитирует "
                             "навес предложения (или demand wall при side=bid). Комбинировать "
                             "с --drift-shock для реалистичного bear-market. Повторяемый.")
    parser.add_argument("-l", "--log-dir", default=None)
    args = parser.parse_args()

    agents, oracle, kernel_start, mkt_close, kernel_seed = build(
        seed=args.seed,
        num_noise=args.num_noise,
        num_value=args.num_value,
        value_lambda_a_override=args.value_lambda_a,
        value_sigma_n_override=args.value_sigma_n,
        mm_wake_freq_override=args.mm_wake_freq,
        num_momentum=args.num_momentum,
        num_zi=args.num_zi,
        num_noise_fx=args.num_noise_fx,
        num_zi_fx=args.num_zi_fx,
        num_momentum_fx=args.num_momentum_fx,
        arb_order_size=args.arb_order_size,
        arb_open_pct=args.arb_open_pct,
        arb_close_pct=args.arb_close_pct,
        arb_wake_freq=args.arb_wake_freq,
        arb_starting_inventory=args.arb_starting_inventory,
        arb_max_exposure=args.arb_max_exposure,
        mm_starting_cash=args.mm_starting_cash,
        arb_starting_cash=args.arb_starting_cash,
        maker_rebate=args.maker_rebate,
        mm_type=args.mm_type,
        as_gamma=args.as_gamma,
        as_k=args.as_k,
        as_sigma_a=args.as_sigma_a,
        as_sigma_b=args.as_sigma_b,
        glft_gamma=args.glft_gamma,
        glft_k=args.glft_k,
        glft_A=args.glft_A,
        glft_sigma=args.glft_sigma,
        rl_gamma_set=_parse_gamma_set(args.rl_gamma_set),
        rl_sigma=args.rl_sigma,
        rl_k=args.rl_k,
        rl_A=args.rl_A,
        rl_policy_path=args.rl_policy_path,
        sb_order_size=args.sb_order_size,
        sb_window_size=args.sb_window_size,
        sb_num_ticks=args.sb_num_ticks,
        sb_anchor=args.sb_anchor,
        adp_min_order_size=args.adp_min_order_size,
        adp_window_size=args.adp_window_size,
        adp_num_ticks=args.adp_num_ticks,
        adp_anchor=args.adp_anchor,
        adp_skew_beta=args.adp_skew_beta,
        fast_mode=args.fast_mode,
        stress_multiplier=args.stress_multiplier,
        megashock_mean=args.megashock_mean,
        megashock_var=args.megashock_var,
        fx_shock_specs=args.fx_shock,
        news_shock_specs=args.news_shock,
        drift_shock_specs=args.drift_shock,
        liquidity_shock_specs=args.liquidity_shock,
        pressure_shock_specs=args.pressure_shock,
    )

    print(f"Built {len(agents)} agents")

    # Используем kernel_seed, зафиксированный в build() сразу после np.random.seed(seed).
    # Это делает latency-noise в Kernel.sendMessage воспроизводимой и НЕЗАВИСИМОЙ
    # от числа шок-агентов / 'random'-времён — без чего pre-shock траектории
    # разъезжаются.
    kernel = Kernel(
        "MultiMarket Baseline Kernel",
        random_state=np.random.RandomState(seed=kernel_seed),
    )

    kernel.runner(
        agents=agents,
        startTime=kernel_start,
        stopTime=mkt_close + pd.to_timedelta("00:01:00"),
        defaultComputationDelay=50,
        defaultLatency=50,
        oracle=oracle,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
