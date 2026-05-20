## Структура проекта

├── src/
│   ├── agents/
│   │   ├── mm_strategies.py          — все MM-стратегии
│   │   ├── unified_mm.py             — MM-агент
│   │   ├── cross_market_state.py     — общий inventory для CM-AS / CM-GLFT
│   │   ├── fx_arb.py                 — FX-арбитражёр
│   │   ├── fx_shock_agent.py         — FX-шок
│   │   ├── liquidity_shock_agent.py  — Liquidity-шок
│   │   └── pressure_agent.py         — Давление заявок с одной стороны стакана
│   ├── multimarket/
│   │   ├── oracle.py                 — Фундаментальная цена
│   │   └── multi_currency_agent.py   — Базовый класс для мультивалютных агентов
│   ├── rl/
│   │   ├── dqn.py                    — DQN для RL GLFT
│   │   ├── env.py                    — обёртка над ABIDES + ReplayBuffer
│   │   └── train.py                  — обучение модели
│   ├── scenarios/
│   │   └── multimarket_baseline.py   — конфигуратор сценария
│   └── analysis/                     — данные для анализа симуляций
│
├── notebooks/
│   ├── comparison.ipynb              — основное сравнение алгоритмов ММ
│   ├── market_microstructure.ipynb   — bid/ask/mid, fundamental vs market, объёмы
│   └── scenarios_overview.ipynb      — графики каждого шокового сценария
│
├── experiments/
│   ├── log/                          — ABIDES-логи
│   └── rl/                           — DQN-чекпойнты + latest_best.pt
├── vendor/abides/                    — ABIDES
└── requirements.txt

## Алгоритмы

Все стратегии реализуют общий интерфейс `MMStrategy.compute_quotes` и
подставляются в `UnifiedMM`. Смена алгоритма = подмена одного объекта
в конфигурации.

| # | Стратегия | Класс | Источник |
|---|---|---|---|
| 1 | **BaselineMM** | `FixedSpreadStrategy` | контроль, фикс-spread |
| 2 | **SpreadBasedMM** | `FixedLadderStrategy` | Chakraborty & Kearns 2011 |
| 3 | **AdaptiveMM** | `SizeSkewLadderStrategy` | Chakraborty & Kearns 2011 + size-skew |
| 4 | **AvellanedaStoikovMM** | `CrossMarketAvellanedaStoikovStrategy` | Bergault et al. 2021 |
| 5 | **GLFTMM** | `CrossMarketGLFTStrategy` | Guéant-Lehalle-Fernandez-Tapia 2013 + cross-market |
| 6 | **RLGLFTMM** | `RLTunedGLFTStrategy` + DQN-policy | надстройка над GLFTMM |

## Сценарии

| Имя | Что моделирует |
|---|---|
| `stationary` | без шоков, базовый режим |
| `megashock` | частые прыжки фундаментальной цены (волатильный режим) |
| `drift_down` | устойчивый тренд вниз |
| `drift_up` | устойчивый тренд вверх |
| `news_shock` | одноразовый сильный прыжок фундаментала|
| `liquidity_crisis` | снос bid-side стакана A_USD |
| `fx_shock` | резкий FX шок |
