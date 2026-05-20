"""Загрузка логов одной симуляции для анализа"""

from pathlib import Path

import pandas as pd


DEFAULT_LOGS_BASE = Path("experiments/log")


def _read_pickle_bz2(path):
    return pd.read_pickle(path, compression="bz2")


def run_dir(run_name, base=None):
    base = Path(base) if base is not None else DEFAULT_LOGS_BASE
    return Path(base) / run_name


def load_orderbook(run_name, symbol, base=None):
    path = run_dir(run_name, base) / f"ORDERBOOK_{symbol}_FREQ_1s.bz2"
    # SparseDtype от wide_book ломает арифметику — приводим к плотному float.
    return _read_pickle_bz2(path).astype(float)


def load_fundamental(run_name, symbol="A_USD", base=None):
    return _read_pickle_bz2(run_dir(run_name, base) / f"fundamental_{symbol}.bz2")


def load_agent_log(run_name, agent_name, base=None):
    return _read_pickle_bz2(run_dir(run_name, base) / f"{agent_name}.bz2")


def load_summary(run_name, base=None):
    return _read_pickle_bz2(run_dir(run_name, base) / "summary_log.bz2")


def list_agents(run_name, base=None):
    rd = run_dir(run_name, base)
    return [
        p.stem for p in sorted(rd.iterdir())
        if p.suffix == ".bz2"
        and not (p.stem.startswith("ORDERBOOK_")
                 or p.stem.startswith("fundamental_")
                 or p.stem == "summary_log")
    ]
