"""EDA metrics: calendar alignment, native token units, and explicit denominators."""

import json
from decimal import Decimal

import numpy as np
import pandas as pd

from .config import ROOT, RAW_DIR, PROCESSED_DIR, REQUIRED_SOURCES, Settings
from .data_processing import prepare_data, daily_series, read_cached_payload
from .onchain import RAW, POOL, load_snapshot, validate_events, daily_activity, compare_event_sets

EDA_DIR = PROCESSED_DIR / "eda"
RESERVES = {"USDC": ("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", 6),
            "WETH": ("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", 18),
            "USDT": ("0xdac17f958d2ee523a2206206994597c13d831ec7", 6)}


def safe_ratio(numerator, denominator):
    return numerator / denominator.where(denominator > 0)


def log_returns(series):
    """Do not bridge missing dates or turn non-positive levels into returns."""
    return np.log(series.where(series > 0)).diff()


def annualized_volatility(returns, periods_per_year=365):
    """Sample standard deviation of observed daily returns, annualized."""
    if periods_per_year <= 0:
        raise ValueError("Annualization frequency must be positive")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def ray_to_apy(value):
    """Official Aave frontend convention: APR in ray, per-second compounding."""
    apr = float(Decimal(str(value)) / Decimal(10**27))
    if not np.isfinite(apr) or apr < 0:
        raise ValueError("Invalid ray APR")
    return float(np.expm1(31536000 * np.log1p(apr / 31536000)))


def lending_snapshot(records):
    rows = []
    for symbol, (asset, decimals) in RESERVES.items():
        matches = [r for r in records if r["underlyingAsset"].lower() == asset and (r["pool"].get("pool") or "").lower() == POOL.lower()]
        if len(matches) != 1:
            raise ValueError(f"Expected one verified reserve: {symbol}")
        r = matches[0]
        if int(r["decimals"]) != decimals:
            raise ValueError(f"Unexpected decimals for {symbol}")
        unit = Decimal(10)**decimals
        amount = lambda key: float(Decimal(r[key]) / unit)
        stable = amount("totalPrincipalStableDebt")
        supply, variable, available = (amount(k) for k in ["totalATokenSupply", "totalCurrentVariableDebt", "availableLiquidity"])
        rows.append({"symbol": symbol, "asset": asset, "decimals": decimals,
            "last_update_utc": pd.to_datetime(r["lastUpdateTimestamp"], unit="s", utc=True),
            "supply_token": supply, "variable_borrow_token": variable, "stable_principal_token": stable,
            "borrow_token": variable if stable == 0 else np.nan,
            "available_token": available, "utilization_reported": float(r["utilizationRate"]),
            "borrow_supply_ratio": variable / supply if stable == 0 and supply > 0 else np.nan,
            "variable_debt_liquidity_ratio": variable / (variable+available) if variable+available > 0 else np.nan,
            "supply_apy": ray_to_apy(r["liquidityRate"]), "variable_borrow_apy": ray_to_apy(r["variableBorrowRate"])})
    return pd.DataFrame(rows).set_index("symbol")


def coverage_row(name, source, frame, frequency, units, limitations, fields=None):
    observed = frame.dropna(how="all")
    return {"Набор": name, "Источник": source, "Период UTC":
            f"{observed.index.min()} — {observed.index.max()}" if len(observed) else "нет данных",
            "Наблюдения": len(observed), "Частота": frequency,
            "Поля": fields or ", ".join(map(str, frame.columns)), "Пропуски": int(frame.isna().sum().sum()),
            "Единицы": units, "Ограничения": limitations}


def prepare_eda():
    gate = json.loads((RAW / "stage3_manifest.json").read_text(encoding="utf-8"))
    if not gate.get("analysis_ready") or not set(REQUIRED_SOURCES) <= set(gate["instruments_with_real_data"]):
        raise RuntimeError("All five sources must be verified before full EDA")
    protocol, market, common, audit, payloads = prepare_data()
    calendar = protocol.index
    p = protocol.copy()
    p["tvl_change_30d"] = p.aave_tvl_usd.pct_change(30, fill_method=None)
    p["tvl_change_90d"] = p.aave_tvl_usd.pct_change(90, fill_method=None)
    trio = protocol[["aave_tvl_usd", "morpho_tvl_usd", "compound_tvl_usd"]]
    p["aave_share_three"] = safe_ratio(p.aave_tvl_usd, trio.sum(axis=1, min_count=3))
    p["tvl_log_change"] = log_returns(p.aave_tvl_usd)
    t = market.copy()
    for coin in ["aave", "eth"]:
        t[f"{coin}_log_return"] = log_returns(t[f"{coin}_price_usd"])
    t["aave_rolling_30d_vol_annual"] = t.aave_log_return.rolling(30, min_periods=30).std(ddof=1) * np.sqrt(365)
    t["market_cap_tvl"] = safe_ratio(t.aave_market_cap_usd, p.aave_tvl_usd)

    economics = pd.DataFrame(index=calendar)
    econ_audit, econ_definitions = {}, {}
    manifest_path = RAW_DIR / "economics_manifest.json"
    econ_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for metric in ["fees", "revenue"]:
        meta = econ_manifest.get(metric, {})
        if meta.get("status") == "ok":
            raw = read_cached_payload(RAW_DIR / meta["file"])
            series, _, quality = daily_series(raw["totalDataChart"], "s", metric+"_usd", calendar)
            economics[metric+"_usd"] = series
            econ_audit[metric] = quality
            econ_definitions[metric] = {"linkedProtocols": raw.get("linkedProtocols"), "children": raw.get("childProtocols"), "scope": raw.get("name")}
        else:
            economics[metric+"_usd"] = np.nan
            econ_audit[metric] = {"unavailable": meta}
        economics[metric+"_tvl_daily"] = safe_ratio(economics[metric+"_usd"], p.aave_tvl_usd)
        economics[metric+"_usd_ma7"] = economics[metric+"_usd"].rolling(7, min_periods=7).mean()

    chains = pd.DataFrame(index=calendar)
    chain_audit = {}
    for chain, values in payloads["aave"]["chainTvls"].items():
        if chain in {"staking", "pool2", "borrowed", "doublecounted", "vesting"} or chain.endswith(("-staking", "-pool2", "-borrowed", "-doublecounted", "-vesting")):
            continue
        rows = [[row["date"], row["totalLiquidityUSD"]] for row in values.get("tvl", [])]
        if not rows:
            continue
        series, _, quality = daily_series(rows, "s", chain, calendar)
        chains[chain] = series
        chain_audit[chain] = quality
    last = calendar[-1]
    chain_latest = chains.loc[last].dropna().sort_values(ascending=False).to_frame("tvl_usd")
    chain_latest["share_parent_tvl"] = chain_latest.tvl_usd / p.loc[last, "aave_tvl_usd"]

    graph = load_snapshot("graph_markets")
    reserves = lending_snapshot(graph["data"]["reserves"])
    graph_history = pd.json_normalize(graph["data"]["reserveParamsHistoryItems"])
    graph_history["date"] = pd.to_datetime(graph_history.timestamp, unit="s", utc=True)
    events_snapshot = load_snapshot("web3_events")
    start = pd.Timestamp(gate["start_utc"])
    end = pd.Timestamp(gate["end_exclusive_utc"])
    events, duplicates = validate_events(pd.DataFrame(events_snapshot["data"]["decoded"]), start, end)
    etherscan_snapshot = load_snapshot("etherscan_events")
    etherscan, etherscan_duplicates = validate_events(pd.DataFrame(etherscan_snapshot["data"]["decoded"]), start, end)
    if duplicates or etherscan_duplicates or not compare_event_sets(events, etherscan)["exact_match"]:
        raise RuntimeError("Verified event sample required")
    hourly = events.groupby([events.block_time.dt.hour, "event_name"]).size().unstack(fill_value=0).reindex(index=range(24), columns=["Supply", "Borrow", "Repay"], fill_value=0)
    participants = pd.Series({kind: int(daily_activity(events.loc[events.event_name.eq(kind)], start, end).active_participant_addresses.iloc[0]) for kind in ["Supply", "Borrow", "Repay"]})
    symbols = {r["underlyingAsset"].lower(): r["symbol"] for r in graph["data"]["reserves"]}
    reserve_counts = events.reserve.value_counts().rename(index=lambda a: symbols.get(a, a[:10]+"…"))

    joint = pd.DataFrame({"Δlog TVL": p.tvl_log_change, "AAVE r": t.aave_log_return, "ETH r": t.eth_log_return,
                          "Δlog Fees (t−1)": log_returns(economics.fees_usd).shift(1),
                          "Δlog Revenue (t−1)": log_returns(economics.revenue_usd).shift(1)}).dropna()
    corr = joint.corr(method="pearson")
    correlations = pd.DataFrame(columns=["left", "right", "n", "pearson"])
    pairs = [("aave_log_return", "eth_log_return")]
    rows = []
    for left, right in pairs:
        frame = t[[left, right]].dropna()
        rows.append({"left": left, "right": right, "n": len(frame), "pearson": frame[left].corr(frame[right])})
    for field in ["AAVE r", "Δlog Fees (t−1)", "Δlog Revenue (t−1)"]:
        rows.append({"left": "Δlog TVL", "right": field, "n": len(joint), "pearson": joint["Δlog TVL"].corr(joint[field])})
    correlations = pd.DataFrame(rows)
    price_dates = t.loc[t.aave_price_usd.notna()].index
    summary = {"tvl_start_usd": float(p.aave_tvl_usd.iloc[0]), "tvl_end_usd": float(p.aave_tvl_usd.iloc[-1]),
        "tvl_period_change": float(p.aave_tvl_usd.iloc[-1]/p.aave_tvl_usd.iloc[0]-1),
        "tvl_change_30d": float(p.tvl_change_30d.iloc[-1]), "tvl_change_90d": float(p.tvl_change_90d.iloc[-1]),
        "share_three_start": float(p.aave_share_three.iloc[0]), "share_three_end": float(p.aave_share_three.iloc[-1]),
        "chain_reconciliation_usd": float(chain_latest.tvl_usd.sum()-p.aave_tvl_usd.iloc[-1]),
        "chain_largest": str(chain_latest.index[0]), "chain_largest_share": float(chain_latest.share_parent_tvl.iloc[0]),
        "aave_price_start": float(t.loc[price_dates[0], "aave_price_usd"]), "aave_price_end": float(t.loc[price_dates[-1], "aave_price_usd"]),
        "aave_cumulative_return": float(t.loc[price_dates[-1], "aave_price_usd"]/t.loc[price_dates[0], "aave_price_usd"]-1),
        "eth_cumulative_return": float(t.loc[price_dates[-1], "eth_price_usd"]/t.loc[price_dates[0], "eth_price_usd"]-1),
        "aave_vol_annual": annualized_volatility(t.aave_log_return), "aave_daily_returns_n": int(t.aave_log_return.notna().sum()),
        "token_corr": float(correlations.iloc[0].pearson), "correlation_joint_n": len(joint),
        "market_cap_tvl_end": float(t.market_cap_tvl.iloc[-1]), "market_cap_tvl_start": float(t.loc[price_dates[0], "market_cap_tvl"]),
        "market_cap_end_usd": float(t.aave_market_cap_usd.iloc[-1]),
        "event_count": len(events), "participants_by_type": participants.to_dict(),
        "unique_transactions": gate["rpc"]["unique_transactions"], "unique_participants": gate["rpc"]["active_participant_addresses"],
        "competitor_changes": {name: float(protocol[name].iloc[-1]/protocol[name].iloc[0]-1) for name in protocol},
        "events_peak_hour": int(hourly.sum(axis=1).idxmax()), "events_peak_hour_count": int(hourly.sum(axis=1).max()),
        "lending_utilization_gap_pp_max": float(((reserves.utilization_reported-reserves.borrow_supply_ratio).abs()*100).max())}
    for metric in ["fees", "revenue"]:
        s = economics[metric+"_usd"]
        summary[metric+"_total_usd"] = float(s.sum(min_count=len(s))) if s.notna().all() else None
        summary[metric+"_first30_mean"] = float(s.iloc[:30].mean()) if s.iloc[:30].notna().all() else None
        summary[metric+"_last30_mean"] = float(s.iloc[-30:].mean()) if s.iloc[-30:].notna().all() else None
        ratio = economics[metric+"_tvl_daily"].iloc[-30:].mean()
        summary[metric+"_last30_tvl_ratio"] = float(ratio) if pd.notna(ratio) else None

    coverage = [coverage_row("TVL трёх протоколов", "DeFiLlama", protocol, "daily", "USD, остаток", "Родительские агрегаты; TVL не равен supply"),
        coverage_row("AAVE и ETH", "CoinGecko", market, "daily, 00:00 UTC", "USD; volume — USD/24 ч", "20 дней недоступны: по 20 пропусков в каждом из 6 рядов"),
        coverage_row("Fees и revenue Aave", "DeFiLlama", economics[["fees_usd", "revenue_usd"]], "daily, поток за сутки", "USD/сутки", "Агрегаты версий, определения childProtocols; не прибыль токена"),
        coverage_row("TVL по сетям", "DeFiLlama", chains, "daily", "USD", "Пропуски в том числе до появления сети; не заполнены нулями"),
        {"Набор": "Резервы на блоке 26316188", "Источник": "The Graph", "Период UTC": "31.10.2026, исторический блок", "Наблюдения": len(graph["data"]["reserves"]), "Частота": "snapshot", "Поля": "asset, decimals, supply, variable debt, available, utilization, APR, lastUpdate", "Пропуски": int(pd.json_normalize(graph["data"]["reserves"]).isna().sum().sum()), "Единицы": "raw → единицы токена; APR ray → APY", "Ограничения": "Разные lastUpdate; выбраны USDC/WETH/USDT; не годовая история"},
        {"Набор": "Параметры резервов", "Источник": "The Graph", "Период UTC": f"{graph_history.date.min()} — {graph_history.date.max()}", "Наблюдения": len(graph_history), "Частота": "событийная", "Поля": "timestamp, reserve, balances, rates, utilization", "Пропуски": int(graph_history.isna().sum().sum()), "Единицы": "raw token units, ray, доля", "Ограничения": "Только одни сутки; не интерполируется до года"}]
    for source in ["web3.py", "Etherscan API V2"]:
        coverage.append({"Набор": "Supply/Borrow/Repay", "Источник": source, "Период UTC": f"{events.block_time.min()} — {events.block_time.max()}", "Наблюдения": len(events), "Частота": "события, полные UTC-сутки 31.10.2026", "Поля": ", ".join(events.columns), "Пропуски": 0, "Единицы": "счётчики; amount_raw uint256 по активу", "Ограничения": "Не все сети Aave; пустые неприменимые поля участников не являются потерей данных"})
    bundle = {"protocol": p, "tokens": t, "economics": economics, "chains": chains, "chain_latest": chain_latest,
              "lending": reserves, "events": events, "hourly": hourly, "participants": participants,
              "reserve_counts": reserve_counts, "joint": joint, "correlations": correlations, "correlation_matrix": corr,
              "coverage": pd.DataFrame(coverage), "summary": summary,
              "audit": {"series": audit["series"], "economics": econ_audit, "chains": chain_audit, "economics_definitions": econ_definitions}}
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    for name, value in bundle.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            value.to_csv(EDA_DIR / f"{name}.csv", index=True)
    (EDA_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (EDA_DIR / "audit.json").write_text(json.dumps(bundle["audit"], ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle
