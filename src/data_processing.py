"""UTC daily alignment and explicit data quality accounting without imputation."""

import json
import hashlib

import numpy as np
import pandas as pd

from .config import RAW_DIR, PROCESSED_DIR, Settings


def read_cached_payload(path):
    """Validate a collected JSON envelope before using its data."""
    envelope = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(envelope["data"], sort_keys=True).encode()).hexdigest()
    if digest != envelope["sha256"]:
        raise ValueError(f"{path.name}: raw cache integrity check failed")
    return envelope["data"]


def daily_series(rows, unit, name, calendar):
    """Select the last actual observation per UTC day; retain gaps as NaN."""
    frame = pd.DataFrame(rows, columns=["timestamp", "value"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit=unit, utc=True, errors="raise")
    if frame["timestamp"].isna().any():
        raise ValueError(f"{name}: missing timestamps")
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    invalid = (~np.isfinite(frame["value"]) | (frame["value"] < 0))
    if invalid.any():
        raise ValueError(f"{name}: invalid or negative observations")
    duplicate_times = int(frame.duplicated("timestamp").sum())
    conflicts = frame.groupby("timestamp")["value"].nunique().gt(1).sum()
    if conflicts:
        raise ValueError(f"{name}: conflicting values at identical timestamps")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    frame["date"] = frame["timestamp"].dt.floor("D")
    selected = frame.loc[frame["date"].isin(calendar)].copy()
    intraday = int(selected.duplicated("date").sum())
    non_midnight = int((selected["timestamp"] != selected["date"]).sum())
    daily = selected.drop_duplicates("date", keep="last").set_index("date")["value"].rename(name)
    aligned = daily.reindex(calendar)
    aligned.index.name = "date"
    audit = {"raw_rows": len(rows), "duplicate_timestamps": duplicate_times,
             "conflicting_timestamps": int(conflicts), "extra_observations_same_day": intraday,
             "non_midnight_observations_in_period": non_midnight,
             "expected_days": len(calendar), "observed_days": int(aligned.notna().sum()),
             "missing_days": int(aligned.isna().sum()), "zero_values": int(aligned.eq(0).sum()),
             "first_date": daily.index.min().isoformat() if len(daily) else None,
             "last_date": daily.index.max().isoformat() if len(daily) else None,
             "missing_dates": [d.isoformat() for d in aligned.index[aligned.isna()]]}
    selected["series"] = name
    return aligned, selected[["series", "timestamp", "date", "value"]], audit


def prepare_data():
    settings = Settings()
    start, end = settings.period
    calendar = pd.date_range(start, end, freq="D", inclusive="left", name="date")
    manifest = json.loads((RAW_DIR / "history_manifest.json").read_text(encoding="utf-8"))
    if manifest["period_utc_half_open"] != [start.isoformat(), end.isoformat()]:
        raise ValueError("Cached research period differs from settings; collect history first")
    audit = {"period_utc_half_open": manifest["period_utc_half_open"], "expected_days": len(calendar),
             "daily_rule": "Last actual observation within UTC day; no fill or interpolation",
             "series": {}, "sources": manifest["sources"]}
    protocols, market, observations = [], [], []
    payloads = {}
    for source, meta in manifest["sources"].items():
        if meta["status"] != "ok":
            continue
        payload = read_cached_payload(RAW_DIR / meta["file"])
        payloads[source] = payload
        if source.startswith("coingecko_"):
            coin = "aave" if source.endswith("_aave") else "eth"
            for field, label in [("prices", "price_usd"), ("market_caps", "market_cap_usd"), ("total_volumes", "volume_24h_usd")]:
                name = f"{coin}_{label}"
                series, raw_frame, quality = daily_series(payload[field], "ms", name, calendar)
                market.append(series)
                observations.append(raw_frame)
                audit["series"][name] = quality
        else:
            name = {"aave": "aave", "morpho": "morpho", "compound-finance": "compound"}[source] + "_tvl_usd"
            rows = [[x["date"], x["totalLiquidityUSD"]] for x in payload["tvl"]]
            series, raw_frame, quality = daily_series(rows, "s", name, calendar)
            protocols.append(series)
            observations.append(raw_frame)
            audit["series"][name] = quality
    protocol = pd.concat(protocols, axis=1) if protocols else pd.DataFrame(index=calendar)
    tokens = pd.concat(market, axis=1) if market else pd.DataFrame(index=calendar)
    combined = protocol.join(tokens, how="outer")
    required = ["aave_tvl_usd", "morpho_tvl_usd", "compound_tvl_usd", "aave_price_usd",
                "eth_price_usd", "aave_market_cap_usd", "aave_volume_24h_usd"]
    common = combined.dropna(subset=required) if all(c in combined for c in required) else combined.iloc[:0]
    audit["common"] = {"required_columns": required, "days": len(common),
                       "first_date": common.index.min().isoformat() if len(common) else None,
                       "last_date": common.index.max().isoformat() if len(common) else None}
    audit["missing_sources"] = [k for k, v in manifest["sources"].items() if v["status"] != "ok"]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in [("protocol_daily", protocol), ("market_daily", tokens), ("combined_daily", combined), ("common_daily", common)]:
        frame.to_csv(PROCESSED_DIR / f"{name}.csv", index_label="date")
    if observations:
        pd.concat(observations, ignore_index=True).to_csv(RAW_DIR / "daily_observations.csv", index=False)
    (PROCESSED_DIR / "data_quality.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return protocol, tokens, common, audit, payloads
