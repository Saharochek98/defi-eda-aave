"""Read-only API diagnostics and cached historical collection."""

import argparse
import json
import hashlib
from datetime import datetime, timezone, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import ROOT, RAW_DIR, Settings, REQUIRED_SOURCES, ETHEREUM_PUBLIC_RPC


def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=1, backoff_factor=1, status_forcelist=[408, 429, 500, 502, 503, 504],
                    allowed_methods=["GET", "POST"], respect_retry_after_header=False)
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers["User-Agent"] = "defi-eda/0.2"
    return session


def check_sources(refresh: bool = False) -> dict:
    """Persist a diagnostic snapshot without writing credentials or full errors."""
    report_path = ROOT / "source_availability.json"
    if report_path.exists() and not refresh:
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if cached.get("required_sources") == list(REQUIRED_SOURCES):
            return cached
    settings = Settings()
    start, end = settings.period
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "research_as_of": settings.as_of.isoformat(),
              "period_start_utc": start.isoformat(), "period_end_exclusive_utc": end.isoformat(),
              "required_sources": list(REQUIRED_SOURCES), "sources": {}, "ranking": {"status": "not_verified_by_api"}}
    session = make_session()

    def probe(name, method, url, validate, **kwargs):
        try:
            response = session.request(method, url, timeout=(5, 10), **kwargs)
            if not response.ok:
                report["sources"][name] = {"status": "http_error", "http_status": response.status_code}
                return None
            payload = response.json()
            if not validate(payload):
                report["sources"][name] = {"status": "invalid_or_error_response"}
                return None
            report["sources"][name] = {"status": "probe_ok", "http_status": response.status_code}
            return payload
        except requests.RequestException as exc:
            report["sources"][name] = {"status": "network_error", "error_type": type(exc).__name__}
        except (ValueError, TypeError, KeyError, AttributeError):
            report["sources"][name] = {"status": "invalid_response"}
        return None

    protocols = probe("DeFiLlama", "GET", "https://api.llama.fi/protocols",
                      lambda p: isinstance(p, list) and bool(p) and all(
                          isinstance(x, dict) and "name" in x and "tvl" in x for x in p))
    if protocols:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_DIR / "defillama_protocols.json").write_text(json.dumps(protocols), encoding="utf-8")
        ranked = sorted((p for p in protocols if isinstance(p.get("tvl"), (int, float))),
                        key=lambda p: p["tvl"], reverse=True)
        # Version-level rows are not a parent-protocol UI ranking.
        matches = [(i, p) for i, p in enumerate(ranked, 1) if p.get("slug") == "aave"]
        if matches:
            rank, aave = matches[0]
            report["ranking"] = {"status": "verified_api_rows", "rank": rank,
                                 "top_50": rank <= 50, "tvl_usd": aave["tvl"],
                                 "definition": "All numeric TVL rows in /protocols, no parent aggregation"}
        else:
            report["ranking"] = {"status": "parent_aggregation_required",
                                 "note": "Use the parent-protocol website ranking; do not rank Aave V3 as Aave."}

    cg_base = "https://pro-api.coingecko.com/api/v3" if settings.coingecko_tier == "pro" else "https://api.coingecko.com/api/v3"
    cg_headers = {}
    if settings.coingecko_key:
        cg_headers["x-cg-pro-api-key" if settings.coingecko_tier == "pro" else "x-cg-demo-api-key"] = settings.coingecko_key
    probe("CoinGecko", "GET", cg_base + "/ping", lambda p: isinstance(p, dict) and "gecko_says" in p,
          headers=cg_headers)

    if settings.graph_key and settings.graph_subgraph:
        probe("The Graph", "POST", f"https://gateway.thegraph.com/api/subgraphs/id/{settings.graph_subgraph}",
              lambda p: isinstance(p, dict) and not p.get("errors") and bool(p.get("data", {}).get("__schema")),
              headers={"Authorization": f"Bearer {settings.graph_key}"},
              json={"query": "{ __schema { queryType { name } } }"})
    else:
        report["sources"]["The Graph"] = {"status": "needs_configuration", "required": ["THEGRAPH_API_KEY", "THEGRAPH_SUBGRAPH_ID"],
                                          "note": "Aave deployment and schema remain to be verified in stage 3."}
    if settings.rpc_url or ETHEREUM_PUBLIC_RPC:
        probe("web3.py", "POST", settings.rpc_url or ETHEREUM_PUBLIC_RPC,
              lambda p: isinstance(p, dict) and p.get("result") == "0x1" and not p.get("error"),
              json={"jsonrpc": "2.0", "id": 1, "method": "eth_chainId", "params": []})
        report["sources"]["web3.py"]["note"] = "HTTP RPC probe only; actual web3.py collection evidence is in the stage 3 manifest."
    else:
        report["sources"]["web3.py"] = {"status": "needs_configuration", "required": ["ETHEREUM_RPC_URL"]}
    if settings.etherscan_key:
        probe("Etherscan", "GET", "https://api.etherscan.io/v2/api",
              lambda p: isinstance(p, dict) and isinstance(p.get("result"), str) and p["result"].startswith("0x") and not p.get("error"),
              params={"chainid": 1, "module": "proxy", "action": "eth_blockNumber", "apikey": settings.etherscan_key})
    else:
        report["sources"]["Etherscan"] = {"status": "needs_configuration", "required": ["ETHERSCAN_API_KEY"]}
    session.close()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


class CollectionError(RuntimeError):
    """Sanitized API failure with optional provider error code."""

    def __init__(self, status, code=None):
        self.status, self.code = status, code
        super().__init__(f"{status}; provider_code={code}")


def fetch_cached(session, name, url, params=None, headers=None, refresh=False, validator=None):
    """Cache validated JSON and provenance together; never persist auth headers."""
    path = RAW_DIR / f"{name}.json"
    signature = {"url": url, "params": params or {}}
    if path.exists() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("request") == signature:
            digest = hashlib.sha256(json.dumps(cached["data"], sort_keys=True).encode()).hexdigest()
            if digest != cached.get("sha256"):
                raise CollectionError("cache_integrity_error")
            return cached
    try:
        response = session.get(url, params=params, headers=headers, timeout=(5, 30))
        try:
            payload = response.json()
        except ValueError:
            raise CollectionError(f"http_{response.status_code}_non_json") from None
        if not response.ok or (isinstance(payload, dict) and ("error" in payload or "status" in payload)):
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            status = error.get("status", {}) if isinstance(error, dict) else {}
            raise CollectionError(f"http_{response.status_code}", status.get("error_code"))
    except requests.RequestException as exc:
        raise CollectionError(type(exc).__name__) from None
    if validator is not None:
        valid = validator(payload)
    elif "api.llama.fi" in url:
        valid = isinstance(payload, dict) and payload.get("isParentProtocol") is True and isinstance(payload.get("tvl"), list) and bool(payload["tvl"])
        valid = valid and all(isinstance(x, dict) and {"date", "totalLiquidityUSD"} <= x.keys() for x in payload["tvl"])
    else:
        valid = isinstance(payload, dict) and all(
            isinstance(payload.get(key), list) and payload[key] and all(
                isinstance(row, list) and len(row) == 2 for row in payload[key])
            for key in ["prices", "market_caps", "total_volumes"])
    if not valid:
        raise CollectionError("unexpected_schema")
    envelope = {"request": signature, "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), "data": payload}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
    return envelope


def collect_economics(refresh=False):
    """Collect only missing daily fee/revenue histories; retain provider definitions."""
    results = {}
    with make_session() as session:
        for name, data_type in [("fees", "dailyFees"), ("revenue", "dailyRevenue")]:
            try:
                envelope = fetch_cached(session, f"defillama_aave_{name}",
                    "https://api.llama.fi/summary/fees/aave",
                    params={"dataType": data_type, "excludeTotalDataChart": "false", "excludeTotalDataChartBreakdown": "true"},
                    refresh=refresh, validator=lambda p: isinstance(p, dict) and isinstance(p.get("totalDataChart"), list)
                    and bool(p["totalDataChart"]) and all(isinstance(r, list) and len(r) == 2 for r in p["totalDataChart"]))
                results[name] = {"status": "ok", "file": f"defillama_aave_{name}.json", "fetched_at_utc": envelope["fetched_at_utc"]}
            except CollectionError as exc:
                results[name] = {"status": "unavailable", "reason": str(exc)}
            print(name, results[name]["status"], flush=True)
    (RAW_DIR / "economics_manifest.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def collect_history(refresh=False):
    """Collect parent TVL and token history, documenting the 365-day fallback."""
    settings = Settings()
    start, end = settings.period
    manifest_path = RAW_DIR / "history_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    period = [start.isoformat(), end.isoformat()]
    manifest = {"period_utc_half_open": period, "research_as_of": settings.as_of.isoformat(),
                "run_at_utc": datetime.now(timezone.utc).isoformat(), "sources": {}}
    with make_session() as session:
        for slug in ["aave", "morpho", "compound-finance"]:
            try:
                name = f"defillama_{slug}"
                envelope = fetch_cached(session, name, f"https://api.llama.fi/protocol/{slug}", refresh=refresh)
                manifest["sources"][slug] = {"status": "ok", "file": name + ".json",
                    "fetched_at_utc": envelope["fetched_at_utc"], "scope": "parent protocol, default TVL"}
            except CollectionError as exc:
                manifest["sources"][slug] = {"status": "failed", "reason": str(exc)}
            print(slug, manifest["sources"][slug]["status"], flush=True)

        base = "https://pro-api.coingecko.com/api/v3" if settings.coingecko_tier == "pro" else "https://api.coingecko.com/api/v3"
        headers = {}
        if settings.coingecko_key:
            headers["x-cg-pro-api-key" if settings.coingecko_tier == "pro" else "x-cg-demo-api-key"] = settings.coingecko_key
        for coin in ["aave", "ethereum"]:
            source = f"coingecko_{coin}"
            url = base + f"/coins/{coin}/market_chart/range"
            params = {"vs_currency": "usd", "from": int(start.timestamp()), "to": int(end.timestamp()) - 1}
            note = None
            old = previous.get("sources", {}).get(source, {})
            if not refresh and previous.get("period_utc_half_open") == period and old.get("status") == "ok":
                old_path = RAW_DIR / old["file"]
                if old_path.exists():
                    cached = json.loads(old_path.read_text(encoding="utf-8"))
                    if cached.get("request", {}).get("url") == url:
                        params = cached["request"]["params"]
                        note = old.get("limitation")
            attempts = []
            try:
                try:
                    envelope = fetch_cached(session, source, url, params, headers, refresh)
                except CollectionError as exc:
                    attempts.append({"status": exc.status, "provider_code": exc.code,
                                     "requested_from": params["from"], "requested_to": params["to"]})
                    if exc.code != 10012:
                        raise
                    # Daily UTC observations start strictly inside the free rolling window.
                    cutoff = (datetime.now(timezone.utc) - timedelta(days=365)).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                    fallback = max(start, cutoff)
                    if fallback >= end:
                        raise CollectionError("no_overlap_with_365_day_window")
                    params = {**params, "from": int(fallback.timestamp())}
                    note = "Full period rejected with CoinGecko 10012; shortened to rolling 365-day availability, no imputation."
                    envelope = fetch_cached(session, source, url, params, headers, refresh)
                manifest["sources"][source] = {"status": "ok", "file": source + ".json",
                    "fetched_at_utc": envelope["fetched_at_utc"], "limitation": note,
                    "attempts": attempts or old.get("attempts", [])}
            except CollectionError as exc:
                manifest["sources"][source] = {"status": "failed", "reason": str(exc), "attempts": attempts}
            print(source, manifest["sources"][source]["status"], flush=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read-only source availability checks")
    parser.add_argument("--refresh", action="store_true", help="Replace the cached diagnostic snapshot")
    parser.add_argument("--history", action="store_true", help="Collect stage 2 historical data")
    args = parser.parse_args()
    print(json.dumps(collect_history(args.refresh) if args.history else check_sources(args.refresh), ensure_ascii=False, indent=2))
