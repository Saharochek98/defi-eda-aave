"""Required authenticated providers and optional public CSV comparison; no Dune API."""

import json
from pathlib import Path

import pandas as pd

from .config import ROOT, Settings
from .data_collection import make_session
from .onchain import (POOL, RAW, ABI, TOPICS, EVENT_COLUMNS, OnchainError,
                      save_snapshot, load_snapshot, decode_log, validate_events, SUBGRAPH_ID)


def api_json(session, method, url, **kwargs):
    try:
        response = session.request(method, url, timeout=(5, 30), **kwargs)
        if not response.ok:
            raise OnchainError(f"http_{response.status_code}")
        return response.json()
    except OnchainError:
        raise
    except Exception as exc:
        raise OnchainError(type(exc).__name__) from None


def collect_etherscan(rpc_snapshot, start, end, refresh=False):
    settings = Settings()
    if not settings.etherscan_key:
        return None
    first, last = (rpc_snapshot["data"][key] for key in ["first_block", "last_block"])
    request = {"chain_id": 1, "pool": POOL.lower(), "first_block": first, "last_block": last, "topics": list(TOPICS)}
    cached = None if refresh else load_snapshot("etherscan_events", request)
    if cached:
        return cached
    raw_cached = None if refresh else load_snapshot("etherscan_raw_events", request)
    if raw_cached:
        logs, pages = raw_cached["data"]["logs"], raw_cached["data"]["pages"]
        return finish_etherscan(request, logs, pages, rpc_snapshot, start, end)
    with make_session() as session:
        logs, pages = [], []
        for topic in TOPICS:
            for page in range(1, 51):
                params = {"chainid": 1, "module": "logs", "action": "getLogs", "address": POOL,
                          "fromBlock": first, "toBlock": last, "topic0": topic, "page": page,
                          "offset": 1000, "apikey": settings.etherscan_key}
                payload = api_json(session, "GET", "https://api.etherscan.io/v2/api", params=params)
                if payload.get("status") == "0" and payload.get("message") == "No records found":
                    rows = []
                elif payload.get("status") == "1" and isinstance(payload.get("result"), list):
                    rows = payload["result"]
                else:
                    raise OnchainError("etherscan_error_response")
                pages.append({"topic": topic, "page": page, "rows": len(rows)})
                logs.extend(rows)
                if len(rows) < 1000:
                    break
            else:
                raise OnchainError("etherscan_pagination_limit")
    save_snapshot("etherscan_raw_events", request, {"logs": logs, "pages": pages})
    return finish_etherscan(request, logs, pages, rpc_snapshot, start, end)


def finish_etherscan(request, logs, pages, rpc_snapshot, start, end):
    for log in logs:
        number = str(int(log["blockNumber"], 16))
        block = rpc_snapshot["data"]["blocks"].get(number)
        if not block or log["blockHash"].lower() != block["hash"]:
            raise OnchainError("etherscan_log_block_hash_mismatch")
    decoded = [decode_log(log, int(log["timeStamp"], 16)) for log in logs]
    frame, duplicates = validate_events(pd.DataFrame(decoded, columns=EVENT_COLUMNS), start, end)
    if duplicates:
        raise OnchainError("etherscan_duplicate_logs_or_ignored_pagination")
    return save_snapshot("etherscan_events", request, {"logs": logs, "decoded": decoded, "pages": pages})


def collect_dune(start, end, csv_path=None, refresh=False):
    if not csv_path:
        return None
    request = {"chain_id": 1, "pool": POOL.lower(), "start_utc": start.isoformat(), "end_exclusive_utc": end.isoformat()}
    path = Path(csv_path).resolve(strict=True)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    validated, duplicates = validate_events(frame, start, end)
    rows = validated.to_dict("records")
    for row in rows:
        row["block_time"] = row["block_time"].isoformat()
    import hashlib
    return save_snapshot("dune_events", {**request, "method": "public_csv", "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                         {"decoded": rows, "duplicates": duplicates, "query_executed_by_collector": False})


def collect_graph(rpc_snapshot, start, end, refresh=False):
    settings = Settings()
    if not settings.graph_key:
        return None
    subgraph = settings.graph_subgraph or SUBGRAPH_ID
    block = rpc_snapshot["data"]["last_block"]
    request = {"subgraph_id": subgraph, "block": block, "start": int(start.timestamp()), "end": int(end.timestamp())}
    cached = None if refresh else load_snapshot("graph_markets", request)
    if cached:
        return cached
    url = f"https://gateway.thegraph.com/api/subgraphs/id/{subgraph}"
    headers = {"Authorization": f"Bearer {settings.graph_key}"}
    with make_session() as session:
        def gql(query, variables=None):
            payload = api_json(session, "POST", url, headers=headers, json={"query": query, "variables": variables or {}})
            if payload.get("errors") or not isinstance(payload.get("data"), dict):
                raise OnchainError("graphql_error_response")
            return payload["data"]
        schema = gql("{ __schema { queryType { fields { name } } types { name fields { name } } } _meta { deployment hasIndexingErrors block { number hash } } }")
        if schema["_meta"]["hasIndexingErrors"] or schema["_meta"]["block"]["number"] < block:
            raise OnchainError("subgraph_indexing_errors_or_behind_requested_block")
        roots = {f["name"] for f in schema["__schema"]["queryType"]["fields"]}
        required_roots = {"reserves", "reserveParamsHistoryItems"}
        if not required_roots <= roots:
            raise OnchainError("unexpected_live_graphql_root_schema")
        types = {t["name"]: {f["name"] for f in t.get("fields") or []} for t in schema["__schema"]["types"]}
        for typ, required in {
            "Reserve": {"id", "underlyingAsset", "symbol", "decimals", "totalATokenSupply", "totalCurrentVariableDebt", "totalPrincipalStableDebt", "availableLiquidity", "liquidityRate", "variableBorrowRate", "utilizationRate", "lastUpdateTimestamp", "pool"},
            "ReserveParamsHistoryItem": {"id", "timestamp", "reserve", "totalATokenSupply", "totalCurrentVariableDebt", "totalPrincipalStableDebt", "availableLiquidity", "utilizationRate", "liquidityRate", "variableBorrowRate", "liquidityIndex", "variableBorrowIndex"},
            "Pool": {"id", "pool"},
        }.items():
            if not required <= types.get(typ, set()):
                raise OnchainError(f"unexpected_live_graphql_fields_{typ}")
        results = {}
        for name, root in [("aave_v3_reserves", "reserves"), ("aave_v3_reserve_history", "reserveParamsHistoryItems")]:
            query = (ROOT / "src" / "graphql" / f"{name}.graphql").read_text(encoding="utf-8")
            records, after = [], ""
            for page in range(100):
                variables = {"block": block, "after": after}
                if root != "reserves":
                    variables.update(start=request["start"], end=request["end"])
                data = gql(query, variables)
                batch = data[root]
                if root == "reserves" and (data["_meta"]["hasIndexingErrors"] or data["_meta"]["block"]["number"] != block):
                    raise OnchainError("historical_subgraph_meta_mismatch")
                if batch and batch[-1]["id"] <= after:
                    raise OnchainError("graph_cursor_not_advancing")
                records.extend(batch)
                if len(batch) < 1000:
                    break
                after = batch[-1]["id"]
            else:
                raise OnchainError("graph_pagination_limit")
            if root == "reserves":
                records = [r for r in records if (r["pool"].get("pool") or "").lower() == POOL.lower()]
            else:
                records = [r for r in records if (r["reserve"]["pool"].get("pool") or "").lower() == POOL.lower()]
            results[root] = records
        if not results["reserves"]:
            raise OnchainError("no_reserves_for_expected_pool")
    return save_snapshot("graph_markets", request, {"schema": schema, **results})
