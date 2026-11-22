"""Bounded Aave V3 event collection and strict cross-source reconciliation."""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from eth_utils import event_abi_to_log_topic
from hexbytes import HexBytes
from web3 import Web3, HTTPProvider
from web3._utils.events import get_event_data

from .config import ROOT, RAW_DIR, PROCESSED_DIR, Settings
from .data_collection import make_session

POOL = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
SUBGRAPH_ID = "Cd2gEDVeqnjBn1hSeqFMitw8Q1iiyV9FYUZkLNRcL87g"
PUBLIC_RPC = "https://eth.drpc.org"
SECOND_RPC = "https://1rpc.io/eth"
RAW = RAW_DIR / "onchain"
OUT = PROCESSED_DIR / "onchain"
ABI = json.loads((ROOT / "src" / "abi" / "aave_v3_pool_events.json").read_text(encoding="utf-8"))
TOPICS = {Web3.to_hex(event_abi_to_log_topic(a)).lower(): a for a in ABI}
EVENT_COLUMNS = ["chain_id", "pool_address", "block_number", "block_time", "tx_hash", "log_index",
                 "event_name", "reserve", "user", "on_behalf_of", "repayer", "amount_raw"]
KEY = ["tx_hash", "log_index"]


class OnchainError(RuntimeError):
    """Only sanitized diagnostics may leave the collector."""


def hex_value(value):
    if isinstance(value, str) and value.startswith("0x"):
        return value.lower()
    return Web3.to_hex(value).lower()


def json_safe(value):
    if isinstance(value, (bytes, HexBytes)):
        return hex_value(value)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def save_snapshot(name, request, data):
    RAW.mkdir(parents=True, exist_ok=True)
    normalized = json_safe(data)
    envelope = {"request": request, "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest(), "data": normalized}
    (RAW / f"{name}.json").write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
    return envelope


def load_snapshot(name, request=None):
    path = RAW / f"{name}.json"
    if not path.exists():
        return None
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if request is not None and envelope["request"] != request:
        return None
    digest = hashlib.sha256(json.dumps(envelope["data"], sort_keys=True).encode()).hexdigest()
    if digest != envelope["sha256"]:
        raise OnchainError("cache_integrity_error")
    return envelope


def get_rpc(url):
    # Disable provider retries; the session performs one bounded retry instead.
    return Web3(HTTPProvider(url, session=make_session(), request_kwargs={"timeout": 20}, exception_retry_configuration=None))


def block_at_or_after(w3, timestamp):
    """Binary search with the first block at or after a UTC boundary."""
    low, high = 0, w3.eth.block_number
    if w3.eth.get_block(high)["timestamp"] < timestamp:
        raise OnchainError("requested_boundary_after_chain_head")
    while low < high:
        mid = (low + high) // 2
        if w3.eth.get_block(mid)["timestamp"] < timestamp:
            low = mid + 1
        else:
            high = mid
    return low


def decode_log(log, timestamp):
    normalized = dict(log)
    for field in ["data", "blockHash", "transactionHash"]:
        normalized[field] = HexBytes(normalized[field])
    normalized["topics"] = [HexBytes(v) for v in normalized["topics"]]
    for field in ["blockNumber", "transactionIndex", "logIndex"]:
        v = normalized[field]
        # Etherscan encodes a zero quantity as either 0x0 or the empty 0x.
        normalized[field] = (0 if v == "0x" else int(v, 16)) if isinstance(v, str) else int(v)
    if normalized["address"].lower() != POOL.lower() or normalized.get("removed", False):
        raise OnchainError("unexpected_pool_or_removed_log")
    topic = hex_value(normalized["topics"][0])
    if topic not in TOPICS:
        raise OnchainError("unexpected_event_topic")
    event = get_event_data(Web3().codec, TOPICS[topic], normalized)
    args = event["args"]
    return {"chain_id": 1, "pool_address": POOL.lower(), "block_number": normalized["blockNumber"],
            "block_time": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
            "tx_hash": hex_value(normalized["transactionHash"]), "log_index": normalized["logIndex"],
            "event_name": event["event"], "reserve": args["reserve"].lower(), "user": args["user"].lower(),
            "on_behalf_of": args.get("onBehalfOf", "").lower(), "repayer": args.get("repayer", "").lower(),
            "amount_raw": str(args["amount"])}


def validate_events(frame, start, end):
    """Validate one pool and complete rows, keeping uint256 amounts as strings."""
    if not set(EVENT_COLUMNS) <= set(frame.columns):
        raise OnchainError("event_columns_missing")
    frame = frame[EVENT_COLUMNS].copy().fillna("")
    for col in ["chain_id", "block_number", "log_index"]:
        values = pd.to_numeric(frame[col], errors="raise")
        if values.isna().any() or (values % 1 != 0).any() or (values < 0).any():
            raise OnchainError("invalid_integer_event_field")
        frame[col] = values.astype("int64")
    if not frame.chain_id.eq(1).all() or not frame.pool_address.str.lower().eq(POOL.lower()).all():
        raise OnchainError("wrong_chain_or_pool")
    frame["block_time"] = pd.to_datetime(frame.block_time, utc=True, errors="raise")
    if frame.block_time.isna().any() or not ((frame.block_time >= start) & (frame.block_time < end)).all():
        raise OnchainError("events_outside_requested_interval")
    if not frame.event_name.isin(["Supply", "Borrow", "Repay"]).all():
        raise OnchainError("unsupported_event")
    for col in ["pool_address", "reserve", "user", "on_behalf_of", "repayer", "tx_hash"]:
        frame[col] = frame[col].astype(str).str.lower()
    for col in ["pool_address", "reserve", "user"]:
        if not frame[col].str.fullmatch(r"0x[0-9a-f]{40}").all():
            raise OnchainError("invalid_address")
    for event, col in [("Supply", "on_behalf_of"), ("Borrow", "on_behalf_of"), ("Repay", "repayer")]:
        if not frame.loc[frame.event_name.eq(event), col].str.fullmatch(r"0x[0-9a-f]{40}").all():
            raise OnchainError("missing_participant_address")
    if not frame.tx_hash.str.fullmatch(r"0x[0-9a-f]{64}").all():
        raise OnchainError("invalid_transaction_hash")
    if not frame.amount_raw.astype(str).str.fullmatch(r"\d+").all():
        raise OnchainError("invalid_raw_amount")
    if frame.amount_raw.map(lambda v: int(v) >= 2**256).any():
        raise OnchainError("amount_exceeds_uint256")
    frame["amount_raw"] = frame.amount_raw.map(lambda v: str(int(v)))
    duplicate_count = int(frame.duplicated(KEY).sum())
    unique = frame.drop_duplicates()
    if unique.duplicated(KEY).any():
        raise OnchainError("conflicting_event_identity")
    return unique.sort_values(["block_number", "log_index"]).reset_index(drop=True), duplicate_count


def collect_rpc(start, end, refresh=False):
    request = {"chain_id": 1, "pool": POOL.lower(), "start_utc": start.isoformat(), "end_exclusive_utc": end.isoformat(),
               "topics": list(TOPICS)}
    cached = None if refresh else load_snapshot("web3_events", request)
    if cached:
        return cached
    settings = Settings()
    w3 = get_rpc(settings.rpc_url or PUBLIC_RPC)
    if w3.eth.chain_id != 1:
        raise OnchainError("rpc_is_not_ethereum_mainnet")
    boundary_cache = None if refresh else load_snapshot("rpc_boundaries", request)
    if boundary_cache:
        first, stop, boundary = (boundary_cache["data"][k] for k in ["first", "stop", "blocks"])
    else:
        first = block_at_or_after(w3, int(start.timestamp()))
        stop = block_at_or_after(w3, int(end.timestamp()))
        # Verify both half-open boundaries; never trust an estimated average block time.
        boundary = {str(n): json_safe(w3.eth.get_block(n)) for n in {first-1, first, stop-1, stop}}
        save_snapshot("rpc_boundaries", request, {"first": first, "stop": stop, "blocks": boundary})
    if not (boundary[str(first-1)]["timestamp"] < start.timestamp() <= boundary[str(first)]["timestamp"] and
            boundary[str(stop-1)]["timestamp"] < end.timestamp() <= boundary[str(stop)]["timestamp"]):
        raise OnchainError("invalid_block_boundaries")
    if not w3.eth.get_code(POOL):
        raise OnchainError("pool_has_no_current_bytecode")
    progress = None if refresh else load_snapshot("rpc_log_progress", request)
    logs = progress["data"]["logs"] if progress else []
    chunks = progress["data"]["chunks"] if progress else []
    resume = chunks[-1]["last_block"] + 1 if chunks else first
    for low in range(resume, stop, 100):
        high = min(low + 99, stop-1)
        batch = w3.eth.get_logs({"address": POOL, "fromBlock": low, "toBlock": high, "topics": [list(TOPICS)]})
        logs.extend(batch)
        chunks.append({"first_block": low, "last_block": high, "log_count": len(batch)})
        save_snapshot("rpc_log_progress", request, {"logs": json_safe(logs), "chunks": chunks})
        print(f"RPC blocks {low}..{high}: {len(batch)} logs", flush=True)
    numbers = sorted({log["blockNumber"] for log in logs})
    blocks = {}
    # Some providers return a real blockTimestamp extension in eth_getLogs.
    # Use it when present, with per-block consistency and sampled header checks.
    for log in logs:
        if log.get("blockTimestamp") is None:
            continue
        n = log["blockNumber"]
        value = log["blockTimestamp"]
        timestamp = int(value, 16) if isinstance(value, str) else int(value)
        header = {"number": n, "timestamp": timestamp, "hash": hex_value(log["blockHash"])}
        if str(n) in blocks and blocks[str(n)] != header:
            raise OnchainError("inconsistent_log_block_metadata")
        blocks[str(n)] = header
    for n in numbers:
        if str(n) not in blocks:
            block = w3.eth.get_block(n)
            blocks[str(n)] = {"number": n, "timestamp": block["timestamp"], "hash": hex_value(block["hash"])}
    sampled = [numbers[i] for i in sorted({0, len(numbers)//2, len(numbers)-1})] if numbers else []
    for n in sampled:
        header = w3.eth.get_block(n)
        if header["timestamp"] != blocks[str(n)]["timestamp"] or hex_value(header["hash"]) != blocks[str(n)]["hash"]:
            raise OnchainError("log_timestamp_does_not_match_block_header")
    if len(blocks) != len(numbers):
        raise OnchainError("block_header_batch_missing_ids")
    if any(hex_value(log["blockHash"]) != blocks[str(log["blockNumber"])]["hash"] for log in logs):
        raise OnchainError("primary_rpc_log_block_hash_mismatch")
    decoded = [decode_log(log, blocks[str(log["blockNumber"])]["timestamp"]) for log in logs]
    validated, duplicates = validate_events(pd.DataFrame(decoded, columns=EVENT_COLUMNS), start, end)
    if duplicates:
        raise OnchainError("duplicate_rpc_logs")
    data = {"first_block": first, "last_block": stop-1, "boundary_blocks": boundary, "blocks": blocks,
            "chunks": chunks, "logs": json_safe(logs), "decoded": json_safe(validated.to_dict("records")),
            "sampled_header_checks": sampled,
            "timestamp_source": "RPC blockTimestamp extension when present; otherwise eth_getBlockByNumber",
            "rpc_provider": "configured (URL redacted)" if settings.rpc_url else PUBLIC_RPC}
    # Timestamps need ISO strings rather than pandas objects for portable JSON.
    for row in data["decoded"]:
        row["block_time"] = row["block_time"].isoformat()
    return save_snapshot("web3_events", request, data)


def compare_event_sets(left, right):
    """Compare identities AND values; equal counts alone do not prove agreement."""
    l = left.set_index(KEY)
    r = right.set_index(KEY)
    common = l.index.intersection(r.index)
    fields = ["chain_id", "pool_address", "block_number", "block_time", "event_name", "reserve", "user", "on_behalf_of", "repayer", "amount_raw"]
    mismatches = sum(not l.loc[k, fields].equals(r.loc[k, fields]) for k in common)
    result = {"left_events": len(l), "right_events": len(r), "common_events": len(common),
              "only_left": len(l.index.difference(r.index)), "only_right": len(r.index.difference(l.index)),
              "value_mismatches": mismatches}
    result["exact_match"] = not any(result[k] for k in ["only_left", "only_right", "value_mismatches"])
    return result


def verify_secondary_rpc(snapshot, start, end, refresh=False):
    request = {"pool": POOL.lower(), "first_block": snapshot["data"]["first_block"],
               "last_block": snapshot["data"]["last_block"], "topics": list(TOPICS), "rpc_provider": SECOND_RPC}
    cached = None if refresh else load_snapshot("secondary_rpc_events", request)
    if cached:
        return cached
    w3 = get_rpc(SECOND_RPC)
    if w3.eth.chain_id != 1:
        raise OnchainError("secondary_rpc_wrong_chain")
    logs = []
    for low in range(request["first_block"], request["last_block"] + 1, 50):
        logs.extend(w3.eth.get_logs({"address": POOL, "fromBlock": low, "toBlock": min(low+49, request["last_block"]), "topics": [list(TOPICS)]}))
    # Use block hashes to verify this is the same canonical historical segment.
    primary_blocks = snapshot["data"]["blocks"]
    for log in logs:
        if hex_value(log["blockHash"]) != primary_blocks[str(log["blockNumber"])]["hash"]:
            raise OnchainError("secondary_rpc_block_hash_mismatch")
    decoded = [decode_log(log, primary_blocks[str(log["blockNumber"])]["timestamp"]) for log in logs]
    frame, dup = validate_events(pd.DataFrame(decoded, columns=EVENT_COLUMNS), start, end)
    for row in decoded:
        row["block_time"] = pd.Timestamp(row["block_time"]).isoformat()
    return save_snapshot("secondary_rpc_events", request, {"logs": json_safe(logs), "decoded": decoded, "duplicates": dup})


def daily_activity(frame, start, end):
    """Counts are event counts, not transaction counts or token volumes."""
    calendar = pd.date_range(start, end, inclusive="left", freq="D", name="date")
    events = frame.assign(date=frame.block_time.dt.floor("D"))
    counts = events.groupby(["date", "event_name"]).size().unstack(fill_value=0).reindex(columns=["Supply", "Borrow", "Repay"], fill_value=0)
    counts = counts.reindex(calendar, fill_value=0)
    users = []
    for row in events.to_dict("records"):
        people = {row["user"], row["repayer"] if row["event_name"] == "Repay" else row["on_behalf_of"]}
        users.extend((row["date"], p) for p in people if p and p != "0x" + "0"*40)
    participants = pd.DataFrame(users, columns=["date", "address"])
    counts["active_participant_addresses"] = participants.groupby("date").address.nunique().reindex(calendar, fill_value=0)
    counts["unique_transactions"] = events.groupby("date").tx_hash.nunique().reindex(calendar, fill_value=0)
    return counts
