"""Unit fixtures are isolated from all collected research data."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from web3 import Web3

from src.onchain import (ABI, TOPICS, POOL, EVENT_COLUMNS, decode_log, validate_events,
                         daily_activity, compare_event_sets, OnchainError, save_snapshot, load_snapshot)


class OnchainTests(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2026, 10, 31, tzinfo=timezone.utc)
        self.end = self.start+timedelta(days=1)
        self.values = {"reserve": "0x"+"11"*20, "user": "0x"+"22"*20,
                       "onBehalfOf": "0x"+"33"*20, "repayer": "0x"+"44"*20,
                       "amount": 2**200+13, "referralCode": 0, "interestRateMode": 2,
                       "borrowRate": 10**25, "useATokens": False}

    def event(self, name="Supply", log_index=7, zero_hex=False):
        abi = next(a for a in ABI if a["name"] == name)
        topic = next(k for k, a in TOPICS.items() if a["name"] == name)
        codec = Web3().codec
        topics = [topic] + [Web3.to_hex(codec.encode([p["type"]], [self.values[p["name"]]]))
                            for p in abi["inputs"] if p["indexed"]]
        data_inputs = [p for p in abi["inputs"] if not p["indexed"]]
        log = {"address": POOL, "topics": topics,
               "data": Web3.to_hex(codec.encode([p["type"] for p in data_inputs], [self.values[p["name"]] for p in data_inputs])),
               "blockNumber": 100, "blockHash": "0x"+"aa"*32, "transactionHash": "0x"+"bb"*32,
               "transactionIndex": 1, "logIndex": log_index, "removed": False}
        if zero_hex:
            log["transactionIndex"] = "0x"
            log["logIndex"] = "0x"
        return decode_log(log, int(self.start.timestamp())+12)

    def test_etherscan_empty_hex_quantity_is_zero(self):
        self.assertEqual(self.event(zero_hex=True)["log_index"], 0)

    def validated(self, rows):
        return validate_events(pd.DataFrame(rows, columns=EVENT_COLUMNS), self.start, self.end)

    def test_three_real_abi_shapes_decode(self):
        for name in ["Supply", "Borrow", "Repay"]:
            with self.subTest(event=name):
                row = self.event(name)
                self.assertEqual(row["amount_raw"], str(2**200+13))
                self.assertEqual(row["event_name"], name)
                self.assertEqual(row["reserve"], self.values["reserve"])
                self.assertEqual(row["user"], self.values["user"])
                field = "repayer" if name == "Repay" else "on_behalf_of"
                self.assertTrue(row[field])

    def test_conflicting_identity_rejected(self):
        row = self.event()
        with self.assertRaisesRegex(OnchainError, "conflicting_event_identity"):
            self.validated([row, {**row, "amount_raw": "1"}])

    def test_identical_duplicate_counted(self):
        row = self.event()
        frame, dup = self.validated([row, row])
        self.assertEqual(len(frame), 1)
        self.assertEqual(dup, 1)

    def test_wrong_pool_rejected(self):
        with self.assertRaisesRegex(OnchainError, "wrong_chain_or_pool"):
            self.validated([{**self.event(), "pool_address": "0x"+"00"*20}])

    def test_exclusive_end_boundary(self):
        with self.assertRaisesRegex(OnchainError, "outside_requested_interval"):
            self.validated([{**self.event(), "block_time": self.end.isoformat()}])

    def test_no_float_rounding_of_uint256(self):
        frame, _ = self.validated([self.event()])
        self.assertEqual(frame.amount_raw.iloc[0], str(2**200+13))
        with self.assertRaisesRegex(OnchainError, "invalid_raw_amount"):
            self.validated([{**self.event(), "amount_raw": "1.5"}])

    def test_participants_are_union_across_events(self):
        frame, _ = self.validated([self.event("Supply", 7), self.event("Borrow", 8), self.event("Repay", 9)])
        counts = daily_activity(frame, self.start, self.end)
        self.assertEqual(counts.iloc[0]["active_participant_addresses"], 3)
        self.assertEqual(counts.iloc[0]["unique_transactions"], 1)
        self.assertEqual(int(counts[["Supply", "Borrow", "Repay"]].sum().sum()), 3)

    def test_equal_counts_with_different_amounts_do_not_match(self):
        left, _ = self.validated([self.event()])
        right, _ = self.validated([{**self.event(), "amount_raw": "1"}])
        comparison = compare_event_sets(left, right)
        self.assertFalse(comparison["exact_match"])
        self.assertEqual(comparison["value_mismatches"], 1)

    def test_different_log_identity_detected(self):
        left, _ = self.validated([self.event(log_index=7)])
        right, _ = self.validated([self.event(log_index=8)])
        comparison = compare_event_sets(left, right)
        self.assertEqual(comparison["only_left"], 1)
        self.assertEqual(comparison["only_right"], 1)

    def test_snapshot_integrity(self):
        with tempfile.TemporaryDirectory() as folder, patch("src.onchain.RAW", Path(folder)):
            save_snapshot("fixture", {"a": 1}, {"b": 2})
            self.assertEqual(load_snapshot("fixture")["data"], {"b": 2})
            path = Path(folder) / "fixture.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["data"]["b"] = 3
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(OnchainError, "cache_integrity_error"):
                load_snapshot("fixture")


if __name__ == "__main__":
    unittest.main()
