"""Project paths, credentials, and UTC research boundaries."""

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
load_dotenv(ROOT / ".env", override=False)

REQUIRED_SOURCES = ("DeFiLlama", "CoinGecko", "The Graph", "Etherscan", "web3.py")
AAVE_ETHEREUM_SUBGRAPH_ID = "Cd2gEDVeqnjBn1hSeqFMitw8Q1iiyV9FYUZkLNRcL87g"
ETHEREUM_PUBLIC_RPC = "https://eth.drpc.org"


def required_sources_verified(instruments) -> bool:
    """Optional sources never replace any of the five mandatory providers."""
    return set(REQUIRED_SOURCES).issubset(instruments)


def research_period(as_of: date) -> tuple[datetime, datetime]:
    """Return 12 full calendar months as a half-open UTC interval."""
    end = datetime(as_of.year, as_of.month, 1, tzinfo=timezone.utc)
    start = end.replace(year=end.year - 1)
    return start, end


def local_today() -> date:
    # Fixed UTC+3 avoids a Windows dependency on IANA timezone databases.
    from datetime import timedelta

    return datetime.now(timezone(timedelta(hours=3))).date()


@dataclass(frozen=True)
class Settings:
    as_of: date = field(default_factory=lambda: date.fromisoformat(
        os.getenv("RESEARCH_AS_OF") or local_today().isoformat()))
    coingecko_key: str = field(default_factory=lambda: os.getenv("COINGECKO_API_KEY", ""), repr=False)
    coingecko_tier: str = field(default_factory=lambda: os.getenv("COINGECKO_API_TIER", "demo"))
    graph_key: str = field(default_factory=lambda: os.getenv("THEGRAPH_API_KEY", ""), repr=False)
    graph_subgraph: str = field(default_factory=lambda: os.getenv("THEGRAPH_SUBGRAPH_ID") or AAVE_ETHEREUM_SUBGRAPH_ID)
    rpc_url: str = field(default_factory=lambda: os.getenv("ETHEREUM_RPC_URL", ""), repr=False)
    etherscan_key: str = field(default_factory=lambda: os.getenv("ETHERSCAN_API_KEY", ""), repr=False)

    def __post_init__(self):
        if self.coingecko_tier not in {"demo", "pro"}:
            raise ValueError("COINGECKO_API_TIER must be demo or pro")

    @property
    def period(self) -> tuple[datetime, datetime]:
        return research_period(self.as_of)
