# Aave DeFi Exploratory Data Analysis

## Project overview

A Python research project examining historical Aave data and comparing Aave with Morpho and Compound. The analysis covers **1 November 2025–31 October 2026**, with explicitly narrower market and on-chain samples. It includes a standalone executed [Jupyter notebook](DeFi_EDA.ipynb) and the preserved, approved [PDF report](reports/DeFi_EDA_Report.pdf). Analytical commentary in these deliverables is in Russian.

## Research questions

- How did Aave's TVL and share of the selected lending protocols change?
- How are TVL changes associated with fees, revenue and AAVE returns?
- Do correlations survive rank-based and outlier sensitivity checks?
- Are indexed reserve balances consistent with historical Ethereum contract state?

## Data sources

| Tool | Actual use |
| --- | --- |
| DeFiLlama | Daily protocol/chain TVL, Aave fees and protocol revenue |
| CoinGecko | AAVE/ETH prices, market capitalization and 24-hour volume |
| The Graph | Aave V3 Ethereum reserve snapshot and 30-day reserve observations |
| Etherscan API V2 | Independent Supply/Borrow/Repay log extraction |
| web3.py / Ethereum RPC | Event extraction and historical USDC contract calls |

The notebook embeds fixed research inputs, source URLs, collection timestamps, cache hashes and historical RPC responses. It does not claim to download fresh data when rerun. Dune is not required.

## Main findings

- Aave TVL fell from **$37.27bn to $16.04bn (−56.97%)**. Its TVL share within the selected trio declined from **77.26% to 56.35%**.
- Aave recorded **$636.23m in fees** and **$90.14m in protocol revenue** over the full research window.
- AAVE returned **−21.65%** over 345 price observations; annualized daily-return volatility was **83.52%**.
- AAVE/ETH correlation remained positive after trimming extremes: Pearson **0.709 → 0.654**. TVL/fees correlation on the full aligned sample weakened from **−0.180 to −0.007**; a robust negative relationship was not established.
- RPC and Etherscan matched **3,260 events** for 31 October 2026. The Graph contained **25 invalid USDC reserve-day observations**. RPC checks on three matching historical blocks showed positive cash balances where the subgraph reported negative liquidity. Original values and quality flags were retained; the precise source of the discrepancy remains unresolved. See the [USDC check](reports/USDC_CHECK_REPORT.md).

**Report note:** the approved PDF is preserved unchanged. Its page 6 correlation table has the correct negative signs; the surrounding prose omits the minus signs for −0.180 and −0.007. The signed results above and in the notebook are authoritative.

## Repository structure

```text
DeFi_EDA.ipynb                 # Standalone notebook with 11 saved figures
reports/
  DeFi_EDA_Report.pdf          # Approved 10-page report
  USDC_CHECK_REPORT.md         # Independent historical-state check
src/                          # Collectors, processing, metrics and plotting
  abi/                        # Aave event ABI
  graphql/                    # Reserve queries
tests/                        # Numerical, data-quality and source checks
requirements.txt
.env.example
.gitignore
```

Stage modules remain where they implement collection or are dependencies of the tested analysis. Intermediate notebooks, caches and earlier reports are excluded. The obsolete PDF builder is not published, preventing accidental replacement of the approved edition.

## Installation and usage

Use Python **3.13** (the tested version):

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab DeFi_EDA.ipynb
```

Run all notebook cells. No API keys, external datasets or `src/` imports are needed for this fixed snapshot. Outputs are written to `eda_output/`, which Git ignores. The notebook is also readable without executing it.

```bash
python -m unittest discover -s tests -v
```

For live collection, copy `.env.example` to `.env` and configure The Graph, Etherscan and optionally CoinGecko/RPC access locally. Never commit credentials.

```bash
python -m src.data_collection --refresh
python -m src.data_collection --history
```

`src/onchain.py` and `src/onchain_sources.py` implement bounded RPC/Graph/Etherscan collection; `src/stage3.py` orchestrates it. Stage entry points require their preceding local caches and are not the recommended way to replay the published snapshot. Provider history limits may prevent collecting the same period again. No local caches are required by the final notebook.

## Limitations

CoinGecko lacks 1–20 November 2025 in all six market series; values are not filled. TVL and flows can have different version/network coverage and imperfect timestamp synchronization. The event sample covers one UTC day; reserve history covers three assets over 30 days in one Ethereum Pool. USDC anomalies are not repaired or extrapolated from the three RPC checks. Addresses are not users, USD-TVL changes are not necessarily capital flows, and correlation does not establish causation.
