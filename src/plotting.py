"""Initial research figures with explicit dates and units."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timezone

from .config import PROCESSED_DIR

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.figsize": (11, 4.5),
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120})


def make_initial_plots(protocol, tokens, common):
    figures = []

    def finish(fig, name):
        for ax in fig.axes:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m", tz=timezone.utc))
        fig.tight_layout()
        folder = PROCESSED_DIR / "figures"
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.png", bbox_inches="tight")
        figures.append((name, fig))

    if "aave_tvl_usd" in protocol and protocol["aave_tvl_usd"].notna().any():
        fig, ax = plt.subplots()
        (protocol["aave_tvl_usd"] / 1e9).plot(ax=ax, color="#4472c4", linewidth=2)
        ax.set(title="TVL Aave — агрегированный протокол", xlabel="Дата (UTC)", ylabel="TVL, млрд USD")
        finish(fig, "aave_tvl")
    if len(protocol.columns) == 3 and len(protocol.dropna()):
        fig, ax = plt.subplots()
        comparison = protocol.loc[protocol.dropna().index] / 1e9
        comparison.rename(columns={"aave_tvl_usd": "Aave", "morpho_tvl_usd": "Morpho", "compound_tvl_usd": "Compound"}).plot(ax=ax)
        ax.set(title="TVL Aave, Morpho и Compound — общий охват дат", xlabel="Дата (UTC)", ylabel="TVL, млрд USD")
        ax.legend(title="Родительский протокол")
        finish(fig, "protocol_comparison")
    if len(common):
        fig, ax = plt.subplots()
        prices = common[["aave_price_usd", "eth_price_usd"]].rename(columns={"aave_price_usd": "AAVE", "eth_price_usd": "ETH"})
        (prices / prices.iloc[0] * 100).plot(ax=ax)
        ax.set(title="AAVE и ETH — цены на общем интервале", xlabel="Дата (UTC)", ylabel="Индекс цены, первая общая дата = 100")
        ax.legend(title="Токен")
        finish(fig, "token_prices_normalized")
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        (common["aave_market_cap_usd"] / 1e9).plot(ax=axes[0], color="#7c4d9a")
        axes[0].set(title="AAVE — капитализация и скользящий 24-часовой объём", ylabel="Капитализация, млрд USD")
        (common["aave_volume_24h_usd"] / 1e9).plot(ax=axes[1], color="#d97732")
        axes[1].set(xlabel="Дата (UTC)", ylabel="Объём за 24 ч, млрд USD")
        finish(fig, "aave_market_data")
    return figures


def make_eda_plots(b):
    """Ten publication-size figures, using only actual cached observations."""
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from .metrics import EDA_DIR

    sns.set_theme(style="whitegrid", font="DejaVu Sans", palette="colorblind",
                  rc={"grid.alpha": .22, "axes.titlesize": 12, "axes.labelsize": 10})
    output = EDA_DIR / "figures"
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    p, t, e = b["protocol"], b["tokens"], b["economics"]

    def finish(fig, name, dated=()):
        for ax in dated:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m", tz=timezone.utc))
            ax.set_xlabel("Дата (UTC)")
        fig.tight_layout()
        path = output / f"{name}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths[name] = path

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    (p.aave_tvl_usd/1e9).plot(ax=ax[0], color="#4472c4", lw=2)
    ax[0].set(title="Aave: TVL и изменения за 30/90 календарных дней", ylabel="TVL, млрд USD")
    (p[["tvl_change_30d", "tvl_change_90d"]]*100).rename(columns={"tvl_change_30d": "30 дней", "tvl_change_90d": "90 дней"}).plot(ax=ax[1])
    ax[1].axhline(0, color="gray", lw=.8)
    ax[1].set_ylabel("Изменение TVL, %")
    finish(fig, "01_tvl", ax)

    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    trio = p[["aave_tvl_usd", "morpho_tvl_usd", "compound_tvl_usd"]].rename(columns={"aave_tvl_usd": "Aave", "morpho_tvl_usd": "Morpho", "compound_tvl_usd": "Compound"})
    (trio/1e9).plot(ax=ax[0])
    ax[0].set(title="Сравнение родительских протоколов DeFiLlama", ylabel="TVL, млрд USD")
    (p.aave_share_three*100).plot(ax=ax[1], color="#4472c4")
    ax[1].set(ylabel="Доля Aave в TVL тройки, %", ylim=(0, 100))
    (trio/trio.iloc[0]*100).plot(ax=ax[2])
    ax[2].set(ylabel="TVL, база 100", title="Общая база: 01.11.2025 = 100")
    finish(fig, "02_competitors", ax)

    fig, ax = plt.subplots(figsize=(11, 6))
    shares = b["chain_latest"].share_parent_tvl.sort_values(ascending=False)
    top = shares.head(7).copy()
    if len(shares) > 7:
        top.loc["Остальные сети"] = shares.iloc[7:].sum()
    (top.sort_values()*100).plot.barh(ax=ax, color="#4472c4")
    for i, v in enumerate(top.sort_values()*100):
        ax.text(v+.6, i, f"{v:.2f}%", va="center", fontsize=9)
    ax.set(title="Распределение TVL Aave по сетям — 31.10.2026, 00:00 UTC", xlabel="Доля в TVL Aave, %", ylabel="Сеть", xlim=(0,100))
    finish(fig, "03_chains")

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    (e[["fees_usd_ma7", "revenue_usd_ma7"]]/1e6).rename(columns={"fees_usd_ma7": "Fees", "revenue_usd_ma7": "Protocol revenue"}).plot(ax=ax[0])
    ax[0].set(title="Экономическая активность Aave — скользящее среднее за 7 дней", ylabel="Средний дневной поток, млн USD")
    (e[["fees_tvl_daily", "revenue_tvl_daily"]].rolling(7, min_periods=7).mean()*1e4).rename(columns={"fees_tvl_daily": "Fees / TVL", "revenue_tvl_daily": "Revenue / TVL"}).plot(ax=ax[1])
    ax[1].set_ylabel("Средний дневной поток / TVL, б.п.")
    if not e.fees_usd.notna().any():
        ax[0].text(.5,.5,"Fees/revenue недоступны; пропуски не заменены нулями",ha="center",transform=ax[0].transAxes)
    finish(fig, "04_economics", ax)

    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for column, (symbol, row) in enumerate(b["lending"].iterrows()):
        scale = 1e6 if symbol != "WETH" else 1e3
        label = "млн "+symbol if symbol != "WETH" else "тыс. WETH"
        axes[0,column].bar(["Supply", "Borrow"], [row.supply_token/scale, row.borrow_token/scale], color=["#4472c4", "#d97732"])
        axes[0,column].set(title=symbol, ylabel=label)
        for i, v in enumerate([row.supply_token/scale, row.borrow_token/scale]):
            axes[0,column].text(i, v, f"{v:,.1f}", ha="center", va="bottom", fontsize=9)
        axes[0,column].margins(y=.2)
        axes[1,column].bar(["Subgraph", "Borrow/Supply"], [row.utilization_reported*100,row.borrow_supply_ratio*100], color=["#328770", "#7890a6"])
        axes[1,column].set(ylabel="Использование, %", ylim=(0,105))
        for i,v in enumerate([row.utilization_reported*100,row.borrow_supply_ratio*100]):
            axes[1,column].text(i,v,f"{v:.2f}%",ha="center",va="bottom",fontsize=9)
        axes[2,column].bar(["Supply", "Variable borrow"], [row.supply_apy*100,row.variable_borrow_apy*100], color=["#4472c4", "#d97732"])
        axes[2,column].set_ylabel("Расчётный APY, %")
    fig.suptitle("Lending snapshot Aave V3 Ethereum — блок 26316188, 31.10.2026\nБалансы в единицах каждого токена; APY при неизменном APR", fontsize=13)
    finish(fig, "05_lending_snapshot")

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    t.aave_price_usd.plot(ax=ax[0],color="#4472c4")
    ax[0].set(title="Рыночная динамика AAVE и ETH — доступная история", ylabel="AAVE, USD")
    prices=t[["aave_price_usd","eth_price_usd"]].dropna()
    (prices/prices.iloc[0]*100).rename(columns={"aave_price_usd":"AAVE","eth_price_usd":"ETH"}).plot(ax=ax[1])
    ax[1].set_ylabel("Цена: первая общая дата = 100")
    finish(fig,"06_token_prices",ax)

    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    (t.aave_log_return*100).plot(ax=ax[0],color="#4472c4",lw=.9)
    ax[0].set(title="AAVE: дневные логарифмические доходности",ylabel="log(Pt/Pt−1), %")
    ax[1].hist(t.aave_log_return.dropna()*100,bins=35,color="#4472c4",alpha=.85)
    ax[1].set(xlabel="Дневная логарифмическая доходность, %",ylabel="Число дней",title="Распределение фактических дневных доходностей")
    (t.aave_rolling_30d_vol_annual*100).plot(ax=ax[2],color="#7c4d9a")
    ax[2].set(title="30-дневная реализованная волатильность, годовое приведение",ylabel="std(log r) × √365, %")
    finish(fig,"07_returns_volatility",[ax[0],ax[2]])

    fig, ax = plt.subplots(2, 1, figsize=(11, 7),sharex=True)
    cap_tvl=pd.DataFrame({"Market Cap AAVE":t.aave_market_cap_usd,"TVL Aave":p.aave_tvl_usd}).loc[t.aave_market_cap_usd.notna()]/1e9
    cap_tvl.plot(ax=ax[0],logy=True)
    ax[0].set(title="Капитализация AAVE и TVL Aave — совпадающие даты",ylabel="млрд USD, логарифмическая шкала")
    t.market_cap_tvl.plot(ax=ax[1],color="#7c4d9a")
    ax[1].set_ylabel("Market Cap / TVL, доля")
    finish(fig,"08_valuation",ax)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    if len(b["joint"]):
        sns.heatmap(b["correlation_matrix"],vmin=-1,vmax=1,cmap="vlag",annot=True,fmt=".2f",ax=ax[0],square=True,cbar=False)
        ax[0].tick_params(axis="x",labelrotation=45,labelsize=8)
        ax[0].tick_params(axis="y",labelsize=8)
        ax[0].set_title(f"Pearson: дневные изменения, n={len(b['joint'])}")
        ax[1].scatter(b["joint"]["Δlog TVL"]*100,b["joint"]["AAVE r"]*100,s=14,alpha=.5,color="#4472c4")
    else:
        ax[0].text(.5,.5,"Недостаточно совместных наблюдений",ha="center",transform=ax[0].transAxes)
    ax[1].set(title="Изменение TVL и доходность AAVE",xlabel="Δlog TVL, %",ylabel="Логарифмическая доходность AAVE, %")
    finish(fig,"09_associations")

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    counts=b["events"].event_name.value_counts().reindex(["Supply","Borrow","Repay"])
    counts.plot.bar(ax=ax[0,0],rot=0,color=["#4472c4","#d97732","#7c4d9a"])
    ax[0,0].set(title="Число событий по типам",xlabel="Тип события",ylabel="События")
    b["hourly"].plot.bar(stacked=True,ax=ax[0,1],width=.85,color=["#4472c4","#d97732","#7c4d9a"])
    ax[0,1].set(title="Распределение активности по часам",xlabel="Час UTC",ylabel="События")
    ax[0,1].set_ylim(0,b["hourly"].sum(axis=1).max()*1.35)
    ax[0,1].legend(title="События",loc="upper right",fontsize=9)
    ax[0,1].tick_params(axis="x",rotation=0,labelsize=8)
    b["participants"].plot.bar(ax=ax[1,0],rot=0,color="#328770")
    ax[1,0].set(title="Уникальные участники по типам операций",xlabel="Тип события",ylabel="Адреса, объединение ролей")
    b["reserve_counts"].head(8).sort_values().plot.barh(ax=ax[1,1],color="#7890a6")
    ax[1,1].set(title="Восемь резервов с наибольшим числом событий",xlabel="События",ylabel="Резерв")
    fig.suptitle("Однодневный on-chain case study — 31.10.2026 UTC, Aave V3 Ethereum",fontsize=13)
    finish(fig,"10_onchain_case")
    return paths
