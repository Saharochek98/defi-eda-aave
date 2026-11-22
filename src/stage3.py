"""Run stage 3 only, reporting unmet data access requirements honestly."""

import argparse
import json
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient
from jupyter_client import KernelManager

from .config import ROOT, RAW_DIR, PROCESSED_DIR, Settings, REQUIRED_SOURCES, required_sources_verified
from .onchain import (RAW, OUT, POOL, SUBGRAPH_ID, EVENT_COLUMNS, OnchainError, collect_rpc,
                      verify_secondary_rpc, validate_events, compare_event_sets, daily_activity, load_snapshot)
from .onchain_sources import collect_dune, collect_graph, collect_etherscan


def load_events(snapshot, start, end):
    return validate_events(pd.DataFrame(snapshot["data"]["decoded"], columns=EVENT_COLUMNS), start, end)[0]


def collect_stage3(days=1, refresh=False, offline=False, dune_csv=None):
    if not 1 <= days <= 7:
        raise ValueError("Use 1 to 7 historical days only")
    _, end = Settings().period
    start = end-timedelta(days=days)
    report = {"start_utc": start.isoformat(), "end_exclusive_utc": end.isoformat(),
              "pool": POOL.lower(), "chain_id": 1, "required_sources": list(REQUIRED_SOURCES),
              "optional_sources": {"Dune": {"status": "optional_public_only", "required": False}},
              "sources": {}, "comparisons": {}}
    snapshots = {}

    def attempt(name, filename, action):
        try:
            snapshot = load_snapshot(filename) if offline else action()
            if snapshot is None:
                report["sources"][name] = {"status": "needs_configuration_or_real_export"}
                return None
            req = snapshot["request"]
            if name in {"web3.py", "Dune"} and (req.get("start_utc") != start.isoformat() or req.get("end_exclusive_utc") != end.isoformat()):
                raise OnchainError("cached_interval_mismatch")
            if name in {"Etherscan", "secondary_rpc"} and (req.get("first_block") != rpc["data"]["first_block"] or req.get("last_block") != rpc["data"]["last_block"]):
                raise OnchainError("cached_block_interval_mismatch")
            if name == "The Graph" and req.get("block") != rpc["data"]["last_block"]:
                raise OnchainError("cached_graph_block_mismatch")
            snapshots[name] = snapshot
            count = len(snapshot["data"].get("decoded", snapshot["data"].get("reserves", [])))
            report["sources"][name] = {"status": "data_collected", "records": count,
                                      "fetched_at_utc": snapshot["fetched_at_utc"], "cache_file": filename + ".json"}
            return snapshot
        except Exception as exc:
            report["sources"][name] = {"status": "failed", "reason": str(exc) if isinstance(exc, OnchainError) else type(exc).__name__}
            return None
        finally:
            print(name, report["sources"].get(name, {}).get("status"), flush=True)

    rpc = attempt("web3.py", "web3_events", lambda: collect_rpc(start, end, refresh))
    if dune_csv:
        attempt("Dune", "dune_events", lambda: collect_dune(start, end, dune_csv, refresh))
    if rpc:
        etherscan = attempt("Etherscan", "etherscan_events", lambda: collect_etherscan(rpc, start, end, refresh))
        attempt("The Graph", "graph_markets", lambda: collect_graph(rpc, start, end, refresh))
        if etherscan and compare_event_sets(load_events(rpc, start, end), load_events(etherscan, start, end))["exact_match"]:
            report["sources"]["secondary_rpc"] = {"status": "not_needed_etherscan_exact_match"}
        else:
            attempt("secondary_rpc", "secondary_rpc_events", lambda: verify_secondary_rpc(rpc, start, end, refresh))
    else:
        for source in ["Etherscan", "The Graph", "secondary_rpc"]:
            report["sources"][source] = {"status": "not_run_without_verified_rpc_boundaries"}
    OUT.mkdir(parents=True, exist_ok=True)
    if rpc:
        frame = load_events(rpc, start, end)
        frame.to_csv(OUT / "web3_events.csv", index=False)
        activity = daily_activity(frame, start, end)
        activity.to_csv(OUT / "daily_activity.csv", index_label="date")
        report["rpc"] = {"first_block": rpc["data"]["first_block"], "last_block": rpc["data"]["last_block"],
                         "log_count": len(frame), "days": len(activity), "unique_transactions": int(frame.tx_hash.nunique()),
                         "active_participant_addresses": int(pd.concat([frame.user, frame.on_behalf_of, frame.repayer]).loc[lambda s: s.ne("") & s.ne("0x"+"0"*40)].nunique()),
                         "counts": {k: int(v) for k, v in frame.event_name.value_counts().items()}}
        for name in ["Etherscan", "Dune", "secondary_rpc"]:
            if name in snapshots:
                other = load_events(snapshots[name], start, end)
                report["comparisons"][name] = compare_event_sets(frame, other)
                other.to_csv(OUT / f"{name.lower()}_events.csv", index=False)
        if "The Graph" in snapshots:
            data = snapshots["The Graph"]["data"]
            pd.json_normalize(data["reserves"]).to_csv(OUT / "graph_reserves_raw.csv", index=False)
            pd.json_normalize(data["reserveParamsHistoryItems"]).to_csv(OUT / "graph_reserve_history_raw.csv", index=False)
            history = data["reserveParamsHistoryItems"]
            timestamps = [int(row["timestamp"]) for row in history]
            if any(not start.timestamp() <= t < end.timestamp() for t in timestamps):
                raise OnchainError("graph_history_outside_requested_interval")
            report["graph"] = {"historical_block": rpc["data"]["last_block"], "reserves": len(data["reserves"]),
                               "parameter_observations": len(history),
                               "first_observation_utc": pd.Timestamp(min(timestamps), unit="s", tz="UTC").isoformat() if timestamps else None,
                               "last_observation_utc": pd.Timestamp(max(timestamps), unit="s", tz="UTC").isoformat() if timestamps else None,
                               "deployment": data["schema"]["_meta"]["deployment"],
                               "live_schema_checked": True}
        make_activity_plot(activity)
    instruments = []
    # Verify cached stage 2 evidence rather than counting mere mentions or probes.
    from .data_processing import prepare_data
    _, _, common, audit, _ = prepare_data()
    if not audit["missing_sources"] and len(common):
        instruments.extend(["DeFiLlama", "CoinGecko"])
    instruments.extend(name for name in ["web3.py", "Etherscan", "The Graph"]
                       if report["sources"].get(name, {}).get("status") == "data_collected" and report["sources"][name]["records"] > 0)
    report["instruments_with_real_data"] = instruments
    report["five_instruments_requirement_met"] = required_sources_verified(instruments)
    report["cross_source_mismatches"] = [name for name, result in report["comparisons"].items() if not result["exact_match"]]
    report["missing_required_sources"] = sorted(set(REQUIRED_SOURCES) - set(instruments))
    report["stage3_complete"] = bool(rpc) and report["five_instruments_requirement_met"] and not report["cross_source_mismatches"] and report["comparisons"].get("Etherscan", {}).get("exact_match", False)
    availability_path = ROOT / "source_availability.json"
    availability = json.loads(availability_path.read_text(encoding="utf-8")) if availability_path.exists() else {}
    report["connection_checks_passed"] = (
        availability.get("required_sources") == list(REQUIRED_SOURCES) and
        all(availability.get("sources", {}).get(name, {}).get("status") == "probe_ok" for name in REQUIRED_SOURCES))
    report["connection_checks_at_utc"] = availability.get("checked_at_utc")
    report["analysis_ready"] = report["stage3_complete"] and report["connection_checks_passed"]
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "stage3_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report)
    return report


def make_activity_plot(activity):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    activity[["Supply", "Borrow", "Repay"]].sum().plot.bar(ax=axes[0], color=["#4472c4", "#d97732", "#7c4d9a"], rot=0)
    axes[0].set(title="События Aave V3 Ethereum", xlabel="Тип события", ylabel="Количество событий")
    activity[["active_participant_addresses", "unique_transactions"]].sum().rename(index={"active_participant_addresses": "Адреса по дням", "unique_transactions": "Транзакции по дням"}).plot.bar(ax=axes[1], rot=0, color=["#328770", "#7890a6"])
    axes[1].set(title="Сумма дневных показателей", xlabel="Показатель", ylabel="Количество")
    first, last = activity.index[0].strftime("%d.%m.%Y"), activity.index[-1].strftime("%d.%m.%Y")
    fig.suptitle(f"Ограниченная выборка: {first}–{last}, UTC; один Pool")
    fig.tight_layout()
    fig.savefig(OUT / "activity.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_report(report):
    lines = ["# Этап 3 — on-chain аналитика", "",
             "**" + ("Критерий этапа выполнен." if report["stage3_complete"] else "Этап выполнен частично: критерий пяти инструментов или сверка ещё не подтверждены.") + "**", "",
             f"Выборка: {report['start_utc']} ≤ время < {report['end_exclusive_utc']}. "
             f"Ethereum mainnet, chain_id=1, Pool Aave V3 `{POOL}`. "
             "Один Ethereum Pool не представляет все сети и версии Aave.", "",
             "## Доступ и фактическое использование", "", "| Источник | Результат | Реальных записей |", "| --- | --- | ---: |"]
    labels = {"data_collected": "Данные получены", "needs_configuration_or_real_export": "Нужна настройка ключа или реальный CSV",
              "failed": "Ошибка", "not_run_without_verified_rpc_boundaries": "Не запускался без проверенных границ RPC",
              "not_needed_etherscan_exact_match": "Не требуется: события полностью сверены с Etherscan"}
    for name, meta in report["sources"].items():
        lines.append(f"| {name} | {labels[meta['status']]}" + (f" ({meta['reason']})" if meta.get("reason") else "") + f" | {meta.get('records', '—')} |")
    lines += ["", f"Инструменты с реально собранными данными: **{', '.join(report['instruments_with_real_data'])}** "
              f"({len(report['instruments_with_real_data'])} из 5 обязательных). "
              "Второй RPC-провайдер является проверкой web3.py, а не отдельным инструментом из задания.", ""]
    lines += [f"Подключения ко всем пяти проверены: **{'да' if report.get('connection_checks_passed') else 'нет'}**. "
              f"Время проверки UTC: {report.get('connection_checks_at_utc')}. "
              f"Готовность к полноценному анализу (`analysis_ready`): **{'да' if report.get('analysis_ready') else 'нет'}**. "
              "Этап 4 этим запуском не выполняется.", "",
              "Dune — только дополнительный публичный источник без API, оплаты и обязательного запуска SQL; "
              "его отсутствие не влияет на готовность проекта.", ""]
    if "rpc" in report:
        rpc = report["rpc"]
        lines += ["## Реальные результаты выборки", "",
                  f"Блоки {rpc['first_block']}–{rpc['last_block']} включительно. "
                  f"Получено {rpc['log_count']} событий: " + ", ".join(f"{k}: {v}" for k, v in rpc["counts"].items()) + ".", "",
                  f"Уникальных транзакций: {rpc['unique_transactions']}; уникальных адресов участников: {rpc['active_participant_addresses']}.", "",
                  "Адреса участников — объединение `user` и `onBehalfOf` в Supply/Borrow, "
                  "`user` и `repayer` в Repay, без нулевого адреса. Это адреса, а не люди и не все пользователи Aave. "
                  "Событий может быть несколько в одной транзакции. Дневные показатели считаются внутри каждого дня.", ""]
    lines += ["## Сопоставление источников", ""]
    if report["comparisons"]:
        for name, result in report["comparisons"].items():
            lines.append(f"- web3.py ↔ {name}: общих событий {result['common_events']}, "
                         f"только в RPC {result['only_left']}, только во втором источнике {result['only_right']}, "
                         f"расхождений полей {result['value_mismatches']}; полное совпадение: {'да' if result['exact_match'] else 'нет'}.")
    else:
        lines.append("Независимая сверка не выполнена: нет второго успешно полученного набора событий. Совпадение результатов не заявляется.")
    lines += ["", "Проверяется не только количество, но идентичности `(tx_hash, log_index)`, блок, время UTC, "
              "тип события, резерв, адреса участников и исходное количество актива. "
              "Для второго RPC дополнительно проверяются хеши блоков.", "",
              "## Подготовленные подключения и воспроизведение", "",
              "- `src/onchain.py`: web3.py, реальные ABI Supply/Borrow/Repay, двоичный поиск блоков по UTC, порции до 100 блоков, кэш и проверки.",
              "- `src/onchain_sources.py`: Etherscan V2 с пагинацией, The Graph с проверкой живой схемы; необязательный публичный CSV Dune без API.",
              "- `sql/`: сохранённые справочные запросы; их запуск не требуется и не входит в обязательный план.",
              "- `src/graphql/`: запросы резервов на фиксированном историческом блоке и параметров внутри интервала. "
              "Наличие схемы из репозитория не означает успешный запрос к gateway.",
              "- `data/raw/onchain/`: исходные события, блоки, границы, официальные справочные исходники, журнал статусов.",
              "- `data/processed/onchain/`: подтверждённые события CSV, дневная активность, таблицы сверки и PNG при наличии данных.", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe -m src.stage3 --offline",
              ".\\.venv\\Scripts\\python.exe -m src.stage3",
              ".\\.venv\\Scripts\\python.exe -m src.stage3 --dune-csv data/raw/onchain/dune_export.csv", "```", "",
              "После настройки ключей в `.env` повторите `python -m src.stage3`. "
              "Dune не входит в обязательные интеграции. Ключи, API, подписка и запуск собственного SQL не требуются. "
              "Допускается дополнительная сверка с доступным публичным результатом или CSV после проверки его источника и методики.", "",
              "Для The Graph используйте `THEGRAPH_API_KEY`; проверенный по официальному списку ID Ethereum V3 — "
              f"`{SUBGRAPH_ID}` (используется по умолчанию, если переменная пуста). "
              "Для Etherscan задайте `ETHERSCAN_API_KEY`. Секреты храните локально, не в notebook или отчёте.", "",
              "## Ограничения", "",
              "`amount_raw` хранится точной десятичной строкой uint256 в минимальных единицах конкретного токена. "
              "Суммы разных резервов не складываются и не выдаются за USD. "
              "Supply/Borrow/Repay — потоки операций, а не остатки задолженности или депозитов. "
              "Только завершённая выборка логов допускает нулевые дневные счётчики; ошибки не превращаются в нули.", "",
              "Без успешно полученных данных The Graph нельзя заявлять исторические supply/borrow/utilization или APY. "
              "Показатели параметров резервов в subgraph обновляются событиями, а не гарантированно раз в сутки. "
              "Даже успешная выборка одного дня не позволяет проверить годовую исследовательскую гипотезу.", "",
              "Этап 4 и подготовка итогового PDF не выполнялись.", "",
              "Источники: [Aave addresses](https://aave.com/docs/reference/addresses), "
              "[address book](https://github.com/bgd-labs/aave-address-book/blob/main/src/AaveV3Ethereum.sol), "
              "[IPool ABI source](https://github.com/aave/aave-v3-core/blob/master/contracts/interfaces/IPool.sol), "
              "[Aave subgraphs](https://github.com/aave/protocol-subgraphs), "
              "[Dune raw event logs](https://docs.dune.com/data-catalog/evm/ethereum/decoded/event-logs), "
              "[Etherscan API](https://docs.etherscan.io/api-reference/endpoint/getlogs-address-topics)."]
    if "graph" in report:
        graph = report["graph"]
        index = lines.index("## Подготовленные подключения и воспроизведение")
        lines[index:index] = ["## Полученные данные The Graph", "",
            f"Живая GraphQL-схема проверена. На историческом блоке {graph['historical_block']} получено "
            f"{graph['reserves']} резервов выбранного Pool. Внутри суток — {graph['parameter_observations']} "
            f"записей параметров резервов, от {graph['first_observation_utc']} до {graph['last_observation_utc']}.", "",
            "Доступны исходные totalATokenSupply, totalCurrentVariableDebt, totalPrincipalStableDebt, "
            "availableLiquidity, utilizationRate, liquidityRate и variableBorrowRate. Балансы представлены "
            "в минимальных единицах каждого актива (decimals сохранены), ставки — в ray (10^27), "
            "utilizationRate — безразмерная величина из subgraph. Stable principal не равен текущему "
            "stable debt с начисленными процентами. Общий Borrow или APY на этом этапе не подменяются этими полями.", "",
            "Остатки на последнем блоке суток и внутрисуточные записи параметров различаются по времени "
            "и смыслу. События из Etherscan сверены с RPC; полная независимая сверка балансов subgraph "
            "на этом этапе не выполнялась.", ""]
    index = lines.index("## Ограничения")
    lines[index:index] = ["## Проверки реализации и обнаруженные особенности API", "",
        "Тесты обработки включают декодирование трёх ABI событий, сохранение точного uint256, "
        "дубликаты и конфликты событий, исключающую верхнюю границу UTC, неверную сеть/контракт, "
        "объединение адресов участников, сверку содержимого и целостность кэша. Запуск: "
        "`python -m unittest discover -s tests -v`.", "",
        "Публичные RPC имеют разные ограничения методов и размера диапазона. Для dRPC используется "
        "диапазон до 100 блоков; HTTP 408 обрабатывается ограниченным повтором. Промежуточный "
        "кэш позволяет продолжить сбор. Расширение `blockTimestamp`, фактически возвращённое "
        "провайдером в логах, проверяется на согласованность внутри блока и по трём заголовкам "
        "в начале, середине и конце выборки; при отсутствии поля заголовки запрашиваются явно. "
        "Границы суток проверены отдельными соседними блоками.", "",
        "Etherscan может возвращать нулевой индекс как `0x`; это обрабатывается как числовой ноль. "
        "После декодирования все события сверены с RPC, включая хеши блоков. "
        "Ни ошибки, ни неполная загрузка не превращаются в успешные наборы данных.", ""]
    (ROOT / "STAGE3_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_notebook():
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md("# Этап 3. On-chain анализ Aave V3 Ethereum\n\n"
                "Ограниченная историческая выборка, реальные события. Статус выполнения требований определяется "
                "по собранным данным, а не по наличию кода подключения. Повторный запуск этих ячеек работает из кэша."),
             code("from pathlib import Path\nimport sys, json\nimport pandas as pd\nfrom IPython.display import display, Markdown, Image\n"
                  "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src' / 'config.py').exists())\n"
                  "sys.path.insert(0, str(ROOT))\nfrom src.onchain import RAW, OUT\n"
                  "report3 = json.loads((RAW / 'stage3_manifest.json').read_text(encoding='utf-8'))\n"
                  "display(pd.DataFrame(report3['sources']).T)\n"
                  "display(Markdown('Инструменты с реальными данными: **' + ', '.join(report3['instruments_with_real_data']) + '**'))\n"
                  "display(Markdown('Критерий этапа выполнен: **' + ('да' if report3['stage3_complete'] else 'нет; см. STAGE3_REPORT.md') + '**'))"),
             code("if 'rpc' in report3:\n"
                  "    display(pd.read_csv(OUT / 'daily_activity.csv'))\n"
                  "    display(pd.read_csv(OUT / 'web3_events.csv', dtype={'amount_raw': str}).head())\n"
                  "    display(Image(filename=str(OUT / 'activity.png')))\n"
                  "else:\n    display(Markdown('Полная выборка RPC недоступна; нулевые показатели не подставлены.'))\n"
                  "display(pd.DataFrame(report3['comparisons']).T)"),
             code("if 'graph' in report3:\n"
                  "    display(pd.DataFrame([report3['graph']]))\n"
                  "    reserves3 = pd.read_csv(OUT / 'graph_reserves_raw.csv', dtype=str)\n"
                  "    display(reserves3[['symbol', 'decimals', 'totalATokenSupply', 'totalCurrentVariableDebt', 'utilizationRate']].head(10))\n"
                  "    display(Markdown('Балансы — минимальные единицы актива, ставки liquidityRate/variableBorrowRate — ray (10^27). Параметры обновляются событиями.'))\n"
                  "else:\n    display(Markdown('Данные резервов The Graph не получены.'))"),
             md("Счётчики Supply/Borrow/Repay считают события; адреса участников включают инициатора и бенефициара "
                "или погашающего адреса. Это один Pool, а не все рынки и сети Aave. `amount_raw` — точные "
                "минимальные единицы конкретного резерва; USD-оценка не выполнялась. Второй RPC-провайдер "
                "не считается отдельным инструментом из задания. Обязательны DeFiLlama, CoinGecko, The Graph, Etherscan и web3.py; "
                "Dune допускается только как дополнительный публичный источник без API и оплаты. "
                "Подробные ограничения и шаги настройки — в `STAGE3_REPORT.md`. "
                "К этапу 4 перехода нет.")]
    for cell in cells:
        cell.metadata["tags"] = ["stage3"]
    nb = nbformat.v4.new_notebook(cells=cells)
    with tempfile.TemporaryDirectory(dir=ROOT / ".venv") as folder:
        km = KernelManager(kernel_name="python3", connection_file=str(Path(folder) / "kernel.json"))
        km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        try:
            NotebookClient(nb, km=km, timeout=180, resources={"metadata": {"path": str(ROOT)}}).execute(
                env=dict(os.environ, IPYTHONDIR=str(ROOT / ".venv" / "ipython")))
        finally:
            if km.has_kernel:
                km.shutdown_kernel(now=True)
            km.cleanup_resources()
    path = ROOT / "notebooks" / "exploration.ipynb"
    existing = nbformat.read(path, as_version=4)
    existing.cells = [c for c in existing.cells if "stage3" not in c.metadata.get("tags", [])] + nb.cells
    nbformat.validate(existing)
    nbformat.write(existing, path)


def main():
    parser = argparse.ArgumentParser(description="Stage 3 only: historical on-chain collection and reconciliation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--offline", action="store_true")
    group.add_argument("--refresh", action="store_true")
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--dune-csv", type=Path)
    args = parser.parse_args()
    report = collect_stage3(args.days, args.refresh, args.offline, args.dune_csv)
    append_notebook()
    print("Stage 3 complete:", report["stage3_complete"])
    print("Instruments with real data:", ", ".join(report["instruments_with_real_data"]))


if __name__ == "__main__":
    main()
