"""Rebuild stage 2 outputs and an executed research notebook."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import nbformat
import pandas as pd
from nbclient import NotebookClient
from jupyter_client import KernelManager

from .config import ROOT, RAW_DIR, PROCESSED_DIR
from .data_collection import collect_history
from .data_processing import prepare_data


def date_label(value):
    return pd.Timestamp(value).strftime("%d.%m.%Y") if value else "нет данных"


def coverage_table(audit):
    return pd.DataFrame([
        {"Ряд": name, "Дней": s["observed_days"], "Пропуски": s["missing_days"],
         "Начало UTC": date_label(s["first_date"]), "Конец UTC": date_label(s["last_date"]),
         "Дубли времени": s["duplicate_timestamps"], "Внутридневные": s["extra_observations_same_day"]}
        for name, s in audit["series"].items()])


def interpretations(protocol, tokens, common):
    notes = {}
    if "aave_tvl_usd" in protocol:
        series = protocol["aave_tvl_usd"].dropna()
        if len(series):
            notes["aave_tvl"] = (
                f"На доступном интервале {date_label(series.index[0])}–{date_label(series.index[-1])} "
                f"TVL Aave изменился с {series.iloc[0]/1e9:.2f} до {series.iloc[-1]/1e9:.2f} млрд USD. "
                "TVL в USD зависит и от количества активов, и от их цен; этот график не измеряет приток средств или объём кредитов.")
    comparable = protocol.dropna()
    if len(comparable) and len(protocol.columns) == 3:
        end = comparable.iloc[-1]
        notes["protocol_comparison"] = (
            f"На {date_label(comparable.index[-1])} TVL: Aave — {end['aave_tvl_usd']/1e9:.2f}, "
            f"Morpho — {end['morpho_tvl_usd']/1e9:.2f}, Compound — {end['compound_tvl_usd']/1e9:.2f} млрд USD. "
            "Использованы готовые родительские агрегаты DeFiLlama без повторного суммирования версий. "
            "Объёмы TVL не означают одинаковый риск и не учитывают различия моделей протоколов.")
    if len(common):
        first, last = common.iloc[0], common.iloc[-1]
        notes["token_prices_normalized"] = (
            f"На общем интервале {date_label(common.index[0])}–{date_label(common.index[-1])} "
            f"индекс AAVE изменился со 100 до {last['aave_price_usd']/first['aave_price_usd']*100:.1f}, "
            f"ETH — со 100 до {last['eth_price_usd']/first['eth_price_usd']*100:.1f}. "
            "Нормирование показывает относительное изменение цен; корреляция и причинные выводы здесь не рассчитываются.")
        notes["aave_market_data"] = (
            f"На {date_label(common.index[-1])} капитализация AAVE составляет "
            f"{last['aave_market_cap_usd']/1e9:.2f} млрд USD, объём торгов за предыдущие 24 часа — "
            f"{last['aave_volume_24h_usd']/1e9:.2f} млрд USD. "
            "Капитализация относится к токену, объём — к рыночным торгам; это не TVL, fees или revenue Aave. "
            "Ряд total_volumes является скользящим 24-часовым объёмом, а не суммой операций за календарные UTC-сутки.")
    return notes


def write_summary(protocol, tokens, common, audit):
    lines = ["# Этап 2 — первичный сбор исторических данных", "",
             "Получены реальные данные DeFiLlama и CoinGecko. Этапы 3–5 не выполнялись.", "",
             "## Периоды и качество", "",
             f"Запрошено: {date_label(audit['period_utc_half_open'][0])}–"
             f"{date_label(datetime.fromisoformat(audit['period_utc_half_open'][1])-timedelta(days=1))}, "
             f"{audit['expected_days']} дней, UTC.", "",
             f"Общий интервал TVL и рыночных данных: {date_label(audit['common']['first_date'])}–"
             f"{date_label(audit['common']['last_date'])}, {audit['common']['days']} наблюдений.", "",
             "| Ряд | Дней | Пропуски | Начало UTC | Конец UTC |", "| --- | ---: | ---: | --- | --- |"]
    for name, s in audit["series"].items():
        lines.append(f"| {name} | {s['observed_days']} | {s['missing_days']} | {date_label(s['first_date'])} | {date_label(s['last_date'])} |")
    lines += ["", "Пропуски не заполнялись. Одинаковые временные метки с разными значениями вызывают ошибку. "
              "При нескольких наблюдениях за UTC-сутки выбирается последнее фактическое значение; "
              "число таких случаев фиксируется отдельно. Непустые значения проверены на отрицательные и бесконечные числа.", "",
              f"Дубли временных меток: {sum(s['duplicate_timestamps'] for s in audit['series'].values())}; "
              f"внутридневные дополнительные наблюдения: {sum(s['extra_observations_same_day'] for s in audit['series'].values())}; "
              f"наблюдения вне 00:00 UTC в выбранном периоде: {sum(s['non_midnight_observations_in_period'] for s in audit['series'].values())}; "
              f"нулевые значения: {sum(s['zero_values'] for s in audit['series'].values())}.", "",
              "Ограничения запросов CoinGecko и причины сокращения периода сохранены ниже. "
              "Это фактические ответы API, а не восстановленные или синтетические наблюдения.", "",
              "## Первые наблюдения", ""]
    source_notes = []
    for name, meta in audit["sources"].items():
        details = ("полный период отклонён с кодом CoinGecko 10012; успешно получено пересечение с бесплатным 365-дневным окном"
                   if meta.get("limitation") else meta.get("reason") or "покрытие не сокращалось")
        source_notes.append(f"- `{name}`: {'получен' if meta['status'] == 'ok' else 'ошибка'}; {details}.")
    lines[-2:-2] = source_notes + [""]
    lines.extend(f"- {note}" for note in interpretations(protocol, tokens, common).values())
    lines += ["", "## Воспроизведение и файлы", "",
              "- `data/raw/defillama_*.json`, `coingecko_*.json` — ответы с URL, параметрами, временем UTC и SHA-256; ключи не сохраняются.",
              "- `data/raw/history_manifest.json` — результаты запросов, ошибки полного периода и доступные источники.",
              "- `data/raw/daily_observations.csv` — исходные наблюдения выбранного периода с оригинальными временными метками.",
              "- `data/processed/protocol_daily.csv`, `market_daily.csv` — ежедневные ряды на целевом календаре.",
              "- `data/processed/combined_daily.csv` — объединение с явными пропусками; `common_daily.csv` — общие наблюдения.",
              "- `data/processed/data_quality.json` — пропуски по датам, дубликаты, нули и фактическое покрытие.",
              "- `data/processed/figures/` — четыре первых графика в PNG.",
              "- `notebooks/exploration.ipynb` — выполненный notebook с таблицами, графиками и интерпретациями.", "",
              "Повторный запуск из локального кэша:", "", "```powershell",
              ".\\.venv\\Scripts\\python.exe -m src.stage2 --offline", "```", "",
              "Без `--offline` проверяются существующие кэши; неизменившиеся запросы не повторяются. "
              "`--refresh` принудительно загружает историю заново. При более позднем обновлении "
              "бесплатное окно CoinGecko сдвинется и может сократить покрытие.", "",
              "## Ограничения и следующий этап", "",
              "TVL — готовый default TVL родительского протокола; дополнительные ряды borrowed, staking "
              "и pool2 не прибавляются. Данные отражают методологию и возможные ретроспективные "
              "пересчёты провайдеров. Объединение дат не означает одинаковый момент оценки всеми источниками.", "",
              "Основная гипотеза о кредитной активности и доходах пока не проверена: supply, borrow, "
              "utilization, fees и revenue на этапе 2 не собирались. Далее, после проверки пользователем, "
              "этап 3: реальные on-chain источники и согласование результатов. Итоговые notebook и PDF "
              "в submission относятся к этапу 5.", "",
              "Источники: [DeFiLlama API](https://api-docs.defillama.com/), "
              "[CoinGecko](https://www.coingecko.com/learn/how-to-fetch-historical-crypto-data-with-python)."]
    (ROOT / "STAGE2_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_notebook():
    md = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    cells = [md("# DeFi EDA — Aave: этап 2\n\n"
                "Первичный сбор исторических данных. Вопрос исследования: сопровождается ли рост TVL "
                "увеличением кредитной активности и доходов и как это связано с AAVE? "
                "На этом этапе проверяются данные и строятся первые графики; on-chain гипотеза ещё не проверена."),
             md("## 1. Источники и сбор\n\n"
                "Используются родительские агрегаты Aave, Morpho и Compound из DeFiLlama и USD-ряды AAVE/ETH "
                "CoinGecko. Все даты — UTC. Выполнение notebook использует сохранённые ответы, без сетевых запросов. "
                "Загрузка/обновление из корня проекта: `python -m src.data_collection --history` (добавьте `--refresh` для обновления). "
                "API-ключи читаются из `.env` и не выводятся."),
             code("from pathlib import Path\nimport sys, json\nimport pandas as pd\nfrom IPython.display import display, Markdown, Image\n"
                  "ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src' / 'config.py').exists()), None)\n"
                  "if ROOT is None:\n    raise RuntimeError('Откройте notebook из папки проекта')\n"
                  "sys.path.insert(0, str(ROOT))\n"
                  "from src.config import RAW_DIR, PROCESSED_DIR\nfrom src.data_processing import prepare_data\n"
                  "from src.plotting import make_initial_plots\nfrom src.stage2 import coverage_table, interpretations\n"
                  "manifest = json.loads((RAW_DIR / 'history_manifest.json').read_text(encoding='utf-8'))\n"
                  "display(pd.DataFrame(manifest['sources']).T[['status', 'file', 'fetched_at_utc']])"),
             md("## 2. Обработка и качество\n\n"
                "Целевой период: 01.11.2025 ≤ дата UTC < 01.11.2026. Пропуски сохраняются как NaN, "
                "интерполяция и forward fill не применяются. Конфликты значений при одной временной метке вызывают ошибку. "
                "Если за сутки есть несколько реальных наблюдений, используется последнее, число случаев учитывается в аудите. "
                "Ряды с нулём сохраняются и помечаются в аудите. Отрицательные и бесконечные значения вызывают ошибку."),
             code("protocol, tokens, common, audit, payloads = prepare_data()\n"
                  "display(coverage_table(audit))\n"
                  "display(Markdown(f\"Общий интервал: **{audit['common']['first_date']} — {audit['common']['last_date']}**, "
                  "наблюдений: **{audit['common']['days']}**.\"))\n"
                  "display(protocol.head(3))\ndisplay(tokens.loc[tokens.notna().any(axis=1)].head(3))"),
             md("**Ограничение CoinGecko.** Бесплатная история ограничена последними 365 днями; "
                "ошибка 10012 приводит к сокращению запрашиваемого интервала. "
                "Фактические даты и число пропусков приведены в таблице выше. "
                "На совместных графиках используется пересечение дат. Это снимок текущего сбора; "
                "после обновления бесплатное окно сдвинется."),
             code("display(pd.DataFrame(manifest['sources']).T.reindex(columns=['status', 'limitation', 'attempts']))"),
             code("figures = dict(make_initial_plots(protocol, tokens, common))\n"
                  "notes = interpretations(protocol, tokens, common)\n"
                  "print('Созданы графики:', ', '.join(figures))")]
    for name, title in [("aave_tvl", "3. TVL Aave"), ("protocol_comparison", "4. Сопоставление TVL"),
                        ("token_prices_normalized", "5. Динамика цен AAVE и ETH"), ("aave_market_data", "6. Капитализация и торговые объёмы AAVE")]:
        cells.extend([md("## " + title), code(
            f"if '{name}' in figures:\n    display(Image(filename=str(PROCESSED_DIR / 'figures' / '{name}.png')))\n    display(Markdown(notes['{name}']))\n"
            "else:\n    display(Markdown('Недостаточно реальных данных для графика. См. history_manifest.json.'))")])
    cells.extend([md("## 7. Проверки и ограничения\n\n"
                    "TVL, капитализация токена и объём торгов имеют разные определения. "
                    "Объём CoinGecko — скользящий показатель за 24 ч на момент наблюдения. "
                    "Нормированные цены — изменение относительно первой общей даты, а не индекс полной доходности. "
                    "TVL не равен сумме депозитов, USD-динамика включает переоценку активов. "
                    "Выводы о borrow, utilization, доходах и причинности требуют следующих этапов. "
                    "Финансовые метрики, on-chain сбор и итоговый PDF здесь не выполняются. "
                    "На этапе 5 будет создан самостоятельный notebook; исследовательский notebook этапа 2 использует `src/`."),
                  code("assert protocol.index.is_unique and tokens.index.is_unique\n"
                       "assert str(protocol.index.tz) == 'UTC' and str(tokens.index.tz) == 'UTC'\n"
                       "assert common.notna().all().all()\n"
                       "assert not audit['missing_sources'], audit['missing_sources']\n"
                       "print('Проверки уникальности, UTC, общих наблюдений и наличия источников пройдены.')\n"
                       "import matplotlib.pyplot as plt\nplt.close('all')")])
    nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3 (.venv)", "language": "python"}})
    path = ROOT / "notebooks" / "exploration.ipynb"
    return execute_notebook(nb, path)


def execute_notebook(nb, path):
    """Execute with the project interpreter and persist embedded outputs."""
    with tempfile.TemporaryDirectory(dir=ROOT / ".venv") as folder:
        km = KernelManager(kernel_name="python3", connection_file=str(Path(folder) / "kernel.json"))
        km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        env = dict(os.environ, IPYTHONDIR=str(ROOT / ".venv" / "ipython"), MPLCONFIGDIR=str(ROOT / ".venv" / "matplotlib"))
        try:
            NotebookClient(nb, km=km, timeout=180, resources={"metadata": {"path": str(ROOT)}}).execute(env=env)
        finally:
            if km.has_kernel:
                km.shutdown_kernel(now=True)
            km.cleanup_resources()
    nbformat.validate(nb)
    nbformat.write(nb, path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Collect, process, and execute the stage 2 notebook")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--offline", action="store_true", help="Use only existing raw files")
    group.add_argument("--refresh", action="store_true", help="Force new historical API requests")
    args = parser.parse_args()
    if not args.offline:
        collect_history(refresh=args.refresh)
    protocol, tokens, common, audit, _ = prepare_data()
    write_summary(protocol, tokens, common, audit)
    path = build_notebook()
    print(f"Executed notebook: {path}")
    print(f"Common daily observations: {len(common)}; missing sources: {audit['missing_sources']}")


if __name__ == "__main__":
    main()
