"""Stage 4 only: reproducible EDA from verified caches, without a PDF."""

import argparse
import json

import nbformat
import pandas as pd

from .config import ROOT
from .data_collection import collect_economics
from .metrics import EDA_DIR, prepare_eda
from .stage2 import execute_notebook


def percent(value):
    return "недоступно" if value is None else f"{value*100:.2f}%"


def million(value):
    return "недоступно" if value is None else f"{value/1e6:.2f} млн USD"


def describe_figures(b):
    s, l = b["summary"], b["lending"]
    fees_change = (s["fees_last30_mean"]/s["fees_first30_mean"]-1) if s["fees_first30_mean"] else None
    rev_change = (s["revenue_last30_mean"]/s["revenue_first30_mean"]-1) if s["revenue_first30_mean"] else None
    rows = [
        ("01_tvl", "TVL и 30/90-дневные изменения", "Как менялась USD-стоимость заблокированных активов Aave?",
         "TVL — готовый родительский агрегат DeFiLlama. Изменения вычисляются по календарным лагам 30 и 90 суток без заполнения пропусков.",
         f"TVL: {s['tvl_start_usd']/1e9:.2f} → {s['tvl_end_usd']/1e9:.2f} млрд USD ({percent(s['tvl_period_change'])}). "
         f"На 31.10.2026 изменение за 30 дней — {percent(s['tvl_change_30d'])}, за 90 — {percent(s['tvl_change_90d'])}. "
         "Это переоценка в USD вместе с изменением количества активов, а не измерение чистого притока."),
        ("02_competitors", "Aave, Morpho и Compound", "Как изменилось положение Aave внутри выбранной группы?",
         "Сравниваются родительские агрегаты за общие 365 дат. Доля = TVL Aave / (TVL Aave + TVL Morpho + TVL Compound).",
         f"Доля Aave в TVL тройки снизилась с {percent(s['share_three_start'])} до {percent(s['share_three_end'])}. "
         f"TVL Morpho изменился на {percent(s['competitor_changes']['morpho_tvl_usd'])}, Compound — на {percent(s['competitor_changes']['compound_tvl_usd'])}. "
         "Это не доля всего lending-рынка и не сумма уникального капитала: между протоколами возможны экономические пересечения."),
        ("03_chains", "Распределение TVL по блокчейнам", "Насколько TVL Aave сосредоточен на Ethereum?",
         "Исторические базовые chainTvls на 31.10.2026; borrowed, staking и pool2 исключены. Не используется текущий currentChainTvls.",
         f"На {s['chain_largest']} приходится {percent(s['chain_largest_share'])} TVL. "
         f"Сумма доступных сетевых значений отличается от родительского TVL на {s['chain_reconciliation_usd']:.0f} USD, что пренебрежимо относительно масштаба. "
         "Доля показывает концентрацию стоимости, а не кредитный риск отдельного резерва."),
        ("04_economics", "Fees и protocol revenue", "Совпадает ли изменение TVL с динамикой экономической активности?",
         "Fees — дневной поток комиссий и процентов по определениям версий DeFiLlama; revenue — часть, поступающая протоколу/казначейству. "
         "На графике 7-дневное среднее; отношение дневного потока к TVL того же календарного ярлыка выражено в базисных пунктах (1 б.п. = 0.0001). Это не APY.",
         f"За покрытый период fees — {million(s['fees_total_usd'])}, revenue — {million(s['revenue_total_usd'])}. "
         f"Средний дневной поток в последние 30 дней относительно первых 30 изменился: fees {percent(fees_change)}, revenue {percent(rev_change)}. "
         "Снижение уровней согласуется со снижением TVL, но не устанавливает механизм: fees включают также ликвидации, flash loans и иные поступления, а не только новые займы."),
        ("05_lending_snapshot", "Lending snapshot трёх резервов", "Как различаются остатки, использование и ставки USDC, WETH и USDT на конец выбранных суток?",
         "Баланс totalATokenSupply и текущий variable debt делятся на 10^decimals. Stable principal у выбранных резервов равен нулю. "
         "Отображаются utilizationRate subgraph и отдельное Borrow/Supply. APY = exp(31536000 × log1p(APR/31536000)) − 1; APR = ray/10^27, как в документации Aave.",
         "; ".join(f"{symbol}: utilization {percent(r.utilization_reported)}, supply APY {percent(r.supply_apy)}, variable borrow APY {percent(r.variable_borrow_apy)}" for symbol,r in l.iterrows()) + ". "
         f"Максимальное отличие utilizationRate от Borrow/Supply — {s['lending_utilization_gap_pp_max']:.3f} п.п. "
         "Поля сохраняют разные lastUpdate и методику subgraph; тождество не навязывается. Это один исторический snapshot, не годовая история. APY предполагает неизменный APR и не является реализованной доходностью."),
        ("06_token_prices", "Цена AAVE и нормированная динамика", "Как AAVE двигался относительно ETH на фактически доступном интервале?",
         "Daily 00:00 UTC, 21.11.2025–31.10.2026. Нормирование цен обоих токенов начинается со 100 в первую общую дату; это не индекс полной доходности.",
         f"AAVE: {s['aave_price_start']:.2f} → {s['aave_price_end']:.2f} USD, изменение {percent(s['aave_cumulative_return'])}; ETH — {percent(s['eth_cumulative_return'])}. "
         "Доходность за эти 344 суточных интервала не называется доходностью за полный год."),
        ("07_returns_volatility", "Доходности и реализованная волатильность", "Каков масштаб дневных колебаний AAVE?",
         "r_t = ln(P_t/P_{t−1}); std с ddof=1 × √365, поскольку крипторынок торгуется ежедневно. Скользящая оценка требует 30 фактических дневных доходностей.",
         f"Получено {s['aave_daily_returns_n']} дневных логарифмических доходности; волатильность с годовым приведением — {percent(s['aave_vol_annual'])}. "
         "Пропуски не соединяются доходностью за несколько дней. Годовое приведение не означает наличие полного года котировок и не является прогнозом риска."),
        ("08_valuation", "Капитализация и Market Cap / TVL", "Как изменилась капитализация токена относительно TVL протокола?",
         "Капитализация и TVL объединяются только на совпадающих UTC-датах. Отношение безразмерное; верхняя панель использует логарифмическую шкалу USD.",
         f"На 31.10.2026 капитализация AAVE — {s['market_cap_end_usd']/1e9:.2f} млрд USD; Market Cap / TVL вырос с {s['market_cap_tvl_start']:.4f} до {s['market_cap_tvl_end']:.4f}. "
         "Рост отношения может происходить при падении обоих показателей. Это не P/E, доходность или доказательство недооценки."),
        ("09_associations", "Наблюдаемые связи дневных изменений", "Связаны ли изменения TVL, экономической активности и цены токена?",
         "Pearson по логарифмическим изменениям, а не только трендовым уровням. Изменение fees/revenue с дневной меткой t−1 сопоставляется с доходностями и изменением TVL, заканчивающимися в 00:00 UTC t. "
         "Это явная договорённость о выравнивании дневных потоков и снимков; минутная синхронность источников не предполагается.",
         f"Корреляция доходностей AAVE и ETH — {s['token_corr']:.3f} (n={s['aave_daily_returns_n']}). " +
         " ".join(f"Δlog TVL ↔ {row.right}: r={row.pearson:.3f}, n={row.n}." for row in b['correlations'].iloc[1:].itertuples()) +
         " Корреляции описательны, чувствительны к выбросам и не доказывают причинность. USD-TVL сам зависит от цен криптоактивов; общий рынок может влиять на обе величины."),
        ("10_onchain_case", "Однодневный on-chain case study", "Как распределилась проверенная активность Pool 31.10.2026 по операциям, часам и резервам?",
         "Полный набор Supply/Borrow/Repay за UTC-сутки, сверенный с Etherscan. Участники — объединение user/onBehalfOf для Supply/Borrow и user/repayer для Repay, без нулевого адреса. amount_raw разных активов не суммируется.",
         f"{s['event_count']} событий в {s['unique_transactions']} транзакциях; {s['unique_participants']} уникальных адресов. "
         f"По типам: Supply {s['participants_by_type']['Supply']}, Borrow {s['participants_by_type']['Borrow']}, Repay {s['participants_by_type']['Repay']} адресов. "
         f"Самый активный час — {s['events_peak_hour']:02d}:00–{s['events_peak_hour']+1:02d}:00 UTC ({s['events_peak_hour_count']} событий). "
         "Адреса могут пересекаться между типами, поэтому эти количества не складываются в число пользователей. Одни сутки не экстраполируются на год.")]
    return {key:{"title":title,"question":q,"metric":m,"interpretation":text} for key,title,q,m,text in rows}


def conclusions(b):
    s=b["summary"]
    sections=describe_figures(b)
    return [
        f"TVL Aave изменился на {percent(s['tvl_period_change'])}, с {s['tvl_start_usd']/1e9:.2f} до {s['tvl_end_usd']/1e9:.2f} млрд USD. "
        f"При этом 90-дневное изменение на конец периода положительное ({percent(s['tvl_change_90d'])}), 30-дневное — {percent(s['tvl_change_30d'])}; траектория не монотонна. "
        f"На Ethereum приходится {percent(s['chain_largest_share'])} TVL в конечную дату.",
        sections["02_competitors"]["interpretation"],
        sections["04_economics"]["interpretation"] + " " +
        "В дневных изменениях связь TVL с предшествующим изменением fees/revenue слабая и отрицательная "
        "в выбранном выравнивании (точные r и n приведены в таблице корреляций). Простая пропорциональность на дневной частоте не подтверждается.",
        sections["05_lending_snapshot"]["interpretation"],
        sections["06_token_prices"]["interpretation"] + f" Волатильность с годовым приведением {percent(s['aave_vol_annual'])}, Pearson доходностей с ETH {s['token_corr']:.3f}.",
        sections["08_valuation"]["interpretation"],
        sections["10_onchain_case"]["interpretation"]]


LIMITATIONS = [
    "CoinGecko покрывает 345 дат, 21.11.2025–31.10.2026; первые 20 дней целевого года отсутствуют. Доходности — 344 суточных интервала.",
    "The Graph и on-chain выборка покрывают только 31.10.2026. Годовую динамику supply/borrow/utilization или активности пользователей проверить нельзя.",
    "Snapshot The Graph отражает состояние сущностей на историческом блоке и разные lastUpdate. Опубликованный utilizationRate не полностью воспроизводится простыми отношениями возвращённых балансов; независимая сверка резервных остатков не выполнялась.",
    "DeFiLlama TVL объединяет версии и сети; fees/revenue включают версии из linkedProtocols, состав агрегатов может различаться. Fees включают проценты, ликвидации, flash loans и другие поступления; revenue — не чистая прибыль и не доход владельцев AAVE.",
    "Daily fees/revenue — потоки с календарной меткой провайдера, TVL/цены — снимки около 00:00 UTC. Fees/TVL и Revenue/TVL — описательные дневные flow/stock отношения, без годового приведения и гарантии идентичного состава агрегатов.",
    "Корреляции не причинны. USD-TVL включает рыночную переоценку; совместные тренды и отдельные выбросы могут влиять на связи. Полная модель кредитного спроса не строилась.",
    "Новые сети имеют пропуски до начала доступной истории; они не заменялись нулями. Сетевое распределение вычислено на историческую конечную дату, не по текущим данным."]


def report_text(b):
    lines=["# Этап 4 — полноценный DeFi EDA Aave", "", "Этап 4 завершён. PDF и самостоятельный submission notebook пока не создавались.", "",
           "## Покрытие данных", "", "| Набор | Источник | Наблюдения | Частота | Пропуски |", "| --- | --- | ---: | --- | ---: |"]
    for row in b["coverage"].to_dict("records"):
        lines.append(f"| {row['Набор']} | {row['Источник']} | {row['Наблюдения']} | {row['Частота']} | {row['Пропуски']} |")
    lines += ["", "Полные периоды, поля, единицы и ограничения — в таблице Data Coverage внутри notebook и `data/processed/eda/coverage.csv`. "
              "TVL и fees/revenue: 01.11.2025–31.10.2026, 365 дней. Котировки: 21.11.2025–31.10.2026, 345 дат. The Graph и события: 31.10.2026.", "",
              "## Рассчитанные показатели и 10 основных графиков", ""]
    for key,section in describe_figures(b).items():
        lines += [f"### {key[:2]}. {section['title']}", "", "**Вопрос.** "+section["question"], "", "**Метрика.** "+section["metric"], "", "**Результат.** "+section["interpretation"], ""]
    lines += ["## Семь предварительных выводов: наблюдаемые результаты", ""]
    lines.extend(f"{i}. {text}" for i,text in enumerate(conclusions(b),1))
    lines += ["", "## Возможные объяснения — гипотезы", "",
              "- Снижение USD-TVL может отражать одновременно переоценку залога и изменение количества активов; без декомпозиции нельзя отделить отток от ценового эффекта.",
              "- Снижение fees/revenue может быть связано с остатками долга, ставками, ликвидациями и составом рынков. Однодневный lending snapshot не позволяет различить эти механизмы за год.",
              "- Общие рыночные факторы могут объяснять часть связи AAVE/ETH и USD-TVL; причинное влияние одной метрики на другую не установлено.", "",
              "## Ограничения", ""]
    lines.extend("- "+text for text in LIMITATIONS)
    lines += ["", "## Что остаётся перед финальной сборкой", "",
              "- Зафиксировать описанные интервалы и ограничения в итоговом документе; не выдавать однодневные данные за годовые.",
              "- На этапе 5 встроить необходимый код и результаты в самостоятельный notebook, затем сформировать PDF из этих же расчётов и графиков и сверить числа. На этапе 4 этого не делалось.",
              "- Для расширения выводов о годовой кредитной активности потребуется отдельный сбор истории The Graph; это дополнительная работа, не выполненная в этом этапе.", "",
              "## Воспроизведение", "", "```powershell", ".\\.venv\\Scripts\\python.exe -m src.stage4 --offline", "```", "",
              "Результаты: `notebooks/exploration.ipynb`, таблицы и аудит в `data/processed/eda/`, 10 PNG в `data/processed/eda/figures/` (220 dpi). "
              "Модули `src/metrics.py` и `src/plotting.py` содержат расчёты и визуализации; `src/stage4.py` — сборку notebook и этого отчёта. "
              "Первичные файлы прежних этапов сохранены; новые fees/revenue загружены один раз и кэшированы.", "",
              "Методика APY: [официальный Aave subgraph README](https://github.com/aave/protocol-subgraphs/blob/main/README.md). "
              "Определения fees/revenue: сохранённые `childProtocols.methodology` и `breakdownMethodology` в JSON DeFiLlama, "
              "[адаптер Aave V3](https://github.com/DefiLlama/dimension-adapters/blob/master/fees/aave-v3.ts)."]
    return "\n".join(lines)+"\n"


def build_notebook():
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    cells=[md("# DeFi EDA протокола Aave — этап 4\n\n"
              "**Исследовательский вопрос:** сопровождается ли изменение TVL изменением экономической активности и существует ли наблюдаемая связь с AAVE? "
              "Сравнение с Morpho и Compound. Данные реальны; отдельно рассматриваются годовые ряды и однодневный case study. "
              "Пять источников проверены до начала анализа. Dune не обязателен. PDF и финальная автономная сборка — следующий этап."),
           md("## Источники и воспроизводимость\n\n"
              "DeFiLlama: родительские TVL, сети, daily fees/revenue. CoinGecko: AAVE/ETH в USD. The Graph: Aave V3 Ethereum core Pool. "
              "web3.py через Ethereum RPC и Etherscan API V2: одинаковые 3260 событий за 31.10.2026. "
              "Notebook читает локальный кэш, проверяет контрольные суммы и не выполняет API-запросов. Использование `src/` допустимо для исследовательской версии; "
              "самостоятельный notebook будет собран на этапе 5.\n\n"
              "Первичные методики: [DeFiLlama API](https://api-docs.defillama.com/), "
              "[определения Aave V3 fees/revenue](https://github.com/DefiLlama/dimension-adapters/blob/master/fees/aave-v3.ts), "
              "[Aave subgraph и формула APY](https://github.com/aave/protocol-subgraphs/blob/main/README.md). "
              "URL запросов, определения childProtocols и время загрузки сохранены вместе с JSON-кэшами."),
           code("from pathlib import Path\nimport sys, json\nimport pandas as pd\nfrom IPython.display import display, Markdown, Image\n"
                "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src' / 'config.py').exists())\n"
                "sys.path.insert(0, str(ROOT))\nfrom src.metrics import prepare_eda, EDA_DIR\n"
                "from src.plotting import make_eda_plots\nfrom src.stage4 import describe_figures, conclusions, LIMITATIONS\n"
                "pd.set_option('display.max_colwidth', None)\nb = prepare_eda()\n"
                "gate = json.loads((ROOT / 'data/raw/onchain/stage3_manifest.json').read_text(encoding='utf-8'))\n"
                "display(pd.DataFrame([{'Источники': ', '.join(gate['instruments_with_real_data']), 'Готовность': gate['analysis_ready']}]))"),
           md("## Data Coverage\n\nПропуски считаются на целевом календаре и не заменяются нулями. "
              "Для новых сетей они включают период до начала истории. Неприменимые роли в событиях — структурные пустые поля, а не потерянные данные."),
           code("display(b['coverage'].style.set_properties(**{'text-align': 'left', 'white-space': 'normal'}))"),
           md("## Расчёты и единицы\n\n"
              "TVL change: TVL(t)/TVL(t−k)−1, k=30/90 календарных дней. Доля Aave относится только к выбранной тройке. "
              "Доходности — ln(Pt/Pt−1), волатильность — выборочная std × √365. Капитализация/TVL — строго на общих UTC-датах. "
              "Fees/revenue — потоки за сутки, делённые на TVL с той же меткой даты; 7-дневное среднее требует 7 реальных значений. "
              "Для корреляций дневные потоки t−1 сопоставляются с изменениями снимков в 00:00 t. "
              "Балансы разных токенов не складываются; преобразование ray APR в APY соответствует официальной формуле Aave и предполагает постоянную ставку."),
           code("display(pd.DataFrame([b['summary']]).drop(columns=['participants_by_type','competitor_changes']).T.rename(columns={0:'Значение'}))\n"
                "display(b['protocol'][['tvl_change_30d','tvl_change_90d','aave_share_three']].tail())\n"
                "display(b['tokens'][['aave_log_return','eth_log_return','aave_rolling_30d_vol_annual','market_cap_tvl']].dropna().head())\n"
                "display(b['correlations'])\ndisplay(b['lending'])\n"
                "display(b['chain_latest'])\npaths = make_eda_plots(b)\nsections = describe_figures(b)\nassert len(paths) == 10")]
    for key,title in [("01_tvl","TVL"),("02_competitors","Конкуренты"),("03_chains","Сети"),("04_economics","Экономическая активность"),
                      ("05_lending_snapshot","Lending snapshot"),("06_token_prices","Цена токена"),("07_returns_volatility","Доходности и риск"),
                      ("08_valuation","Market Cap / TVL"),("09_associations","Проверка исследовательского вопроса"),("10_onchain_case","Однодневный case study")]:
        cells.extend([md(f"## График {int(key[:2])}. {title}"),code(
            f"section = sections['{key}']\n"
            "display(Markdown('**Исследовательский вопрос:** ' + section['question']))\n"
            "display(Markdown('**Метрика:** ' + section['metric']))\n"
            f"display(Image(filename=str(paths['{key}'])))\n"
            "display(Markdown('**Интерпретация:** ' + section['interpretation']))")])
    cells.extend([md("## Предварительные выводы — наблюдаемые результаты"),code(
        "display(Markdown('\\n\\n'.join(f'{i}. {text}' for i,text in enumerate(conclusions(b),1))))"),
        md("## Возможные объяснения\n\n"
           "Переоценка активов и чистые изменения их количества могут одновременно влиять на TVL. "
           "Процентные ставки, долговые остатки, ликвидации и состав рынков могут влиять на fees/revenue. "
           "Общий рынок может частично объяснять движение AAVE и ETH. Эти объяснения являются гипотезами; причинность не установлена."),
        md("## Ограничения и границы выводов"),code("display(Markdown('\\n\\n'.join('- '+text for text in LIMITATIONS)))"),
        md("## Контроль результатов и следующий этап\n\n"
           "Годовая динамика кредитных остатков не восстановлена из однодневной выборки. "
           "Следующая стадия — автономный notebook и PDF из этих же результатов; сейчас этап 5 не выполняется."),
        code("assert len(paths) == 10\nassert b['protocol'].index.is_unique\n"
             "assert b['tokens'].index.is_unique\nassert str(b['tokens'].index.tz) == 'UTC'\n"
             "assert b['events'].amount_raw.map(lambda x: isinstance(x,str)).all()\n"
             "assert not b['events'].duplicated(['tx_hash','log_index']).any()\n"
             "print('10 графиков; UTC, уникальность, точность amount_raw проверены. Этап 5 не выполнялся.')")])
    nb=nbformat.v4.new_notebook(cells=cells,metadata={"kernelspec":{"name":"python3","display_name":"Python 3 (.venv)","language":"python"},"eda_stage":4})
    return execute_notebook(nb, ROOT/"notebooks"/"exploration.ipynb")


def main():
    parser=argparse.ArgumentParser(description="Stage 4 EDA only; never creates a PDF")
    parser.add_argument("--offline",action="store_true",help="Use only verified local data; do not request missing fees/revenue")
    args=parser.parse_args()
    if not args.offline:
        collect_economics()
    b=prepare_eda()
    build_notebook()
    (ROOT/"STAGE4_REPORT.md").write_text(report_text(b),encoding="utf-8")
    print("Stage 4 complete: executed notebook, 10 main figures. No PDF generated.")


if __name__=="__main__":
    main()
