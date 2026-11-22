"""Offline quality review; bounded Graph collection is a separate explicit command."""
import json
from decimal import Decimal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nbformat
from .config import ROOT
from .metrics import prepare_eda, log_returns, lending_snapshot, RESERVES, EDA_DIR
from .onchain import load_snapshot, POOL
from .stage2 import execute_notebook

OUT = EDA_DIR / 'review'

def correlations(x, y):
    """Pairwise complete cases; symmetric 1/99% marginal trimming, no winsorization."""
    pair = pd.concat([x.rename('x'), y.rename('y')], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    low, high = pair.quantile(.01), pair.quantile(.99)
    kept = pair.loc[pair.ge(low).all(axis=1) & pair.le(high).all(axis=1)]
    def coefficient(frame, ranks=False):
        if len(frame)<3 or frame.nunique().min()<2:return np.nan
        if ranks:frame=frame.rank(method='average')
        return frame.x.corr(frame.y)
    return {'n': len(pair), 'pearson': coefficient(pair), 'spearman': coefficient(pair,True),
            'trim_n': len(kept), 'removed': len(pair)-len(kept), 'trim_pearson': coefficient(kept),
            'trim_spearman': coefficient(kept,True)}

def table(frame):
    rows=[list(frame.columns), ['---']*len(frame.columns), *frame.astype(str).values.tolist()]
    return '\n'.join('| '+' | '.join(row)+' |' for row in rows)

def normalized(frame):
    if frame.iloc[0].isna().any() or (frame.iloc[0] <= 0).any():
        raise ValueError('A positive common initial base is required')
    return frame.div(frame.iloc[0])*100

def event_order(row):
    parts=row['id'].split(':')
    if len(parts)!=5 or not parts[2].startswith('0x'):
        raise ValueError('Unrecognized event identity')
    return tuple(int(parts[i]) for i in [0,1,3,4])

def reserve_quality(frame):
    out=frame.copy()
    out['quality_invalid']=(out.available_token<0)|~out.utilization_reported.between(0,1)|(out.supply_token<0)|(out.borrow_token<0)
    out['utilization_valid']=out.utilization_reported.where(~out.quality_invalid)
    return out

def daily_lending():
    rows, ambiguities = [], []
    other = load_snapshot('graph_markets')['data']['reserves']
    ties = load_snapshot('graph_review_ties')['data']
    dates = pd.date_range('2026-10-02', '2026-10-31', tz='UTC')
    for day in dates:
        cache = load_snapshot('graph_review_'+day.strftime('%Y%m%d'))
        if cache is None:
            raise ValueError('Incomplete 30-day collection')
        for symbol in RESERVES:
            values = cache['data'][symbol]
            if not values:
                raise ValueError('Missing daily reserve observation')
            r = values[0]
            alias=symbol+day.strftime('%Y%m%d')
            if len(values)>1 and values[1]['timestamp']==r['timestamp']:
                group=ties[alias]
                if not group or len(group)>=1000 or any(v['timestamp']!=r['timestamp'] or v['reserve']['id']!=r['reserve']['id'] for v in group):
                    raise ValueError('Incomplete or mixed timestamp group')
                values=sorted(group,key=event_order,reverse=True)
                r=values[0]
            stamp = pd.to_datetime(int(r['timestamp']), unit='s', utc=True)
            if not day <= stamp < day + pd.Timedelta(days=1):
                raise ValueError('Observation outside UTC day')
            # Timestamp ties cannot be ordered by transaction hash: omit ambiguous states.
            keys = ['totalATokenSupply','totalCurrentVariableDebt','totalPrincipalStableDebt','availableLiquidity','utilizationRate','liquidityRate','variableBorrowRate']
            ambiguous = len(values)>1 and event_order(values[1])==event_order(r) and any(values[1][k]!=r[k] for k in keys)
            native = {**r, **r['reserve'], 'lastUpdateTimestamp': int(r['timestamp'])}
            records = [native if a['underlyingAsset'].lower()==RESERVES[symbol][0] else a for a in other]
            row = lending_snapshot(records).loc[symbol].to_dict()
            row.update(day=day, symbol=symbol, observation_utc=stamp, age_seconds=float((day+pd.Timedelta(days=1)-stamp).total_seconds()), ambiguous=ambiguous)
            if ambiguous:
                ambiguities.append((str(day),symbol))
                for k in ['supply_token','borrow_token','utilization_reported']:
                    row[k]=np.nan
            rows.append(row)
    return reserve_quality(pd.DataFrame(rows)), ambiguities

def review():
    OUT.mkdir(parents=True, exist_ok=True)
    b=prepare_eda(); p,t,e=b['protocol'],b['tokens'],b['economics']
    rawcols=[c for c in t if c.endswith('_usd')]
    missing=pd.DataFrame([{'column':c,'missing':int(t[c].isna().sum()),'first_missing':str(t.index[t[c].isna()].min()),
                          'last_missing':str(t.index[t[c].isna()].max()),'internal_gaps':int(t.loc[t[c].first_valid_index():t[c].last_valid_index(),c].isna().sum())} for c in rawcols])
    missing_dates=t[rawcols].isna().rename_axis('date_utc')
    rows=[]
    x=p.tvl_log_change
    for name,left,right in [('AAVE/ETH',t.aave_log_return,t.eth_log_return),('TVL/AAVE',x,t.aave_log_return)]:
        rows.append({'pair':name,'sample':'pairwise','shift':0,**correlations(left,right)})
    for metric in ['fees','revenue']:
        for shift in [0,1,2]:
            y=log_returns(e[metric+'_usd']).shift(shift)
            for sample, idx in [('full',p.index),('market_common',b['joint'].index)]:
                rows.append({'pair':'TVL/'+metric,'sample':sample,'shift':shift,**correlations(x.loc[idx],y.loc[idx])})
    corr=pd.DataFrame(rows)
    trio=normalized(p[['aave_tvl_usd','morpho_tvl_usd','compound_tvl_usd']])
    spread=trio.max(axis=1)-trio.min(axis=1)
    windows=[]
    for a,bb in [('aave_tvl_usd','morpho_tvl_usd'),('aave_tvl_usd','compound_tvl_usd'),('morpho_tvl_usd','compound_tvl_usd')]:
        gap=trio[a]-trio[bb]
        end=gap.diff(30).abs().idxmax()
        windows.append({'pair':a+'/'+bb,'start':str(end-pd.Timedelta(days=30)), 'end':str(end), 'gap_change_index_points':float(gap.diff(30).loc[end]), 'max_abs_gap_date':str(gap.abs().idxmax()),'max_abs_gap':float(gap.abs().max())})
    lending,ambiguities=daily_lending()
    definition=[]
    for r in load_snapshot('graph_review_definition')['data']['reserves']:
        if (r['pool'].get('pool') or '').lower()!=POOL.lower(): continue
        matches=[s for s,(asset,_) in RESERVES.items() if asset==r['underlyingAsset'].lower()]
        if not matches: continue
        total,available=Decimal(r['totalLiquidity']),Decimal(r['availableLiquidity'])
        formula=Decimal(1)-available/total
        definition.append({'symbol':matches[0], 'reported':float(r['utilizationRate']), 'one_minus_available_totalLiquidity':float(formula),
                           'error_pp':float((Decimal(r['utilizationRate'])-formula)*100),
                           'totalLiquidity_token':float(total/Decimal(10)**RESERVES[matches[0]][1]),
                           'aTokenSupply_minus_totalLiquidity_token':float((Decimal(r['totalATokenSupply'])-total)/Decimal(10)**RESERVES[matches[0]][1])})
    definition=pd.DataFrame(definition)
    history=pd.json_normalize(load_snapshot('graph_markets')['data']['reserveParamsHistoryItems'])
    frequency=[]
    for symbol,(asset,_) in RESERVES.items():
        h=history.loc[history['reserve.underlyingAsset'].str.lower().eq(asset)]
        stamps=pd.to_numeric(h.timestamp).sort_values().drop_duplicates()
        frequency.append({'symbol':symbol,'day':'2026-10-31','event_observations':len(h),'unique_timestamps':len(stamps),'median_gap_seconds':float(stamps.diff().median()),'max_gap_seconds':float(stamps.diff().max())})
    result={'missing':missing,'missing_dates':missing_dates,'correlations':corr,'normalized':trio,'divergence_windows':pd.DataFrame(windows),
            'lending_daily':lending,'definition':definition,'frequency':pd.DataFrame(frequency)}
    result['alignment']=pd.DataFrame({'snapshot_end_utc':p.index,'snapshot_start_utc':p.index-pd.Timedelta(days=1),
        'flow_day_shift1':p.index-pd.Timedelta(days=1),'tvl_log_change':p.tvl_log_change,
        'aave_log_return':t.aave_log_return,'fees_log_change_shift1':log_returns(e.fees_usd).shift(1),
        'revenue_log_change_shift1':log_returns(e.revenue_usd).shift(1)})
    result['summary']={'max_spread_date':str(spread.idxmax()),'max_spread_points':float(spread.max()),'end_indices':trio.iloc[-1].to_dict(),
                       'ambiguous_daily_observations':ambiguities,'max_observation_age_seconds':float(lending.age_seconds.max()),
                       'graph_queries':32,'graph_daily_rows':len(lending),'graph_dates':30,
                       'invalid_reserve_days':int(lending.quality_invalid.sum()),
                       'resolved_timestamp_groups':len(load_snapshot('graph_review_ties')['data'])}
    for name,value in result.items():
        if isinstance(value,pd.DataFrame):value.to_csv(OUT/(name+'.csv'),index=True)
    (OUT/'summary.json').write_text(json.dumps(result['summary'],ensure_ascii=False,indent=2),encoding='utf-8')
    return result

def plot_history(r):
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
    for i,s in enumerate(RESERVES):
        h=r['lending_daily'].query('symbol == @s').set_index('day')
        scale=1e6 if s!='WETH' else 1e3
        for column,label in [('supply_token','Supply'),('borrow_token','Borrow')]:
            axes[i,0].plot(h.index,h[column]/scale,label=label)
        axes[i,0].legend()
        axes[i,0].set(title=s,ylabel=('млн ' if s!='WETH' else 'тыс. ')+s)
        axes[i,1].plot(h.index,h.utilization_valid*100,color='#328770',marker='.',label='В пределах [0,100%]')
        bad=h.loc[h.quality_invalid]
        if len(bad):
            axes[i,1].scatter(bad.index,bad.utilization_reported*100,color='red',marker='x',label='Аномалия источника')
            axes[i,1].axhline(100,color='gray',linestyle=':')
            axes[i,1].legend(fontsize=8)
            axes[i,0].set_title(s+' — остатки subgraph, требуется сверка')
        axes[i,1].set(title=s+' — utilization subgraph',ylabel='%')
    fig.suptitle('02–31.10.2026: последние событийные наблюдения суток (UTC)')
    import matplotlib.dates as mdates
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        ax.set_xlabel('Дата UTC')
    fig.tight_layout(); path=OUT/'11_lending_30d.png'
    fig.savefig(path,dpi=220,bbox_inches='tight');plt.close(fig)
    return path

def report(r):
    c=r['correlations'];s=r['summary']
    daily=r['lending_daily']
    changes=[]
    for sym in RESERVES:
        h=daily.query('symbol == @sym').sort_values('day')
        utilization=(f"utilization {h.utilization_reported.iloc[0]:.2%} → {h.utilization_reported.iloc[-1]:.2%}" if not h.quality_invalid.any() else 'вывод о utilization отклонён: аномальные значения источника')
        changes.append(f"- {sym}: supply {h.supply_token.iloc[-1]/h.supply_token.iloc[0]-1:+.2%}, borrow {h.borrow_token.iloc[-1]/h.borrow_token.iloc[0]-1:+.2%} по subgraph; {utilization}.")
    return '\n\n'.join([
        '# Этап 4.1 — Quality Review & Research Improvements',
        '## Пропуски CoinGecko',
        table(r['missing']),
        'Все 120 пропусков — 20 дат 01–20.11.2025 включительно × 6 исходных столбцов. Внутренних разрывов нет. Полная матрица дат: `missing_dates.csv`. '
        'Из-за ограничения бесплатной истории доходность AAVE/ETH относится к 21.11.2025–31.10.2026 (345 цен, 344 доходности), а не ко всему году. '
        'Истинные годовые доходность и волатильность неизвестны; влияние отсутствующих дней нельзя измерить без котировок. '
        'Первый rolling30 volatility появляется 21.12.2025: 20 отсутствующих дат + 1 лаг + 29 прогрева = 50 NaN. '
        'Market Cap/TVL имеет 345 наблюдений; TVL, fees/revenue и их годовые итоги от CoinGecko не зависят. '
        'Старые корреляции TVL/fees/revenue использовали общий market sample из 344 строк; теперь дополнительно рассчитана полная попарная выборка.',
        '## Временное выравнивание',
        'Исходные TVL и цены внутри периода имеют метки 00:00 UTC без дублей и внутридневных сдвигов. Δlog TVL(t) и доходность цены(t) относятся к интервалу (t−1,t]. '
        'Fees/revenue — потоки, помеченные датой суток. При трактовке F(d) как потока [d,d+1) изменение log F(t−1) выравнивается с изменением остатков, заканчивающимся в t. '
        'Поэтому основной shift=1 сохранён; shift=0 использует ещё не завершившиеся относительно снимка t сутки, shift=2 — предшествующие сутки. '
        'Все три варианта приведены как чувствительность, лучший коэффициент не выбирается. Например, доходность на метке 03.10 относится к 02.10 00:00–03.10 00:00 и сопоставляется с изменением потока за 02.10 относительно 01.10. '
        'Проверенный runAdapter.ts задаёт fromTimestamp=end−window, toTimestamp=end−1 и dateString=floorUTC(toTimestamp), что поддерживает маркировку по дню потока. Таблица alignment.csv сохраняет соответствие интервалов и значений. '
        'Документация адаптеров подтверждает интервальную природу потоков, '
        'но API не возвращает исходные времена снимков всех дочерних протоколов: точная синхронность агрегатов не доказана. '
        'Fees/TVL с той же датой — поток суток к остатку на их начало; это описательный коэффициент, не доходность и не APY. '
        'UTC-метки агрегатора не доказывают, что каждый дочерний TVL был фактически измерен ровно в полночь.',
        '## Корреляции и выбросы',
        'Pearson и Spearman рассчитаны на одной выборке каждой строки. Trim: удаляется объединение наблюдений ниже 1-го или выше 99-го перцентиля любого из двух рядов; '
        'пороги вычисляются по исходной попарной выборке, даты и исходные данные не изменяются. Это проверка чувствительности, не способ улучшить результат. '
        'Работаем с логарифмическими изменениями, не с трендовыми уровнями. Доверительные интервалы и причинные выводы не заявляются.',
        table(c.round(4)),
        '## Конкуренты: единая база 100',
        'График 02 дополнен третьей панелью: TVL / TVL(01.11.2025) × 100; прежние абсолютные TVL и доля Aave сохранены. '
        f"Максимальный размах max−min: {s['max_spread_points']:.2f} пункта, {s['max_spread_date'][:10]}. Конечные индексы: {s['end_indices']}. "
        'Ниже для каждой пары показано окно с максимальным абсолютным изменением разрыва за 30 календарных дней; это заранее определённый критерий расхождения, не визуальный выбор.',
        table(r['divergence_windows'].round(3)),
        '## The Graph: 30-дневный lending case study',
        'Проба двух крайних дат подтвердила все три резерва. Затем выполнены 28 запросов остальных дат и один запрос определения utilization: всего 31 HTTP-запрос без автоматических повторов. '
        'В каждом дневном запросе три alias с first:2; выгружаются только последние два события, не вся история. '
        'Ещё одним запросом получены все события в конечных секундах с совпадениями; итоговый объём — 32 запроса. '
        'Документация The Graph указывает 100 000 бесплатных запросов в месяц; 32 запроса составляют 0.032% этого лимита. Остаток квоты/счёт аккаунта недоступен, нулевая фактическая стоимость не подтверждается. Оплата и подписка не подключались.',
        f"Период 02–31.10.2026: {len(daily)} reserve-day строк, 30 дат × 3 резерва; блок 26316188, прежний проверенный Subgraph ID и Ethereum core Pool. "
        f"Максимальный возраст наблюдения до конца суток: {s['max_observation_age_seconds']/60:.2f} мин. Неоднозначные последние timestamp: {s['ambiguous_daily_observations']}. "
        f"Разрешено {s['resolved_timestamp_groups']} групп совпадающих timestamp: все события секунды сортируются численно по blockNumber, transactionIndex, logIndex, transactionLogIndex согласно официальному id-generation.ts. "
        'Если одна идентичность имела бы конфликтующие значения, график оставлял бы пропуск. '
        'Это последнее обновление суток, не точный balanceOf на конец дня и не среднесуточная utilization. '
        'Индексация событийная; нижеследующая оценка частоты относится только к полной ранее собранной дате, не экстраполируется на месяц.',
        table(r['frequency'].round(1)),
        f"**Нарушения качества: {s['invalid_reserve_days']} из 90 точек.** USDC имеет отрицательную availableLiquidity и utilization >100% в 25 датах (02–20, 22–25, 27, 29 октября). "
        'Исходные значения сохранены, quality_invalid=True; utilization_valid оставляет NaN вместо них, без обрезания до 100%. Красные кресты на графике показывают аномальные данные поставщика. '
        'WETH/USDT таких нарушений не имеют. Эти проверки не доказывают точность остальных балансов. Изменения supply/borrow USDC остаются описанием subgraph, '
        'не подтверждённым изменением реальных остатков. Причина аномалии не установлена; необходима отдельная сверка с историческими вызовами контрактов или другим deployment. '
        'Проверенный snapshot 31.10 сохранён; полная валидная 30-дневная utilization USDC не получена.',
        '\n'.join(changes),
        '## APR, APY и остатки',
        'APR = rate / 10^27; APY = expm1(31536000 × log1p(APR / 31536000)). При APR=5% получается APY≈5.1271%. '
        'Это эквивалент при постоянной ставке и ежесекундном реинвестировании по frontend-конвенции Aave; не фактически реализованная доходность за месяц. '
        'USDC/USDT масштабируются на 10^6, WETH на 10^18. Балансы разных активов не складываются. '
        'Stable principal для выбранных резервов равен нулю; поэтому borrow здесь = current variable debt. Ненулевой stable principal нельзя подменять начисленным stable debt.',
        'В официальном mapping utilization = truncate8(1 − availableLiquidity / totalLiquidity). '
        'totalLiquidity не равен totalATokenSupply; поэтому прежнее расхождение с Borrow/Supply само по себе не означает ошибку ray или decimals. '
        'Сверка на том же историческом блоке:',
        table(r['definition']),
        'Проверка формулы не является независимой RPC-сверкой балансов. Поля subgraph обновляются обработчиками событий и могут отличаться по моментам актуализации; '
        'supply − borrow не следует принудительно приравнивать availableLiquidity. Исходный код main служит методологическим ориентиром; его идентичность байт-коду deployment не доказана.',
        '## Что изменилось в интерпретации',
        'Годовые TVL, fees/revenue и наблюдаемая доходность токенов не пересчитываются на вымышленные даты. '
        'Положительная связь AAVE/ETH сохраняется: Pearson 0.709 → 0.654 после trim, Spearman 0.709 → 0.683. '
        'TVL/AAVE также остаётся положительной: Pearson 0.599 → 0.495, Spearman 0.578 → 0.532. '
        'Для TVL/fees (полная выборка, shift=1) Pearson −0.180 → −0.007, Spearman −0.073 → −0.019: отрицательная связь неустойчива к выбросам. '
        'TVL/revenue: Pearson −0.117 → −0.165, но Spearman всего −0.048 → −0.043; сильной монотонной связи нет. '
        'Смена сдвига меняет величину и иногда знак: оснований для причинной интерпретации нет. '
        '30-дневные резервы позволяют обсуждать октябрьскую динамику выбранных активов, но не объясняют годовое изменение TVL всех сетей. '
        'История USDC ограничена выявленными аномалиями и не используется для вывода об изменении загрузки резерва. '
        'Определение utilization уточнено отдельно от простого Borrow/Supply.',
        '## Воспроизведение и границы этапа',
        '`python -m src.stage41` выполняет review и исследовательский notebook исключительно по кэшам. '
        'Существующие 10 графиков сохранены (02 улучшен); добавлен один отдельный график 30-дневных резервов. '
        'PDF и submission notebook не создаются. Таблицы и аудит: `data/processed/eda/review/`.',
        'Источники методики: [Aave APY](https://github.com/aave/protocol-subgraphs/blob/main/README.md), '
        '[Aave utilization mapping](https://github.com/aave/protocol-subgraphs/blob/main/src/helpers/reserve-logic.ts), '
        '[DeFiLlama интервалы](https://docs.llama.fi/list-your-project/other-dashboards), '
        '[DeFiLlama runner](https://github.com/DefiLlama/dimension-adapters/blob/master/adapters/utils/runAdapter.ts), '
        '[Aave порядок событий](https://github.com/aave/protocol-subgraphs/blob/main/src/utils/id-generation.ts), '
        '[The Graph квота](https://thegraph.com/docs/en/subgraphs/providers/subgraph-studio/introduction/).'])+'\n'

def main():
    r=review();text=report(r)
    (ROOT/'STAGE4_1_REPORT.md').write_text(text,encoding='utf-8')
    path=ROOT/'notebooks/exploration.ipynb'
    nb=nbformat.read(path,as_version=4)
    nb.cells=[c for c in nb.cells if not c.metadata.get('stage41')]
    for cell in [nbformat.v4.new_markdown_cell('## Этап 4.1 — дополнительная проверка\n\nЭтот раздел уточняет выводы этапа 4 и имеет приоритет в вопросах utilization и корреляционной чувствительности.'),
                 nbformat.v4.new_code_cell("from src.stage41 import review, report, plot_history\nr = review()\ndisplay(Markdown(report(r)))\ndisplay(Image(filename=str(plot_history(r))))")]:
        cell.metadata['stage41']=True;nb.cells.append(cell)
    nb.metadata['eda_stage']='4.1'
    execute_notebook(nb,path)
    print('Stage 4.1 complete; no PDF or submission notebook.')

if __name__=='__main__':main()
