"""Bounded stage 4.1 collection: three reserves, 30 daily event observations."""
import argparse
import pandas as pd
import requests
from .config import Settings
from .metrics import RESERVES
from .onchain import load_snapshot, save_snapshot, RAW
from .onchain_sources import api_json

FIELDS = 'id timestamp totalATokenSupply totalCurrentVariableDebt totalPrincipalStableDebt availableLiquidity utilizationRate liquidityRate variableBorrowRate reserve { id underlyingAsset symbol decimals pool { id pool } }'

def collect_ties():
    cfg=Settings()
    if load_snapshot('graph_review_ties'): return
    parts=[]
    for day in pd.date_range('2026-10-02','2026-10-31'):
        data=load_snapshot('graph_review_'+day.strftime('%Y%m%d'))['data']
        for symbol in RESERVES:
            rows=data[symbol]
            if len(rows)>1 and rows[0]['timestamp']==rows[1]['timestamp']:
                r=rows[0];alias=symbol+day.strftime('%Y%m%d')
                parts.append(f'{alias}: reserveParamsHistoryItems(first:1000,block:{{number:26316188}},where:{{reserve:"{r["reserve"]["id"]}",timestamp:{r["timestamp"]}}}) {{ {FIELDS} }}')
    with requests.Session() as session:
        result=api_json(session,'POST',f'https://gateway.thegraph.com/api/subgraphs/id/{cfg.graph_subgraph}',headers={'Authorization':'Bearer '+cfg.graph_key},json={'query':'{'+ ' '.join(parts)+'}'})
    if result.get('errors') or any(len(rows)>=1000 for rows in result['data'].values()):raise RuntimeError('Incomplete tie query')
    save_snapshot('graph_review_ties',{'subgraph_id':cfg.graph_subgraph,'block':26316188,'purpose':'all observations at tied terminal timestamps'},result['data'])
    print('Tie groups',len(parts),'records',sum(map(len,result['data'].values())))

def collect(probe=False):
    cfg = Settings()
    snapshot = load_snapshot('graph_markets')
    reserves = {r['symbol']: r for r in snapshot['data']['reserves'] if r['symbol'] in RESERVES and r['underlyingAsset'].lower() == RESERVES[r['symbol']][0]}
    dates = pd.date_range('2026-10-02', '2026-10-31', tz='UTC')
    # Probe both endpoints first; the full run reuses these two responses.
    if probe:
        dates = dates[[0, -1]]
    for day in dates:
        name = 'graph_review_' + day.strftime('%Y%m%d')
        request = {'subgraph_id': cfg.graph_subgraph, 'block': 26316188, 'day': str(day), 'sampling': 'last_two_event_observations_per_reserve'}
        if load_snapshot(name, request):
            continue
        parts = []
        for symbol, r in reserves.items():
            parts.append(f'{symbol}: reserveParamsHistoryItems(first:2, orderBy:timestamp, orderDirection:desc, block:{{number:26316188}}, where:{{reserve:"{r["id"]}",timestamp_gte:{int(day.timestamp())},timestamp_lt:{int((day+pd.Timedelta(days=1)).timestamp())}}}) {{ {FIELDS} }}')
        query = '{ _meta(block:{number:26316188}) { hasIndexingErrors block {number hash} } ' + ' '.join(parts) + ' }'
        with requests.Session() as session:
            result = api_json(session, 'POST', f'https://gateway.thegraph.com/api/subgraphs/id/{cfg.graph_subgraph}', headers={'Authorization': 'Bearer '+cfg.graph_key}, json={'query': query})
        if result.get('errors') or 'data' not in result:
            raise RuntimeError('Graph query rejected; collection stopped')
        data = result['data']
        if data['_meta']['hasIndexingErrors'] or data['_meta']['block']['number'] != 26316188:
            raise RuntimeError('Historical indexing check failed')
        save_snapshot(name, request, data)
        print(day.strftime('%Y-%m-%d'), {s: len(data[s]) for s in reserves}, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--ties', action='store_true')
    args=parser.parse_args()
    if args.ties:collect_ties()
    else:collect(args.probe)
