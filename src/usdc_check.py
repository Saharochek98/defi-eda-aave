"""Three fixed historical blocks; no changes to the research data or quality flags."""
import requests
from decimal import Decimal
from web3 import Web3
from .config import Settings, ROOT
from .onchain import POOL, load_snapshot, save_snapshot, get_rpc
from .stage41 import event_order
from .onchain_sources import api_json

ASSET='0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48'
DATES=['20261002','20261015','20261029']

def selected():
    ties=load_snapshot('graph_review_ties')['data']
    return [(day,max(ties.get('USDC'+day,load_snapshot('graph_review_'+day)['data']['USDC']),key=event_order)) for day in DATES]

def collect():
    if load_snapshot('usdc_independent_check'):return
    cfg=Settings();results=[]
    for day,row in selected():
        block=event_order(row)[0]
        record={'day':day,'block':block,'graph_event':row,'attempts':[]}
        for label,url in [('configured',cfg.rpc_url or 'https://eth.drpc.org'),('publicnode','https://ethereum-rpc.publicnode.com')]:
            try:
                w=get_rpc(url)
                if w.eth.chain_id!=1:raise ValueError('wrong_chain')
                header=w.eth.get_block(block)
                if int(header.timestamp)!=int(row['timestamp']):raise ValueError('timestamp_mismatch')
                calls=[]
                def call(to,signature,argtypes=(),args=()):
                    data=Web3.keccak(text=signature)[:4]+w.codec.encode(argtypes,args)
                    result=w.eth.call({'to':Web3.to_checksum_address(to),'data':data},block_identifier=block)
                    calls.append({'to':to,'signature':signature,'input':Web3.to_hex(data),'output':Web3.to_hex(result)})
                    return result
                reserve=call(POOL,'getReserveData(address)',['address'],[ASSET])
                words=[reserve[i:i+32] for i in range(0,len(reserve),32)]
                if len(words)<15:raise ValueError('unexpected_reserve_abi')
                atoken=Web3.to_checksum_address('0x'+words[8][-20:].hex())
                debt=Web3.to_checksum_address('0x'+words[10][-20:].hex())
                if atoken.lower()!='0x98c23e9d8f34fefb1b7bd6a91b7ff122f4e16f5c':raise ValueError('unexpected_atoken')
                available=int.from_bytes(call(ASSET,'balanceOf(address)',['address'],[atoken]),'big')
                supply=int.from_bytes(call(atoken,'totalSupply()'),'big')
                borrow=int.from_bytes(call(debt,'totalSupply()'),'big')
                decimals=int.from_bytes(call(ASSET,'decimals()'),'big')
                if decimals!=6:raise ValueError('decimals_mismatch')
                optional={}
                try:optional['virtual_balance_raw']=str(int.from_bytes(call(POOL,'getVirtualUnderlyingBalance(address)',['address'],[ASSET]),'big'))
                except Exception as exc:optional['virtual_balance_status']=type(exc).__name__
                record['rpc']={'provider':label,'block_hash':Web3.to_hex(header.hash),'timestamp':int(header.timestamp),'aToken':atoken,'variableDebtToken':debt,
                    'available_raw':str(available),'supply_raw':str(supply),'borrow_raw':str(borrow),'decimals':decimals,'calls':calls,**optional}
                print(day,block,'RPC OK',flush=True);break
            except Exception as exc:
                record['attempts'].append({'provider':label,'error_type':type(exc).__name__})
                print(day,label,type(exc).__name__,flush=True)
        results.append(record)
    # Same-block Graph state distinguishes event-state timing from persisted state.
    parts=[]
    for day,row in selected():
        parts.append(f'd{day}: reserve(id:"{row["reserve"]["id"]}",block:{{number:{event_order(row)[0]}}}) {{ id availableLiquidity totalLiquidity totalATokenSupply totalCurrentVariableDebt utilizationRate lastUpdateTimestamp }}')
    with requests.Session() as s:
        g=api_json(s,'POST',f'https://gateway.thegraph.com/api/subgraphs/id/{cfg.graph_subgraph}',headers={'Authorization':'Bearer '+cfg.graph_key},json={'query':'{'+ ' '.join(parts)+'}'})
    if g.get('errors'):raise RuntimeError('Same-block Graph query failed')
    save_snapshot('usdc_independent_check',{'chain_id':1,'pool':POOL,'asset':ASSET,'dates':DATES,'subgraph_id':cfg.graph_subgraph}, {'observations':results,'same_block_graph':g['data']})

def report():
    data=load_snapshot('usdc_independent_check')['data'];lines=[]
    for r in data['observations']:
        g=data['same_block_graph']['d'+r['day']];rpc=r.get('rpc')
        lines.append({'date':r['day'],'block':r['block'],'graph_event_available':float(Decimal(r['graph_event']['availableLiquidity'])/10**6),
            'graph_block_available':float(Decimal(g['availableLiquidity'])/10**6),'graph_utilization':float(g['utilizationRate']),
            'rpc_available':float(Decimal(rpc['available_raw'])/10**6) if rpc else None,
            'rpc_supply':float(Decimal(rpc['supply_raw'])/10**6) if rpc else None,
            'rpc_borrow':float(Decimal(rpc['borrow_raw'])/10**6) if rpc else None})
    import pandas as pd
    frame=pd.DataFrame(lines);frame.to_csv(ROOT/'data/processed/eda/review/usdc_independent.csv',index=False)
    print(frame.to_string(index=False))
    return frame

if __name__=='__main__':collect();report()
