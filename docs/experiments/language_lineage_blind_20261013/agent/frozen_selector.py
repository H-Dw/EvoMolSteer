import re,json,hashlib
from pathlib import Path

BASE=Path(__file__).parent; OUT=BASE/'answers'; OUT.mkdir(exist_ok=True)
starts=[1,5,10,20,30]

def floats(s): return [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',s)]
def parse_block(txt, ident):
    d={'id':ident}
    m=re.search(r'当前噪声/生成槽位云：[^\n]*回转半径 ([\d.-]+).*?三个固定轴的二阶矩\(([^)]+)\).*?2.5 Å短程对核 ([\d.-]+)，7 Å长程对核 ([\d.-]+).*?最短槽位间距 ([\d.-]+)',txt,re.S)
    if not m: return None
    d['cur_rg']=float(m.group(1)); d['cur_mom']=floats(m.group(2)); d['cur_pair2']=float(m.group(3)); d['cur_pair7']=float(m.group(4)); d['cur_gap']=float(m.group(5))
    m=re.search(r'此刻模型预测的终点云：[^\n]*回转半径 ([\d.-]+).*?三个固定轴的二阶矩\(([^)]+)\).*?2.5 Å短程对核 ([\d.-]+)，7 Å长程对核 ([\d.-]+).*?最短槽位间距 ([\d.-]+)',txt,re.S)
    d['fc_rg']=float(m.group(1)); d['fc_mom']=floats(m.group(2)); d['fc_pair2']=float(m.group(3)); d['fc_pair7']=float(m.group(4)); d['fc_gap']=float(m.group(5))
    for kind,pat in [('cur',r'区域分布：R00的高斯质量 ([\d.-]+)、局部偏移\(([^)]+)\) Å、3 Å壳层核 ([\d.-]+)；R05的高斯质量 ([\d.-]+)、局部偏移\(([^)]+)\) Å、3 Å壳层核 ([\d.-]+)；R15的高斯质量 ([\d.-]+)、局部偏移\(([^)]+)\) Å、3 Å壳层核 ([\d.-]+)'),('fc',r'区域分布：R00的高斯质量 ([\d.-]+)、局部偏移\(([^)]+)\) Å、3 Å壳层核 ([\d.-]+)；R05的高斯质量 ([\d.-]+)、局部偏移\(([^)]+)\) Å、3 Å壳层核 ([\d.-]+)；R15的高斯质量 ([\d.-]+)、局部偏移\(([^)]+)\) Å、3 Å壳层核 ([\d.-]+)')]:
        # first region line is current, second is forecast
        ms=list(re.finditer(pat,txt))
        mm=ms[0 if kind=='cur' else 1]
        vals=[]
        for j in (1,4,7): vals += [float(mm.group(j)),*floats(mm.group(j+1)),float(mm.group(j+2))]
        d[kind+'_regions']=vals
    m=re.search(r'当前到预测终点的同槽位剩余RMS为 ([\d.-]+) Å；本次沿真实父子边的当前位移RMS为 ([\d.-]+) Å，终点预测修订RMS为 ([\d.-]+)',txt)
    d['rms']=floats(m.group(0))[:3] if m else []
    return d

def cards(path, proto=False):
    s=path.read_text(encoding='utf-8'); out=[]
    heads=list(re.finditer(r'^## (P\d+|C[0-9a-f]+)',s,re.M))
    for i,h in enumerate(heads):
        b=s[h.start(): heads[i+1].start() if i+1<len(heads) else len(s)]
        x=parse_block(b,h.group(1))
        if x:
            pm=re.search(r'已发生的父子边来自 ([A-Za-z0-9]+)',b)
            if pm:x['parent']=pm.group(1)
            out.append(x)
    return out

def vec(d):
    return [d['cur_rg'],*d['cur_mom'],d['cur_pair2'],d['cur_pair7'],d['cur_gap'],*d['cur_regions'],d['fc_rg'],*d['fc_mom'],d['fc_pair2'],d['fc_pair7'],d['fc_gap'],*d['fc_regions'],*d['rms']]
def score(a,b):
    # Current and forecast each receive equal mass; RMS terms are secondary.
    va,vb=vec(a),vec(b); scales=[]
    for i in range(len(va)):
        if i<5: scales.append(.08)
        elif i<22: scales.append(.12)
        elif i<27: scales.append(.25)
        elif i<44: scales.append(.18)
        else: scales.append(.35)
    return sum(abs(x-y)/sc for x,y,sc in zip(va,vb,scales))/len(va)
def evidence(a,p,sc):
    return f"current Rg={a['cur_rg']:.3f} vs P{p['id'][1:]} {p['cur_rg']:.3f}; forecast Rg={a['fc_rg']:.3f} vs {p['fc_rg']:.3f}; R00 mass={a['fc_regions'][0]:.3f}; score={sc:.3f}. 反例：R05/R15占据与短程/长程核仍可能偏离；几何匹配不表示能量或亲和力。"
def rank(states,protos):
    rows=[]
    for a in states:
        ds=[(score(a,p),p) for p in protos]; sc,p=min(ds,key=lambda z:z[0])
        rows.append((sc,a,p))
    rows.sort(key=lambda z:(z[0],z[1]['id']))
    return rows[:5]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    log=[]; protos={}
    for t in range(1,51):
        fp=BASE/f'prototypes_{t:03d}.zh-CN.md'; protos[t]=cards(fp,True); log.append({'action':'read','file':fp.name})
    initial={};
    # Seal each start ranking before reading any later state card.
    for step in starts:
        fp=BASE/f'states_{step:03d}.zh-CN.md'; st=cards(fp); log.append({'action':'read','file':fp.name})
        rr=rank(st,protos[step]); rows=[]
        for i,(sc,a,p) in enumerate(rr,1): rows.append({'rank':i,'candidate_id':a['id'],'confidence':'medium' if sc<.8 else 'low','matched_description':evidence(a,p,sc),'counterevidence':'局部区域与预测核仍有残差；早期槽位云不等于成品分子。'})
        obj={'t':step/100,'method_summary':'等权当前/预测几何描述匹配；局部R00/R05/R15、Rg/二阶矩、短长程核、间距及父子位移RMS，未使用标签。','ranking':rows}
        out=OUT/f'ranking_{step:03d}.json'; out.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); log.append({'action':'seal','file':out.name,'sha256':sha(out)})
        initial[step]=rows
    # sequential continuation; each node is written before its next state file is read
    for step in starts:
        curr=[x['candidate_id'] for x in initial[step]]; nodes=[{'t':step/100,'eligible_count':len(curr),'ranking':initial[step],'status':'retained'}]; stop=step
        for nxt in range(step+1,51):
            prevout=OUT/f'continuation_{step:03d}.json'
            # Seal current node in the in-memory log by writing the continuation file before reading next card.
            pre={'start_t':step/100,'nodes':nodes,'stop_t':stop}; prevout.write_text(json.dumps(pre,ensure_ascii=False,indent=2),encoding='utf-8')
            fp=BASE/f'states_{nxt:03d}.zh-CN.md'; st=cards(fp); log.append({'action':'read','file':fp.name})
            eligible=[a for a in st if a.get('parent') in curr]
            if not eligible:
                nodes.append({'t':nxt/100,'eligible_count':0,'ranking':[],'status':'empty','disappearance_reason':'保留方向在下一状态卡中没有可见真实子状态。'}); stop=nxt/100; break
            rr=rank(eligible,protos[nxt]); rows=[]
            for i,(sc,a,p) in enumerate(rr,1): rows.append({'rank':i,'candidate_id':a['id'],'match_score':round(sc,6),'evidence':evidence(a,p,sc)})
            nodes.append({'t':nxt/100,'eligible_count':len(eligible),'ranking':rows,'status':'retained'}); curr=[x['candidate_id'] for x in rows]; stop=nxt/100
        final={'start_t':step/100,'nodes':nodes,'stop_t':stop}; out=OUT/f'continuation_{step:03d}.json'; out.write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8'); log.append({'action':'seal','file':out.name,'sha256':sha(out)})
    method='''# 语言先验盲选方法\n\n规则由人工预先固定：对每个匿名状态与同一 t 的五条 P 描述计算几何绝对差。当前云与预测终点云各半权重；使用 Rg、三个二阶矩、2.5/7 Å 对核、最短间距，及 R00/R05/R15 的质量、局部偏移和壳层核。父子当前位移 RMS 与终点修订 RMS作为较低权重稳定项。所有量按固定容差归一化后取平均，分数越小越匹配。局部区域和全局形状同时参与；current 与 forecast 分开比较，不把终点硬标签或未来状态用于早期。该规则只表达结构相似性，不能推出能量下降、亲和力或终态质量。\n\n回放使用状态卡中的真实 parent 关系：每一步只保留上一排名 ID 的可见子状态，排序后最多五个；无子状态即停止。程序自动执行数值排名，人工决定的是上述公开规则。'''
    (OUT/'method.zh-CN.md').write_text(method,encoding='utf-8')
    log.append({'action':'seal','file':'method.zh-CN.md','sha256':sha(OUT/'method.zh-CN.md')})
    (OUT/'action_log.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()
