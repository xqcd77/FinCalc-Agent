"""Independent table coordinate locator for offline annotation, not runtime RAG.

Reads raw colspan counts and full-table continuation rows. It produces candidates
only; explicit review decisions certify qrels. Never reads rankings or answers.
"""
import collections,json,re
from pathlib import Path

ALIASES={
'OP':['营业利润','经营利润'],'NP':['净利润'],'ANP':['归母净利润','归属于母公司股东的净利润','归属于母公司股东净利润','归属母公司股东的净利润'],
'R':['营业收入'],'TR':['营业总收入'],'COST':['营业成本'],'TA':['资产总计','总资产','资产合计'],
'CA':['流动资产合计','流动资产'],'NCA':['非流动资产合计','非流动资产'],
'TL':['负债合计','总负债'],'CL':['流动负债合计','流动负债'],'NCL':['非流动负债合计','非流动负债'],
'EQ':['所有者权益合计','所有者权益','股东权益总和','股东权益合计','权益合计','净资产'],
'PEQ':['归属于母公司所有者权益合计','归属母公司所有者权益','归母所有者权益','归母权益','归属于母公司股东权益'],
'MIN':['少数股东权益'],'EM':['权益乘数'],'DAR':['资产负债率'],'DER':['产权比率'],'ROE':['净资产收益率','ROE'],
'BPS':['每股净资产','BPS'],'EPS':['每股收益','EPS'],'MC':['总市值','市值'],
'PRICE':['收盘价'],'PCT':['涨跌幅'],'CHANGE':['涨跌金额','涨跌额','涨跌'],
'OCFO':['经营活动产生的现金流量净额','经营活动现金流量净额','经营现金流净额'],
'FCFO':['筹资活动产生的现金流量净额','筹资活动现金流量净额'],'CASH':['货币资金'],'TFA':['交易性金融资产'],
'AR':['应收账款'],'AP':['应付账款'],'INV':['存货'],'FA':['固定资产'],'GM':['销售毛利率','毛利率'],'GP':['毛利'],
'CR':['流动比率'],'QR':['速动比率'],'QUICK':['速动资产'],'DSO':['应收账款周转天数'],'DPO':['应付账款周转天数'],
'FV':['公允价值变动收益'],'INVEST':['投资收益'],'NOI':['营业外收支净额','营业外净收入'],
'NR':['非经常性损益'],'TAX':['所得税费用','所得税'],'PAIDTAX':['支付的所得税','所得税支付','支付所得税'],
'INT':['利息费用','利息支出'],'DA':['折旧与摊销','折旧及摊销','折旧摊销'],'EBIT':['息税前利润','EBIT'],
'TIER1':['一级资本净额'],'LOANS':['贷款总额','贷款总额本金','发放贷款和垫款','贷款及垫款总额'],
'FLOAT':['流通A股','流通股本','流通股数量'],
'CCC':['现金循环周期'],'WC':['营运资本'],'MARGIN':['销售净利率','净利率'],
}
UNIT=re.compile(r'元/股|万元|百万元|亿元|万股|亿股|元|%|百分比|天|倍')
NUM=re.compile(r'^\s*([+-]?\d[\d,，\s]*(?:\.\d+)?)')

def metric_role(text,code):
    if code in {'TA','CA','NCA','TL','CL','NCL','EQ','PEQ','MIN','NP','ANP','OP','FA','GP','AR','AP','INV','OCFO'} and re.search('占.*比|同比|环比|增长率|周转率|周转天数|收益率|净利润率|经营现金流净额 /',text):return False
    hits=[(len(a),c) for c,aa in ALIASES.items() for a in aa if a.lower() in text.lower()]
    if not hits:return False
    best=max(x[0] for x in hits)
    return any(c==code and n==best for n,c in hits)

def headers(rows,headernums):
    result=[]
    for r in rows:
        if r['row'] not in headernums:continue
        vals=r.get('cells',r.get('display_cells',[]));raw=r.get('raw_cells',[]);exp=[]
        for i,v in enumerate(vals):
            m=re.search(r'\\multicolumn\{(\d+)\}',raw[i] if i<len(raw) else '')
            exp += [str(v)]*(int(m[1]) if m else 1)
        result.append(exp)
    return result

def period_match(text,period):
    if len(period)==4:
        if period not in text:return False
        return not bool(re.search(r'半年度|中期|半年|中报|H1|\d{4}中|06[-/]30|6[-/]30|6月30|季度',text))
    y,m,d=period[:4],str(int(period[4:6])),str(int(period[6:]))
    return bool(re.search(r'(?:'+y+r'[-/年]?)?0?'+m+r'[-/月]0?'+d+r'(?:日)?',text) or period in text)

def numeric(cell):
    cell=str(cell).replace('\\','');m=NUM.match(cell)
    if not m:return None
    tail=cell[m.end():].strip()
    if tail and not re.match(r'^(?:元/股|万元|百万元|亿元|万股|亿股|元|股|%|天|倍|[（(*])',tail):return None
    return re.sub(r'[,，\s]','',m[1])

def index(chunks,questions):
    groups=collections.defaultdict(list)
    for c in chunks:
        if c['content_type']=='table':groups[c['table_id']].append(c)
    names={q['company_with_period'] for q in questions}
    codes={}
    for cs in groups.values():
        for c in cs:
            for r in c.get('rows',[]):
                cells=r.get('display_cells',[])
                for i,s in enumerate(cells):
                    if re.search(r'\d{4,6}\.(?:SH|SZ|HK)',str(s)):
                        found=next((n for n in sorted(names,key=len,reverse=True) if any(n in str(x) for x in cells)),None)
                        if found:codes[re.search(r'\d{4,6}\.(?:SH|SZ|HK)',str(s))[0]]=found
    facts=[]
    for tid,cs in groups.items():
        rowmap={r['row']:r for c in cs for r in c.get('rows',[])}
        rows=[rowmap[i] for i in sorted(rowmap)];hnums=cs[0].get('header_row_numbers',[])
        hh=headers(rows,hnums);colhead={i:' '.join(h[i] for h in hh if i<len(h) and h[i]) for i in range(max([len(h) for h in hh] or [0]))}
        cap=str(cs[0].get('table_caption',''));section=' '.join(cs[0].get('section_path',[]));docentity=cs[0]['company']
        if docentity not in names:
            docentity=next((n for n in names if cs[0]['doc_id'].startswith(n+'_')),docentity)
        explicit_entities={n for n in names if n in cap}
        tableentity=next(iter(explicit_entities)) if len(explicit_entities)==1 else docentity
        owner=None;lastrowlabel=None
        ownercols=[i for i,text in colhead.items() if re.search(r'股票简称|公司简称|公司名称|公司（|公司\(|^公司(?: |$)|^股票名称',text)]
        for k,r in enumerate(rows):
            if r['row'] in hnums:continue
            cells=list(map(str,r.get('display_cells',r.get('cells',[]))))
            if not cells:continue
            label=' '.join(cells[:2]);owners=[n for n in sorted(names,key=len,reverse=True) if any(n in s for s in cells[:3])]
            rowcodes=[m[0] for s in cells[:2] for m in re.finditer(r'\d{4,6}\.(?:SH|SZ|HK)',s)]
            if owners:owner=owners[0]
            elif any(x in codes for x in rowcodes):owner=next(codes[x] for x in rowcodes if x in codes)
            elif any(rowcodes):owner='UNKNOWN_PEER'
            elif any(j<len(cells) and cells[j].strip() and numeric(cells[j]) is None for j in ownercols):
                owner='UNKNOWN_PEER'
            elif cells[0] and numeric(cells[0]) is None and not re.fullmatch(r'.*(?:单位|原始披露|人民币).*',cells[0]):
                # A named company outside the question collection is still a peer.
                if any('股票简称' in x or '公司' in x for x in hh[0] if hh):
                    if not any(a in cells[0] for aa in ALIASES.values() for a in aa):owner='UNKNOWN_PEER'
            if not cells[0] or re.search(r'^\[.*?\]\s*单位|^单位',cells[0]):
                rolelabel=lastrowlabel or label
            else:
                rolelabel=label;lastrowlabel=label
            is_units=any('单位' in x for x in cells[:1]) and not any(numeric(x) for x in cells[1:])
            if is_units:continue
            for col,cell in enumerate(cells):
                val=numeric(cell)
                if val is None or re.fullmatch(r'\d{4,6}\.(SH|SZ|HK)',cell) or re.fullmatch(r'(19|20)\d{2}(?:[-/]\d{1,2}[-/]\d{1,2}|年度|年报|年)?',cell):continue
                ch=colhead.get(col,'')
                # Metric rows override global headings; metric columns override captions.
                mettext=rolelabel if any(a in rolelabel for aa in ALIASES.values() for a in aa) else ch
                if not any(a.lower() in mettext.lower() for aa in ALIASES.values() for a in aa):mettext=cap+' '+section+' '+ch
                pertext=ch+' '+rolelabel+' '+' '.join(s for s in cells[:3] if re.fullmatch(r'(?:19|20)\d{2}(?:\s*年报|\s*年度|\s*年末|\s*年)?',s))
                un=UNIT.findall(cell)
                unit=un[-1] if un else None
                unitquote=[]
                if not unit:
                    for rr in rows[max(0,k-1):k+3]:
                        cc=rr.get('display_cells',rr.get('cells',[]))
                        if col<len(cc) and re.search(r'单位',str(cc[0])):
                            us=UNIT.findall(str(cc[col]));
                            if us:unit=us[-1];unitquote.append(rr['row']);break
                if not unit:
                    unitcols=[i for i,t in colhead.items() if t.strip()=='单位']
                    for j in unitcols:
                        if j<len(cells) and UNIT.fullmatch(cells[j].strip()):unit=cells[j].strip();break
                if not unit:
                    for text in (ch,rolelabel,cap):
                        us=list(dict.fromkeys(UNIT.findall(text)))
                        if len(us)==1:unit=us[0];break
                if not unit:
                    common=list(dict.fromkeys(cs[0].get('unit_mentions',[])))
                    if len(common)==1:unit=common[0]
                if any(metric_role(mettext,x) for x in ['EM','CR','QR','DAR','DER']):
                    unit='%' if '%' in (cell+' '+ch+' '+rolelabel) else 'decimal'
                cid=next(c['chunk_id'] for c in cs if any(z['row']==r['row'] for z in c['rows']))
                colowners=[n for n in sorted(names,key=len,reverse=True) if n in ch]
                factowner=colowners[0] if colowners else owner or tableentity
                facts.append(dict(table_id=tid,chunk_id=cid,entity=factowner,metric_text=mettext,
                    period_text=pertext,value=val,unit=unit,row=r['row'],column=col,cell=cell,
                    header_text=ch,row_label=rolelabel,unit_rows=unitquote,
                    source_path=cs[0]['source_path'],source_lines=[cs[0]['source_start_line'],cs[0]['source_end_line']]))
    return facts
