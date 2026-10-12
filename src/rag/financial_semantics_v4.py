"""Local financial fact attribution for V4. No labels, answers or eval files."""
from __future__ import annotations
from decimal import Decimal
import re
from financial_knowledge_v4 import METRICS, RECORDS, aliases, canonical as exact_canonical, term_pattern
from exact_cpu_cache import ExactCPUCache

RATIO_METRICS={'权益乘数','产权比率','流动比率','速动比率','市净率','市盈率','销售毛利率','涨跌幅'}
RATIO_METRICS.update(k for k,r in RECORDS.items() if r['unit_kind']=='ratio')
DAY_METRICS={k for k,r in RECORDS.items() if r['unit_kind']=='day'}
UNIT=r'万元/股|元/股|人民币元|百万元|亿元|千元|万元|万股|亿股|元|股|[%％]|倍|天'
AMOUNT=re.compile(r'(?P<value>[-+]?\d(?:[\d,]|\s*,\s*)*(?:\.\d+)?)\s*(?P<unit>'+UNIT+r')?')

class V4Error(ValueError):
    def __init__(self, category, message, **details):
        super().__init__(message)
        self.category, self.details = category, details

def number(value):
    try:
        d=Decimal(re.sub(r'[\s,，]','',str(value)))
    except Exception as exc:
        raise V4Error('format','数值必须只包含数字；单位单独填写') from exc
    if not d.is_finite(): raise V4Error('format','数值不是有限数')
    return d



_METRIC_PATTERNS = tuple((name, re.compile(term_pattern(term), re.I))
                         for name, terms in METRICS.items() for term in terms)
_METRIC_CACHE = ExactCPUCache(4096)

def metric_spans(text):
    return _METRIC_CACHE.get_or_compute(text, lambda: _metric_spans_uncached(text))

def _metric_spans_uncached(text):
    candidates=[]
    for name,pattern in _METRIC_PATTERNS:
        for m in pattern.finditer(text): candidates.append((m.start(),m.end(),name))
    accepted=[]
    for item in sorted(candidates,key=lambda x: (-(x[1]-x[0]),x[0])):
        if not any(item[0]<x[1] and x[0]<item[1] for x in accepted):accepted.append(item)
    return sorted(accepted)


def canonical(metric):
    key=exact_canonical(metric)
    if key in METRICS:return key
    stripped=re.sub(r'(变化率|变动幅度|增减幅度|增长幅度|下降幅度)$','',metric)
    candidate=exact_canonical(stripped)
    return candidate if candidate in METRICS else key

def issuer_mentions(text, target, names=()):
    known=set(names)|{target}
    out=[(m.start(),m.end(),name) for name in known if name and not name.startswith('未识别')
         for m in re.finditer(re.escape(name),text)]
    # Includes '(股票代码：000938.SZ)' and HK comparators missed by the V3 pattern.
    for m in re.finditer(r'([\u4e00-\u9fffA-Za-z·]{2,24})\s*[（(]\s*(?:股票代码\s*[:：]\s*)?\d{4,6}(?:\.(?:SH|SZ|BJ|HK))?\s*[）)]',text,re.I):
        start,end=m.span(1)
        matches=[x for x in out if start<=x[0]<end]
        if not matches:out.append((start,end,target if target in m.group(1) else '__explicit_peer__'))
    return sorted(out)

def facts(text, target, names=(), default_target=False, extra_metrics=()):
    """Attribute each number to the closest preceding metric/year/issuer in its clause.

    This is a conservative verifier, not a semantic gold annotator. It returns
    offsets into original text; values with ambiguous attribution stay unverified.
    """
    result=[]; subject=target if default_target else None
    mentions=issuer_mentions(text,target,names)
    last_end=0
    for sentence in re.finditer(r'[^。；;\n]+',text):
        body=sentence.group(); origin=sentence.start()
        between=[x for x in mentions if last_end<=x[0]<origin]
        if between:subject=max(between,key=lambda x:x[1])[2]
        local_mentions=[(a-origin,b-origin,n) for a,b,n in mentions if origin<=a<sentence.end()]
        spans=metric_spans(body)
        for term in extra_metrics:
            if canonical(term) not in METRICS:
                spans.extend((m.start(),m.end(),term) for m in re.finditer(re.escape(term),body))
        for m in AMOUNT.finditer(body):
            raw=m.group('value').strip()
            try: value=number(raw)
            except ValueError: continue
            if not m.group('unit') and re.fullmatch(r'20\d{2}(?:\d{4})?',re.sub(r'[\s,]','',raw)):continue
            prior=[x for x in spans if x[1]<=m.start() and m.start()-x[1]<=100]
            if not prior:continue
            metric=max(prior,key=lambda x:x[1])[2]
            unit=m.group('unit')
            if unit is None and metric not in RATIO_METRICS|DAY_METRICS and not metric.endswith(('率','倍数')):continue
            if unit is None and not re.search(r'(为|是|达|至|录得|[:：])\s*$',body[max(prior,key=lambda x:x[1])[1]:m.start()]):continue
            unit='%' if unit=='％' else '元' if unit=='人民币元' else unit or 'decimal'
            local_subject=[x for x in local_mentions if x[1]<=m.start()]
            owner=max(local_subject,key=lambda x:x[1])[2] if local_subject else subject
            # A comparator named at this sentence's beginning changes inherited subject.
            if owner!=target:continue
            years=list(re.finditer(r'(?<!\d)(20\d{2})(?=\d{4}(?!\d)|(?!\d))',body[:m.start()]))
            year=years[-1].group(1) if years else None
            # Reject a closer, explicitly named *other* financial metric before the value.
            result.append(dict(metric=metric,value=str(value),unit=unit,year=year,
                start=origin+m.start('value'),end=origin+m.end('value'),
                quote=body.strip(),subject=owner))
        if local_mentions:subject=max(local_mentions,key=lambda x:x[1])[2]
        last_end=sentence.end()
    return result

def supports(field, fact):
    metric=canonical(field['metric'])
    metric_ok=metric==fact['metric']
    if field.get('date') and 'condition_kind' in fact and fact.get('date') != field['date']:
        return False
    change_role = field.get('id') == 'change' or (fact.get('condition_kind') == 'percent_change' and field.get('unit') == '%')
    if not change_role:
        if metric in DAY_METRICS:
            return metric_ok and fact['unit'] in {'天','decimal'} and (not field.get('year') or field['year']==fact['year'])
        ratio=metric in RATIO_METRICS or metric.endswith(('率','倍数'))
        if ratio and fact['unit'] not in {'decimal','%','倍'}:metric_ok=False
        if not ratio and fact['unit'] in {'decimal','%','倍'}:metric_ok=False
    return metric_ok and (not field.get('year') or field['year']==fact['year'])

def verify_fact(field, value, quote, text, target, names=(), default_target=False, parsed_facts=None):
    if not isinstance(quote,str) or not quote or quote not in text:
        raise V4Error('provenance','引文没有逐字出现在声明来源中')
    starts=[m.start() for m in re.finditer(re.escape(quote),text)]
    parsed = parsed_facts if parsed_facts is not None else facts(text,target,names,default_target,(field['metric'],))
    matches=[f for f in parsed
             if number(f['value'])==number(value) and supports(field,f)
             and any(s<=f['start'] and f['end']<=s+len(quote) for s in starts)]
    # Unit can be outside a numeric-only quote, but must be in the same verified source fact.
    if not matches:
        wrong=[f for f in parsed if number(f['value'])==number(value)
               and any(s<=f['start'] and f['end']<=s+len(quote) for s in starts)]
        expected=[dict(metric=f['metric'],year=f['year'],value=f['value'],unit=f['unit'])
                  for f in parsed if supports(field,f)][:4]
        actual=[dict(metric=f['metric'],year=f['year'],value=f['value'],unit=f['unit']) for f in wrong][:4]
        message='该数值的公司、指标或年份与所需字段不一致'
        if actual:
            message+=f"：请求{canonical(field['metric'])}/{field.get('year') or '未指定'}年，所选数值属于"+'、'.join(f"{f['metric']}/{f['year']}年" for f in actual)+'；必须重新选取请求年份和指标的值，不能仅改逗号或单位'
        raise V4Error('semantic',message,field=field['id'],requested_year=field.get('year'),
                      actual_facts=actual,expected_candidates=expected)
    return matches
