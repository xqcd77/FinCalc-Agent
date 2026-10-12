"""Question-only facts and planning guidance; no report values or eval labels."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import re

from financial_semantics_v4 import facts, metric_spans, canonical

HYPOTHESIS = re.compile(r'假如|如果|假设|倘若|若')
DIRECTION = r'提高|下降|增长|增加|减少|降低|上升|上涨|下跌|调高|上调|调低|下调'
FINAL_LEVEL = r'(?:调整为|变为|改为|调至|增至|降至|到|至)'


def resolve_question_entity(question, target=None):
    """Use the supplied corpus issuer; parse a leading stock-code issuer only as fallback."""
    if target and not target.startswith('未识别'):
        return target
    cleaned=re.sub(r'^(?:假如|如果|假设|若|倘若)\s*','',question.strip())
    coded=re.match(r'([\u4e00-\u9fffA-Za-z·\-]{2,32}?)\s*[（(]\s*(?:股票代码\s*[:：]\s*)?\d{4,6}(?:\.(?:SH|SZ|BJ|HK))?\s*[）)]',cleaned,re.I)
    if coded:
        return coded.group(1)
    from select_context_v2 import target_entity
    return target_entity('未识别机构',question)


def question_metric_view(text):
    """Mask only a nonnumeric calculation-basis parenthesis after a main metric.

    Same-length whitespace keeps numeric/quote offsets tied to original text.
    Ordinary metric parentheses, dates, values and separate clauses remain visible.
    """
    spans=metric_spans(text)
    characters=list(text)
    for match in re.finditer(r'[（(]([^()（）]*)[）)]',text):
        content=match.group(1)
        previous=[s for s in spans if s[1]<=match.start() and not text[s[1]:match.start()].strip()]
        # A calculation basis is not an adjusted accounting scope. Never mask
        # e.g. 扣非口径/调整后口径/归母口径 or numeric parenthetical facts.
        basis=re.fullmatch(r'(?:按|以)?\s*(?:总股本|总股数|流通股本|流通股数)\s*(?:算|计算|计算口径)',content.strip())
        if previous and basis:
            characters[match.start():match.end()]=' '*(match.end()-match.start())
    return ''.join(characters)


def resolved_change(question, start, end, metric):
    """Recognise a disclosed final level of the SAME metric in the change clause."""
    suffix=re.sub(r'^\s*[%％]\s*','',question[end:])
    suffix=re.sub(r'^[，,]\s*(?=由|从)','',suffix)
    clause=re.split(r'(?<!\d)[,，]|[,，](?!\d)|[。；;]',suffix,1)[0]
    final=re.search(FINAL_LEVEL+r'\s*[-+]?\d',clause)
    if not final:
        # "调高3%为新值" and "调高3%，由旧值为新值" are explicit
        # results. "提高3%，原值为..." is not a disclosed final level.
        final=re.search(r'^\s*为\s*[-+]?\d|^\s*(?:由|从)\s*[-+]?\d[\d,.]*\s*(?:亿元|百万元|万元|千元|元|[%％]|倍|天)?\s*为\s*[-+]?\d',clause)
    if not final:return False
    between=metric_spans(question_metric_view(clause[:final.start()]))
    return not between or all(s[2]==metric for s in between)


def question_body(question):
    # Precision instructions are metadata, not hypothetical financial operands.
    marker = re.search(r'[（(]\s*(?:注[：:]|其他条件|请|输出)', question)
    return question[:marker.start()] if marker else question


def date_mentions(text):
    out = []
    pattern = r'(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)|(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日'
    for match in re.finditer(pattern, text):
        parts = match.groups()[:3] if match.group(1) else match.groups()[3:]
        try:
            day = datetime(*map(int, parts)).strftime('%Y-%m-%d')
        except ValueError:
            continue
        out.append((match.start(), match.end(), day))
    return out


def question_facts(question, target, extra_metrics=()):
    body = question_body(question)
    target=resolve_question_entity(question,target)
    view=question_metric_view(body)
    bare_changes=list(re.finditer(r'涨跌\s*(?:为|是|[:：=])\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(元/股|人民币元|元|[%％])',view))
    parsed = facts(view, target, default_target=True, extra_metrics=(*extra_metrics,*(['涨跌'] if bare_changes else [])))
    sentences=list(re.finditer(r'[^。；;\n]+',body))
    dates = date_mentions(body)
    years = set(re.findall(r'(?<!\d)(20\d{2})(?!\d)', body))
    for fact in parsed:
        bare=next((m for m in bare_changes if m.start(1)==fact['start']),None)
        if bare and fact['metric']=='涨跌':
            fact['metric']='涨跌幅' if bare.group(2) in {'%','％'} else '涨跌金额'
            fact['metric_resolution']='bare_change_disambiguated_by_explicit_unit'
        # A masked parser view is not an evidence quote. Restore the exact source.
        sentence=next(s for s in sentences if s.start()<=fact['start']<s.end())
        fact['quote']=sentence.group().strip()
        if view != body:fact['metric_resolution']='question_calculation_basis_parenthesis_ignored'
        prefix = body[:fact['start']]
        direction = re.search(r'(?:' + DIRECTION + r')\s*$', prefix)
        fact['condition_kind'] = ('percent_change' if fact['unit'] == '%' else 'amount_change') if direction else 'level'
        prior = [d for d in dates if d[1] <= fact['start'] and fact['start'] - d[1] < 100]
        if prior and canonical(fact['metric']) in {'股价','涨跌金额','涨跌幅','成交量','总市值','流通股本','市净率','市盈率'}:
            fact['date'] = prior[-1][2]
            fact['year'] = fact['date'][:4]
        # A year appearing only in the requested annual result can scope a
        # preceding hypothesis. Multiple years or transaction dates stay strict.
        if fact['year'] is None and len(years) == 1 and not dates:
            fact['year'] = next(iter(years))
            fact['year_resolution'] = 'one_explicit_annual_year_in_question'
            fact['quote'] = body
    return parsed


def question_conditions(question, target):
    body = question_body(question)
    hypothesis = HYPOTHESIS.search(body)
    if not hypothesis:
        return []
    result = []
    for fact in question_facts(body, target):
        if fact['start'] < hypothesis.end():
            continue
        prefix = body[:fact['start']]
        kind = fact['condition_kind']
        if kind == 'level' and not re.search(r'(?:为|是|达|至|录得|[:：=])\s*$', prefix):
            continue
        if kind == 'percent_change':
            if resolved_change(body,fact['start'],fact['end'],fact['metric']):continue
        result.append(fact)
    return result


def missing_question_condition(question):
    body = question_body(question)
    match = HYPOTHESIS.search(body)
    if not match:
        return None
    clause = re.split(r'(?<!\d)[,，]|[,，](?!\d)|[。；;]', body[match.end():], 1)[0]
    spans = metric_spans(clause)
    if not spans:
        return None
    metric = spans[-1][2]
    suffix = clause[spans[-1][1]:].strip()
    if re.fullmatch(r'[（()）\s]*', suffix):
        target = resolve_question_entity(question)
        wanted = canonical(metric)
        # The phrase may refer to an already quantified value elsewhere in the
        # question (including a specified closing balance used by a custom
        # calculation). Do not turn a recoverable value into a refusal.
        if any(canonical(f.get('metric','')) == wanted for f in question_facts(body, target)):
            return None
        # A metric named as the requested output is not a missing assumption.
        actions = list(re.finditer(r'(?:计算|求取|求出|求得|推算|测算|算出)', body))
        requested = body[actions[-1].start():] if actions else body
        if any(canonical(name) == wanted for _, _, name in metric_spans(requested)):
            return None
        return 'missing_question_condition:假设的' + metric + '未提供数值；原题片段=' + clause
    if re.search(r'\d+在\d+', suffix):
        return 'missing_question_condition:假设的' + metric + '数字含污染字符，不能恢复；原题片段=' + clause
    return None


def condition_ledger(question, target):
    entries = []
    for fact in question_conditions(question, target):
        prefix = question[:fact['start']]
        if not re.search(r'(?:为|是|至|到|[:：=]|' + DIRECTION + r')\s*$', prefix):
            continue
        entry = {k: fact[k] for k in ('metric', 'year', 'value', 'unit', 'condition_kind')}
        if fact.get('date'):
            entry['date'] = fact['date']
        entries.append(entry)
    return json.dumps(entries[:5], ensure_ascii=False, separators=(',', ':')) if entries else ''


def task_guidance(question):
    notes = []
    if re.search(r'涨跌|收盘价|股价|市净率|市盈率|换手率', question):
        notes.append('行情字段写date；同一年不同交易日不是同一字段。题目只给前收盘价/涨跌额时，后续价格放在expression推导，不能把计算出的价格写成source=question。')
        dates = date_mentions(question)
        interval = re.search(r'(20\d{6})到(20\d{6})', question)
        if interval:
            begin, end = (datetime.strptime(x, '%Y%m%d') for x in interval.groups())
            if 0 <= (end - begin).days <= 14:
                weekdays = [(begin + timedelta(days=i)).strftime('%Y-%m-%d') for i in range((end-begin).days+1) if (begin+timedelta(days=i)).weekday()<5]
                notes.append('区间工作日候选=' + ','.join(weekdays) + '；交易日须有报告行情表证据；禁止扩展到区间外或凭空添加周末。')
        notes.append('最高/最低用max/min；次高/次低用nth_largest(2,...)/nth_smallest(2,...)。PE=价格/EPS，PB=价格/BPS；股价以元/股表示，round_cent仅用于股价。')
    if '周转天数' in question:
        notes.append('周转天数=报告年度天数×平均余额/收入或成本；360/365以已显示方法为准，不能用round_cent给天数取整。百分数单位%由工具除100，expression不要再次除100。绝对增减金额保留原单位，用基准余额±变化额。')
    if re.search(r'假如|如果|假设', question):
        notes.append('假设指标与直接公式字段不同，先根据会计恒等关系展开依赖；不得把资产金额填为负债，不得为通过校验添加不参与计算的字段。确无唯一关系时说明缺少哪项条件。')
    if re.search(r'市销率|PS',question,re.I):
        notes.append('PS=总市值/营业收入；若仅有股价，须再取总股本计算市值。股价不能直接除总收入。')
    return '\n'.join(notes)
