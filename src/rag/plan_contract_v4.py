"""Lossless plan repairs backed by the original question, never evaluation labels.

This boundary layer repairs representation only. It does not supply report
numbers, choose a custom formula, overwrite conflicting values, or resolve
ambiguous financial facts. Every repair is returned as an audit event.
"""
from __future__ import annotations

from copy import deepcopy
import ast
from decimal import Decimal, InvalidOperation
import json
import re

from financial_semantics_v4 import V4Error, canonical, facts, supports
from question_conditions_v4 import question_facts, question_conditions


def _normalise_question_adjustment_metric(p,question,target,events):
    """An explicit hypothetical final level can carry an adjustment-state label.

    Scope is tied to one verified question fact; never apply this to report
    metrics or substantive accounting labels such as 扣非/经调整 EBITDA.
    """
    from financial_semantics_v4 import aliases as metric_aliases
    conditions=question_conditions(question,target)
    for field in p.get('fields',[]):
        original=field['metric']
        if field.get('source')!='question' or not original.startswith('调整后'):continue
        base=canonical(re.sub(r'^调整后(?:的)?','',original))
        if base==original or not field.get('quote'):continue
        try:
            value,_=_number_payload(field.get('value'),field.get('unit'),field['id'])
            quote,_=restore_quote(field['quote'],question,field['id'])
        except V4Error:continue
        starts=[m.start() for m in re.finditer(re.escape(quote),question)]
        matches=[f for f in conditions if f['condition_kind']=='level' and f['metric']==base
                 and Decimal(f['value'])==Decimal(value) and (not field.get('year') or f['year']==field['year'])
                 and (not field.get('date') or f.get('date')==field['date'])
                 and any(s<=f['start'] and f['end']<=s+len(quote) for s in starts)]
        if _unique(matches):
            field['metric']=base;field['aliases']=metric_aliases(base)[:4]
            events.append(dict(operation='normalise_question_adjustment_metric',field=field['id'],
                               original_metric=original,metric=base,basis='unique_explicit_hypothetical_level'))

FIELD_ID = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")
UNIT_PATTERN = r"万元/股|元/股|人民币元|百万元|亿元|千元|万元|万股|亿股|元|股|[%％]|倍|天"
NUMERIC = r"[-+]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
VALUE = re.compile(r"(?P<number>" + NUMERIC + r")\s*(?P<unit>" + UNIT_PATTERN + r")?\Z")
CHANGE = re.compile(r"(?:提高|下降|增长|增加|减少|降低|上升|上涨|下跌|调高|上调|调低|下调)\s*(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)\s*[%％]")
PUNCTUATION = str.maketrans({"，": ",", "；": ";", "：": ":", "（": "(", "）": ")", "％": "%", "？": "?", "！": "!"})


def source_unit(unit):
    return {"％": "%", "人民币元": "元"}.get(unit, unit)


def _quote_key(text):
    chars, offsets = [], []
    for i, char in enumerate(text):
        if char.isspace():
            continue
        chars.append(char.translate(PUNCTUATION))
        offsets.append(i)
    return "".join(chars), offsets


def restore_quote(quote, source, field_id):
    """Restore one original span after space/punctuation changes; keep digits/signs."""
    if not isinstance(quote, str) or not quote:
        raise V4Error("schema", f"字段{field_id}缺少原文quote", code="missing_question_quote", field=field_id, path="quote")
    if quote in source:
        return quote, False
    needle, _ = _quote_key(quote)
    haystack, offsets = _quote_key(source)
    starts = [m.start() for m in re.finditer("(?=" + re.escape(needle) + ")", haystack)] if needle else []
    if len(starts) != 1:
        raise V4Error("schema", f"字段{field_id}的quote无法唯一对应原题；逐字复制含数值的原文片段",
                      code="question_quote_not_unique", field=field_id, path="quote", candidate_count=len(starts))
    start = starts[0]
    return source[offsets[start]:offsets[start + len(needle) - 1] + 1], True


def question_candidates(field, question, target):
    """Unique candidates by issuer/metric/year AND absolute-value/change role."""
    changes = [m.span("value") for m in CHANGE.finditer(question)]
    candidates = []
    for fact in question_facts(question, target, extra_metrics=(field.get("metric", ""),)):
        if not supports(field, fact):
            continue
        is_change = any(a <= fact["start"] < b for a, b in changes)
        change_role = field.get('id') == 'change' or (field.get('unit') == '%' and fact.get('condition_kind') == 'percent_change')
        if is_change != change_role:
            continue
        # Recover only explicit question conditions, not arbitrary numbers in the task.
        if not is_change and not re.search(r"(?:为|是|达|至|录得|[:：=])\s*$", question[:fact["start"]]):
            continue
        candidates.append(fact)
    return candidates


def _unique(candidates):
    signatures = {(f["metric"], f["year"], f.get('date'), f["value"], f["unit"]) for f in candidates}
    return candidates[0] if len(signatures) == 1 else None


def _number_payload(value, unit, ident):
    if isinstance(unit,str) and not unit.strip():unit=None
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise V4Error("schema", f"字段{ident}.value必须是数值字符串；单位单独填unit",
                      code="question_value_not_numeric", field=ident, path="value")
    match = VALUE.fullmatch(str(value).strip())
    if not match:
        raise V4Error("schema", f"字段{ident}.value无法解析：只填原始数字，不填计算式或说明",
                      code="question_value_not_numeric", field=ident, path="value")
    suffix = source_unit(match.group("unit"))
    if suffix and unit is not None and source_unit(unit) != suffix:
        raise V4Error("schema", f"字段{ident}的value附带单位与unit矛盾，不能自动换算",
                      code="question_unit_conflict", field=ident, value_unit=suffix, declared_unit=unit)
    try:
        parsed = Decimal(re.sub(r"[,，]", "", match.group("number")))
    except InvalidOperation as exc:
        raise V4Error("schema", f"字段{ident}.value不是有限数值", code="question_value_not_numeric", field=ident) from exc
    if not parsed.is_finite():
        raise V4Error("schema", f"字段{ident}.value不是有限数值", code="question_value_not_numeric", field=ident)
    return str(parsed), source_unit(unit) if unit is not None else suffix


def _normalise_ids(plan, events):
    fields = plan.get("fields")
    if not isinstance(fields, list):
        return
    by_id, unique = {}, []
    for field in fields:
        if not isinstance(field, dict) or not isinstance(field.get("id"), str):
            raise V4Error("schema", "字段必须是含字符串id的对象", code="malformed_field", path="fields")
        old = field["id"]
        if not old:
            raise V4Error("schema", "字段id不能留空", code="missing_field_id", path="fields[].id")
        if old in by_id:
            if field != by_id[old]:
                raise V4Error("schema", f"字段ID {old}重复且内容不同，无法确定表达式指向哪个字段；为不同角色使用不同ID",
                              code="ambiguous_duplicate_field_id", field=old, conflicting_fields=[by_id[old], field])
            events.append(dict(operation="deduplicate_identical_field", field=old))
            continue
        by_id[old] = field
        unique.append(field)
    mapping, used = {}, {f["id"] for f in unique if FIELD_ID.fullmatch(f["id"])}
    for i, field in enumerate(unique, 1):
        old = field["id"]
        if FIELD_ID.fullmatch(old):
            continue
        # Do not reinterpret an arithmetic expression or numeric literal as an
        # identifier. Table-like colons and identifier spelling are reversible;
        # operators, decimals, quotes and whitespace need a fresh model proposal.
        if not re.fullmatch(r"[A-Za-z0-9_\u4e00-\u9fff:]+", old) or old.isdigit():
            raise V4Error('schema', f'字段ID {old}含数值/运算符或非标识字符，不能自动改名；使用唯一变量名并同步expression',
                          code='unsafe_field_id_rename', field=old, path='fields[].id')
        new = re.sub(r"[^a-z0-9_]+", "_", old.lower()).strip("_") or f"field_{i}"
        if not new[0].isalpha():
            new = "f_" + new
        new = new[:32]
        if new in used:
            raise V4Error("schema", f"字段ID规范化后冲突：{old} → {new}；请使用唯一ID",
                          code="field_id_collision", field=old, normalized_id=new)
        used.add(new)
        mapping[old] = new
        field["id"] = new
    if mapping:
        expression = plan.get("expression")
        if isinstance(expression, str):
            # One simultaneous exact-token substitution preserves the declared
            # variable relationship, even for table-like IDs containing colons.
            tokens = "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
            plan["expression"] = re.sub(r"(?<![A-Za-z0-9_])(?:" + tokens + r")(?![A-Za-z0-9_])",
                                         lambda m: mapping[m.group()], expression)
        events.append(dict(operation="rename_field_ids", mapping=mapping, expression_rewritten=isinstance(expression, str)))
    plan["fields"] = unique


def _normalise_registered_roles(p,question,target,roles,events,registered_expression):
    """Rename a complete, unique metric/year/change-role bijection; never swap values."""
    fields=p.get('fields',[])
    required={r['id'] for r in roles}
    if not roles or {f['id'] for f in fields}==required:return
    if len(fields)!=len(roles) or any(not r.get('metric') for r in roles):return
    mapping={};used=set()
    for role in roles:
        candidates=[]
        for f in fields:
            if canonical(f['metric'])!=canonical(role['metric']):continue
            year=f.get('year')
            matched=[]
            if f.get('source')=='question':
                matched=question_candidates({**f,'id':'change' if role['id']=='change' else f['id'],
                                             **({'unit':'%'} if role['id']=='change' else {})},question,target)
                try:
                    raw,_=_number_payload(f.get('value'),f.get('unit'),f['id'])
                    matched=[v for v in matched if Decimal(v['value'])==Decimal(raw)]
                except V4Error:
                    matched=[]
                one=_unique(matched)
                if year is None and one:year=one['year']
                try:
                    raw,_=_number_payload(f.get('value'),f.get('unit'),f['id'])
                    shape=[v for v in question_facts(question,target) if v['metric']==canonical(f['metric']) and Decimal(v['value'])==Decimal(raw)
                           and (not f.get('year') or v['year']==f['year']) and (not f.get('date') or v.get('date')==f['date'])]
                except V4Error:shape=[]
                kinds={v.get('condition_kind') for v in shape}
                if role['id']=='change' and kinds!={'percent_change'}:continue
                if role['id']!='change' and kinds in ({'percent_change'},{'amount_change'}):continue
            elif role['id']=='change':continue
            if role.get('year') is not None and year!=role['year']:continue
            if role.get('date') and f.get('date') and f['date']!=role['date']:continue
            candidates.append(f)
        if len(candidates)!=1 or candidates[0]['id'] in used:return
        old=candidates[0]['id'];used.add(old);mapping[old]=role['id']
    if len(used)!=len(fields):return
    expression=p.get('expression')
    rewritten=expression
    if isinstance(expression,str) and expression.strip():
        # Registered formulas must not silently replace a different supplied
        # expression. Only an exact AST-equivalent role rename is reversible.
        if not registered_expression:return
        pattern='|'.join(re.escape(k) for k in sorted(mapping,key=len,reverse=True))
        rewritten=re.sub(r'(?<![A-Za-z0-9_])(?:'+pattern+r')(?![A-Za-z0-9_])',lambda m:mapping[m.group()],expression)
        try:
            if ast.dump(ast.parse(rewritten,mode='eval'))!=ast.dump(ast.parse(registered_expression,mode='eval')):return
        except SyntaxError:return
    for f in fields:f['id']=mapping[f['id']]
    if rewritten is not expression:p['expression']=rewritten
    events.append(dict(operation='normalise_registered_role_ids',mapping=mapping,basis='unique_metric_year_and_change_role_bijection'))


def repair_contract(plan, question, target, registered_roles=(), registered_expression=None):
    """Return copied proposal + reversible repairs; all conflicts still fail closed."""
    p, events = deepcopy(plan), []
    try:
        _normalise_ids(p, events)
        _normalise_expression_field_access(p,events)
        _normalise_question_adjustment_metric(p,question,target,events)
        _normalise_registered_roles(p,question,target,registered_roles,events,registered_expression)
        return _repair_question_fields(p, question, target, registered_roles, events)
    except V4Error as exc:
        if events:
            exc.details['contract_normalisations'] = events
        raise


def _repair_question_fields(p, question, target, registered_roles, events):
    roles = {r["id"]: r for r in registered_roles}
    for field in p.get("fields", []):
        ident = field["id"]
        candidates = question_candidates(field, question, target)
        role = roles.get(ident)
        registered = role and role.get("metric") and canonical(role["metric"]) == canonical(field.get("metric", "")) and role.get("year") == field.get("year")
        fact = _unique(candidates)
        if field.get("source") == "report" and registered and fact:
            field["source"] = "question"
            events.append(dict(operation="use_explicit_question_override", field=ident))
        if field.get("source") != "question":
            continue
        before = {k: field.get(k) for k in ("value", "unit", "quote", "year")}
        missing_value = field.get("value") is None or field.get("value") == ""
        # A known formula's baseline field with no question payload belongs to
        # report retrieval. A percentage adjustment is never that baseline amount.
        if missing_value and not fact and not candidates and registered and not any(field.get(k) for k in ("unit", "quote")):
            field["source"] = "report"
            events.append(dict(operation="defer_registered_baseline_to_report", field=ident, metric=field["metric"], year=field.get("year")))
            continue
        if missing_value:
            if not fact:
                raise V4Error("schema", f"字段{ident}缺value且原题没有唯一匹配的公司/指标/年份数值；报告值应source=report",
                              code="missing_question_value", field=ident, path="value", candidate_count=len(candidates))
            field["value"] = fact["value"]
        numeric, unit = _number_payload(field["value"], field.get("unit"), ident)
        field["value"] = numeric
        matching = [f for f in candidates if Decimal(f["value"]) == Decimal(numeric)]
        if field.get("quote"):
            quote, restored = restore_quote(field["quote"], question, ident)
            field["quote"] = quote
            if restored:
                events.append(dict(operation="restore_original_quote", field=ident, quote=quote))
            starts = [m.start() for m in re.finditer(re.escape(quote), question)]
            matching = [f for f in matching if any(s <= f["start"] and f["end"] <= s + len(quote) for s in starts)]
        fact = _unique(matching)
        if unit is None:
            if not fact:
                raise V4Error("schema", f"字段{ident}缺unit且原题单位无法唯一核实，不能默认金额单位",
                              code="missing_question_unit", field=ident, path="unit", candidate_count=len(matching))
            unit = fact["unit"]
        field["unit"] = unit
        if not field.get("quote"):
            if not fact:
                raise V4Error("schema", f"字段{ident}缺quote且无法唯一核实其公司/指标/年份数值",
                              code="missing_question_quote", field=ident, path="quote", candidate_count=len(matching))
            field["quote"] = fact["quote"]
        if field.get("year") is None and fact and fact.get("year"):
            field["year"] = fact["year"]
        if field.get('date') is None and fact and fact.get('date'):
            field['date']=fact['date']
            events.append(dict(operation='recover_explicit_question_date',field=ident,date=fact['date']))
        after = {k: field.get(k) for k in before}
        if before != after:
            events.append(dict(operation="recover_question_payload", field=ident, before=before, after=after))
    for field in p.get('fields',[]):
        role=roles.get(field['id'])
        if role and role.get('date') and field.get('date') is None and field.get('year') in {None,role.get('year')} and canonical(field['metric'])==canonical(role['metric']):
            field['date']=role['date'];field['year']=role['year']
            events.append(dict(operation='recover_registered_role_date',field=field['id'],date=role['date'],basis='one_explicit_question_date'))
    return p, events


def _normalise_expression_field_access(plan,events):
    """Normalize only declared fields.name / fields[integer].value accesses.

    No general attribute evaluation, indexing, calls or numeric values are used.
    All other AST nodes retain the ordinary expression validator's restrictions.
    """
    expression=plan.get('expression')
    if not isinstance(expression,str) or not expression.strip():return
    try:tree=ast.parse(expression,mode='eval')
    except SyntaxError:return
    fields=plan.get('fields',[]);ids={f['id'] for f in fields};rewrites=[]
    class Access(ast.NodeTransformer):
        def visit_Attribute(self,node):
            ident=None
            if isinstance(node.value,ast.Name) and node.value.id=='fields' and node.attr in ids:ident=node.attr
            elif node.attr=='value' and isinstance(node.value,ast.Subscript) and isinstance(node.value.value,ast.Name) and node.value.value.id=='fields':
                index=node.value.slice
                if isinstance(index,ast.Constant) and type(index.value) is int and 0<=index.value<len(fields):ident=fields[index.value]['id']
            if ident:
                rewrites.append(dict(original=ast.unparse(node),field_id=ident))
                return ast.copy_location(ast.Name(id=ident,ctx=ast.Load()),node)
            return self.generic_visit(node)
    tree=Access().visit(tree)
    if rewrites:
        plan['expression']=ast.unparse(tree)
        events.append(dict(operation='normalise_expression_field_access',mapping=rewrites,basis='explicit_declared_field_access'))


def planning_issues(proposal,question,target=None,limit=6):
    """Collect multiple independent problems without inventing a corrected plan."""
    from field_plan_v4 import FORMULAS
    from financial_semantics_v4 import verify_fact
    from question_conditions_v4 import resolve_question_entity
    target=resolve_question_entity(question,target);issues=[]
    def add(code,message,field=None):
        item=dict(code=code,message=message[:220])
        if field:item['field']=field
        if item not in issues:issues.append(item)
    if not isinstance(proposal,dict):return [dict(code='plan_not_object',message='完整JSON对象未解析成功')]
    name=proposal.get('formula_id');fields=proposal.get('fields',[])
    if not isinstance(name,str):add('invalid_formula_id','formula_id必须是字符串');name=None
    if name not in FORMULAS and name not in {'custom','unsupported'}:add('unknown_formula_id','使用提示ID或custom；业务名不能当公式ID')
    if not isinstance(fields,list):return [dict(code='fields_not_list',message='fields必须是对象列表')]
    if name not in FORMULAS and name!='unsupported' and not proposal.get('expression'):add('missing_expression','自定义业务计划缺少expression')
    if name in FORMULAS and {f.get('id') for f in fields if isinstance(f,dict) and isinstance(f.get('id'),str)}!=set(FORMULAS[name][0]):
        add('registered_roles','当前字段数/角色不满足注册公式，不混用两套依赖')
    for f in fields[:12]:
        if not isinstance(f,dict):add('malformed_field','每个字段必须是对象');continue
        ident=str(f.get('id',''))
        if not isinstance(f.get('source'),str) or f['source'] not in {'question','report'}:add('invalid_source','source只能question/report；推导值写入expression',ident);continue
        if f.get('source')!='question':continue
        if not f.get('metric'):add('missing_metric','缺少指标名',ident);continue
        try:
            value,unit=_number_payload(f.get('value'),f.get('unit'),ident)
            quote,_=restore_quote(f.get('quote'),question,ident)
            fact_list=question_facts(question,target,(f['metric'],))
            matches=verify_fact(f,value,quote,question,target,default_target=True,parsed_facts=fact_list)
            if unit and not any(c['unit']==unit or (unit=='元/股' and c['unit']=='元') for c in matches):
                add('unit_mismatch','保留该原题数值的真实单位，%不要改写为倍数或金额',ident)
        except (ValueError,TypeError,KeyError) as exc:add(getattr(exc,'details',{}).get('code','question_fact_mismatch'),str(exc),ident)
    expression=proposal.get('expression')
    if isinstance(expression,str):
        try:
            tree=ast.parse(expression,mode='eval')
            if any(isinstance(n,(ast.Attribute,ast.Subscript)) for n in ast.walk(tree)):add('expression_access','expression使用声明的裸ID，不访问fields数组或属性')
            if any(isinstance(n,ast.Constant) and n.value==100 for n in ast.walk(tree)):add('duplicate_percent_conversion','unit=%已由工具除100，expression不得再次除100')
            used={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}-{'round_cent','min','max','nth_largest','nth_smallest'}
            declared={f.get('id') for f in fields if isinstance(f,dict) and isinstance(f.get('id'),str)}
            if used!=declared:add('variable_contract','未使用字段='+','.join(sorted(str(x) for x in declared-used))+'；未声明变量='+','.join(sorted(used-declared)))
            if 'round_cent' in used or any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='round_cent' for n in ast.walk(tree)):
                if '周转天数' in question:add('rounding_scope','round_cent只用于元/股股价，不用于周转天数')
        except SyntaxError:add('expression_syntax','expression不是完整合法的受限算术表达式')
    return issues[:limit]


def repair_actions(error, question=''):
    """Error-specific instructions; never supply report numbers or final answers."""
    details=getattr(error,'details',{});code=details.get('code');message=str(error)
    actions=[]
    if code=='unknown_formula_id':actions.append('本题注册公式用提示ID；其他任务用custom和完整expression，不自造业务/日期ID。')
    if code=='registered_role_ids':actions.append('字段无法唯一匹配注册角色；按提示的指标/年份重建依赖，或custom写明另一合法公式，不只机械改ID。')
    if code in {'blueprint_dependency_mismatch','blueprint_extra_fields','blueprint_expression_mismatch'}:
        actions.append('按本题依赖骨架重新建fields和expression；区分基准余额与变化，保持年度和来源，不添加未参与的参考值。')
    if details.get('actual_facts'):
        actions.append('所选值属于错误指标/年份；不得只改标签来通过。若题目假设与目标不同，按会计关系展开依赖，所需其他值source=report。')
    if code=='scenario_condition_omitted':
        actions.append('逐项纳入缺失假设：绝对增减额是独立question字段，原余额是report，表达式余额±增减额；百分比由工具换算一次；已给最终值直接使用。')
    if code in {'question_quote_not_unique','invalid_question_quote','missing_question_quote','missing_question_value','question_value_not_numeric'} or 'value absent from quote' in message:
        actions.append('question仅填原题直接数值和原文quote；推导结果不得冒充题目值。把原余额/价格列为report，把题给变化列为question，在expression内计算。')
    if details.get('unused_ids') or details.get('undeclared_ids'):
        actions.append('只声明实际参与表达式的基础字段。无用参考值可移除；关键假设必须参与，不能为消除报错删掉它；同步变量名，不混用fields索引。')
    if 'custom公式与目标' in message:
        actions.append('当前运算关系不是目标金融公式；按本题公式骨架重建分子/分母及期间，禁止仅改formula_id重复旧表达式。')
    if code in {'market_date_missing','market_date_outside_interval','non_trading_weekend','next_day_dependency_missing','ranking_operation_missing'}:
        actions.append('重新列出问题所需的真实交易日：下一日价格需下一日行情或涨跌证据；不加周末/区间外日期；最高最低次高次低用对应函数。')
    if '维度' in message or 'round_cent' in message or 'unattributed numeric literal' in message:
        actions.append('核对字段单位与公式：%工具自动除100，金额单位工具统一；round_cent只用于股价，天数/比率最终统一舍入。')
        if re.search(r'市销率|\bPS\b',question,re.I):actions.append('PS=总市值/营业收入，或股价×总股本/营业收入，不能股价/总收入。')
    return '修复动作：'+''.join(actions) if actions else '修复动作：针对上述定位重建完整计划，保持原题数值、公司、年份和来源。'


def repair_feedback(proposal, error, question='', target=None):
    """Compact full plan structure + actionable, field-specific contract diagnostics."""
    if not isinstance(proposal, dict):
        return "具体错误：" + str(error)
    fields=proposal.get('fields',[])
    if not isinstance(fields,list):fields=[]
    compact = {k: v for k, v in proposal.items() if k in {"formula_id", "answer_unit", "expression", "reason"}}
    compact["fields"] = [{k: v for k, v in f.items() if k in {"id", "metric", "year", "date", "source", "value", "unit"}}
                         if isinstance(f, dict) else f for f in fields]
    issues = []
    if proposal.get("formula_id") == "custom" and not proposal.get("expression"):
        issues.append("custom缺expression；写出引用全部字段ID的计算表达式，不填最终答案")
    for f in fields:
        if not isinstance(f, dict):
            continue
        if not isinstance(f.get("id"), str) or not FIELD_ID.fullmatch(f["id"]):
            issues.append(f"字段{f.get('id')}：ID须小写字母开头，仅小写字母/数字/下划线，1到32字符；禁止tab:和E编号")
        if f.get("source") == "question":
            absent = [k for k in ("value", "unit", "quote") if f.get(k) in (None, "")]
            if absent:
                issues.append(f"字段{f.get('id')}缺" + "/".join(absent) + "；复制原题给定数值和单位；未给绝对金额则source=report")
    details = getattr(error, "details", {})
    if details.get('code') == 'unknown_formula_id':
        issues.append('formula_id不是业务名称；只使用本题提示中的注册ID，其他任务必须填字面量custom并给expression；禁止自造带年份/日期的ID')
    if details.get('code') == 'scenario_condition_omitted':
        issues.append('假设条件必须绑定相同指标/年份/日期/单位；与目标字段不同则展开会计依赖，不能只补一个未参与表达式的字段')
    if details.get('code') == 'registered_role_ids':
        issues.append('注册公式ID固定；若采用另一套依赖须用custom且说明完整expression，不混用注册ID和自定义字段')
    if 'unattributed numeric literal' in str(error):
        issues.append('单位换算由工具完成：百分数字段unit=%已除100，表达式不得再除100；其他真实金额必须声明source字段')
    useful = {k: details[k] for k in ("code", "field", "path", "candidate_count", "declared_ids", "undeclared_ids", "unused_ids", "required_ids", "missing_conditions", "actual_facts", "expected_candidates") if k in details}
    renames=[e['mapping'] for e in details.get('contract_normalisations',[]) if e['operation']=='rename_field_ids']
    if renames:useful['id_mapping']=renames
    return ("上次计划结构（quote/aliases省略，原回复已存轨迹）：" + json.dumps(compact, ensure_ascii=False,separators=(',',':')) +
            "\n具体错误：" + str(error) + "\n错误定位：" + json.dumps(useful, ensure_ascii=False) +
            "\n修复清单：" + "；".join(dict.fromkeys(issues)) +
            '\n'+repair_actions(error,question)+
            '\n多项检查：'+json.dumps(planning_issues(proposal,question,target),ensure_ascii=False,separators=(',',':'))+
            "\n完整重写JSON；question字段填value/unit/原文quote，report字段不猜数字，保持公司/指标/年份一致。")
