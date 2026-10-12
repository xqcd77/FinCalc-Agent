"""Question-driven dependency blueprints. No case IDs, answers or audit inputs.

A blueprint is a proposal shown to the model, never a silent replacement of a
failed model plan. Report values still go through retrieval and provenance checks.
"""
import ast
import json
import re
from datetime import datetime,timedelta
from financial_semantics_v4 import V4Error, canonical, metric_spans
from question_conditions_v4 import question_conditions, date_mentions, DIRECTION, resolve_question_entity


def short_quote(question,fact):
    """Copy an original metric-to-number span, without rewriting any digits."""
    metrics=[s for s in metric_spans(question[:fact['start']]) if s[2]==fact['metric']]
    start=metrics[-1][0] if metrics else fact['start']
    dates=[d for d in date_mentions(question[:start]) if start-d[1]<40]
    years=list(re.finditer(r'(?<!\d)20\d{2}(?!\d)',question[:start]))
    if dates and fact['metric'] in {'股价','涨跌金额','涨跌幅','总市值','成交量','流通股本'}:start=dates[-1][0]
    elif years and start-years[-1].end()<30:start=years[-1].start()
    end=fact['end']
    suffix=re.match(r'\s*(?:万元/股|元/股|人民币元|百万元|亿元|千元|万元|万股|亿股|元|股|[%％]|倍|天)',question[end:])
    if suffix:end+=suffix.end()
    return question[start:end]


def day_basis(question,method_notes=()):
    """Require an explicit, relevant sourced day convention; never assume 365."""
    target=next((m for m in ('应收账款','应付账款','存货','现金循环周期') if m in question),None)
    conventions=set()
    for text in method_notes:
        for sentence in re.split(r'[。；;\n]',text):
            if not target or target not in sentence or not any(t in sentence for t in ('周转','计算','现金循环')):continue
            conventions.update(re.findall(r'(?<!\d)(360|365)\s*(?:天|日|[/×*])',sentence))
    return next(iter(conventions)) if len(conventions)==1 else None


def _annual_year(question,method_notes=()):
    dates=date_mentions(question)
    years=[m.group(1) for m in re.finditer(r'(?<!\d)(20\d{2})(?!\d)',question) if not any(a<=m.start()<b for a,b,_ in dates)]
    if years:return years[-1]
    if not dates:return None
    cutoff=min(int(d[2][:4]) for d in dates)
    if any(re.search(r'静态市盈率.*总市值.*上一年度.*归属于母公司',text) for text in method_notes):return str(cutoff-1)
    source_years=[int(y) for text in method_notes for y in re.findall(r'(?<!\d)(20\d{2})(?!\d)',text) if int(y)<cutoff]
    return str(max(source_years)) if source_years else None


def blueprint(question,entity=None,method_notes=()):
    entity=resolve_question_entity(question,entity)
    conditions=question_conditions(question,entity)
    year=_annual_year(question,method_notes)
    if not year and '持有期间的收益率' in question and date_mentions(question):year=date_mentions(question)[0][2][:4]
    if not year:return None
    prior=str(int(year)-1)
    notes=[];fields=[]
    unit_match=re.search(r'(?:是多少|为多少|有多少)\s*(亿元|百万元|万元|千元|元)',question)
    money_unit=unit_match.group(1) if unit_match else '元'

    def condition(metric,kind=None,y=year):
        found=[c for c in conditions if c['metric']==metric and c.get('year')==y and (kind is None or c['condition_kind']==kind)]
        return found[0] if len(found)==1 else None

    def report(ident,metric,y=year):
        f=dict(id=ident,metric=metric,year=y,source='report')
        if any(c['metric']==metric and c['year']==y for c in conditions):f['value_basis']='historical'
        fields.append(f);return ident

    def given(ident,c):
        f=dict(id=ident,metric=c['metric'],year=c['year'],source='question',value=c['value'],unit=c['unit'],quote=short_quote(question,c))
        if c.get('date'):f['date']=c['date']
        fields.append(f);return ident

    def change(c):
        given('delta',c)
        m=re.search(r'(?:'+DIRECTION+r')\s*$',question[:c['start']])
        if not m:raise V4Error('semantic','变化数值无唯一方向',code='blueprint_change_direction')
        negative=bool(re.match(r'下降|减少|降低|下跌|下调|调低',m.group()))
        return '-' if negative else '+'

    def done(family,expression,unit='decimal',extra=()):
        # Do not offer a blueprint that silently drops an independent hypothesis.
        represented={(f['metric'],f['year'],f.get('date'),f.get('value')) for f in fields if f['source']=='question'}
        if any((c['metric'],c['year'],c.get('date'),c['value']) not in represented for c in conditions):return None
        return dict(family=family,plan=dict(formula_id='custom',answer_unit=unit,expression=expression,fields=fields),
                    notes=list(extra)+notes,source='question_conditions_and_displayed_methodology')

    if re.search(r'下一个交易日|下一交易日',question) and '市盈率' in question:
        next_dates=[m.group(1) for text in method_notes if (m:=re.search(r'已核验下一交易日：(20\d{2}-\d{2}-\d{2})',text))]
        if len(set(next_dates))!=1:return dict(family='next_pe_needs_calendar',plan=None,notes=['先核验基准日及下一交易日行情；不要用已知前日价直接计算下一日PE。'])
        day=next_dates[0];c=condition('每股收益','level');given('eps',c) if c else report('eps','每股收益')
        changes=[c for c in conditions if c['metric']=='涨跌金额' and c.get('date')==day and c['condition_kind']=='level']
        from question_conditions_v4 import question_facts
        prices=[f for f in question_facts(question,entity) if f['metric']=='股价' and f.get('date') and f['date']<day]
        if len(changes)==1 and prices:
            given('previous_price',min(prices,key=lambda f:f['start']));given('price_delta',changes[0])
            return done('next_pe_from_price_change','round_cent(previous_price + price_delta) / eps')
        fields.append(dict(id='next_price',metric='股价',year=day[:4],date=day,source='report'))
        return done('next_pe_from_sourced_date','next_price / eps')

    # Ranking dates must come from displayed, issuer-checked price evidence.
    interval=re.search(r'(20\d{6})到(20\d{6})',question)
    ranking=next((w for w in ('次高','次低','最高','最低') if w in question),None)
    if interval and ranking and re.search(r'静态市盈率|市净率',question):
        calendars=[];series_metric='股价'
        for metric,label in [('股价','收盘价'),('涨跌幅','涨跌幅'),('涨跌金额','涨跌金额')]:
            found=[]
            for note in method_notes:
                m=re.search(r'已核验'+label+r'日期：(\[[^\]]*\])',note)
                if m:found.extend(json.loads(m.group(1)))
            if found:calendars=found;series_metric=metric;break
        days=sorted({d for d in calendars if interval.group(1)<=d.replace('-','')<=interval.group(2)})
        begin,end=(datetime.strptime(x,'%Y%m%d') for x in interval.groups())
        if not 0<=(end-begin).days<=14:return None
        weekdays={(begin+timedelta(days=i)).strftime('%Y-%m-%d') for i in range((end-begin).days+1) if (begin+timedelta(days=i)).weekday()<5}
        if set(days)!=weekdays:return dict(family='rank_valuation_needs_calendar',plan=None,notes=['先检索目标公司区间行情表，逐项核验实际交易日；工作日尚未完全覆盖或可能休市时，先核对交易日历，不凭缺行推断。'])
        metric='每股收益' if '市盈率' in question else '每股净资产'
        c=condition(metric,'level');given('per_share',c) if c else report('per_share',metric)
        prices=[];derived=None
        if series_metric!='股价':
            from question_conditions_v4 import question_facts
            bases=[f for f in question_facts(question,entity) if f['metric']=='股价' and f.get('date') and f['date']<days[0]]
            if len(bases)!=1:return None
            base_date=datetime.strptime(bases[0]['date'],'%Y-%m-%d')+timedelta(days=1)
            while base_date.weekday()>=5:base_date+=timedelta(days=1)
            if base_date.strftime('%Y-%m-%d')!=days[0]:return None
            derived=given('previous_price',bases[0])
        for day in days:
            ident=('price_' if series_metric=='股价' else 'change_')+day.replace('-','')
            if series_metric!='股价':
                fields.append(dict(id=ident,metric=series_metric,year=day[:4],date=day,source='report'))
                derived=f'round_cent({derived} * (1 + {ident}))' if series_metric=='涨跌幅' else f'round_cent({derived} + {ident})'
                prices.append(derived);continue
            facts=[c for c in conditions if c['metric']=='股价' and c.get('date')==day and c['condition_kind']=='level']
            if len(facts)==1:given(ident,facts[0])
            elif not facts:fields.append(dict(id=ident,metric='股价',year=day[:4],date=day,source='report'))
            else:return None
            prices.append(ident)
        if len(prices)>10 or (ranking in {'次高','次低'} and len(prices)<2):return None
        function={'最高':'max','最低':'min','次高':'nth_largest','次低':'nth_smallest'}[ranking]
        expression=function+'('+('2, ' if ranking.startswith('次') else '')+', '.join(prices)+') / per_share'
        return done('rank_valuation_from_sourced_calendar' if series_metric=='股价' else 'rank_valuation_from_daily_changes',expression,extra=('行情日期来自本次显示的目标公司表；涨跌幅由工具按%换算一次，逐日从前一收盘价递推并按0.01取价，不把推导价格伪装为原题直接值。',))

    c=condition('带息债务','level')
    if c and re.search(r'筹资活动现金流量净额\s*/\s*有息负债',question):
        report('financing_cashflow','筹资现金流');given('interest_debt',c)
        return done('financing_cashflow_over_interest_debt','financing_cashflow / interest_debt')

    if '静态市盈率' in question and any(re.search(r'静态市盈率.*总市值.*上一年度.*归属于母公司',text) for text in method_notes):
        caps=[c for c in conditions if c['metric']=='总市值' and c.get('date') and c['condition_kind']=='level']
        if len(caps)==1:
            given('market_cap',caps[0]);report('parent_profit','归母净利润')
            return done('static_pe_from_market_cap','market_cap / parent_profit',extra=('总市值和总归母净利润口径成对；不能将总市值除以每股收益。财年按已显示上一年度定义。',))

    if '持有期间的收益率' in question and re.search(r'不考虑.*分红.*交易费用',question):
        dates=sorted({d[2] for d in date_mentions(question)})
        if len(dates)==2:
            sales=[c for c in conditions if c['metric']=='股价' and c.get('date')==dates[1] and c['condition_kind']=='level']
            if len(sales)==1:
                fields.append(dict(id='buy_price',metric='股价',year=dates[0][:4],date=dates[0],source='report'));given('sell_price',sales[0])
                return done('holding_return_without_costs','(sell_price - buy_price) / buy_price',extra=('无分红及费用时，买卖股数相同，可从收益率分子分母约去；不能把卖出价引用为买入价。',))

    if re.search(r'市销率|\bPS\b',question,re.I) and len({d[2] for d in date_mentions(question)})==1:
        day=date_mentions(question)[0][2]
        c=condition('营业收入','level');given('revenue',c) if c else report('revenue','营业收入')
        caps=[c for c in conditions if c['metric']=='总市值' and c.get('date')==day and c['condition_kind']=='level']
        if len(caps)==1:given('market_cap',caps[0])
        elif not caps:fields.append(dict(id='market_cap',metric='总市值',year=day[:4],date=day,source='report'))
        else:return None
        return done('ps_market_cap_over_revenue','market_cap / revenue')

    c=condition('流动资产','level')
    if c and re.search(r'(?:总资产|资产总计|股东权益总和|所有者权益(?:合计)?)是多少',question):
        given('current_assets',c);report('noncurrent_assets','非流动资产')
        if re.search(r'股东权益总和|所有者权益(?:合计)?是多少',question):
            report('liabilities','负债合计')
            return done('equity_from_asset_components','current_assets + noncurrent_assets - liabilities',money_unit)
        return done('assets_from_components','current_assets + noncurrent_assets',money_unit)

    c=condition('权益乘数','level')
    if c and re.search(r'流动负债是多少',question):
        report('assets','资产总计');given('multiplier',c);report('noncurrent_liabilities','非流动负债')
        return done('current_liabilities_from_multiplier','assets - assets / multiplier - noncurrent_liabilities',money_unit)
    if c and '少数股东权益是多少' in question:
        report('assets','资产总计');given('multiplier',c);report('parent_equity','归母权益')
        return done('minority_from_total_assets','assets / multiplier - parent_equity',money_unit)

    c=condition('资产负债率','level')
    if c and re.search(r'归属于母公司所有者权益是多少|归母权益是多少',question):
        report('assets','资产总计');given('debt_ratio',c);report('minority','少数股东权益')
        return done('parent_equity_from_debt_ratio','assets * (1 - debt_ratio) - minority',money_unit)

    c=condition('归母权益','percent_change')
    if c and '非流动资产是多少' in question:
        report('liabilities','负债合计');report('parent_equity','归母权益');report('minority','少数股东权益');report('current_assets','流动资产')
        sign=change(c)
        return done('noncurrent_assets_changed_parent_equity',f'liabilities + parent_equity * (1 {sign} delta) + minority - current_assets',money_unit)

    c=condition('流动负债','percent_change')
    if c and re.search(r'(?:负债合计|总负债)是多少',question):
        report('current_liabilities','流动负债');report('noncurrent_liabilities','非流动负债');sign=change(c)
        return done('total_liabilities_changed_component',f'current_liabilities * (1 {sign} delta) + noncurrent_liabilities',money_unit)

    c=condition('流动负债','level')
    if c and '产权比率' in question and re.search(r'其他条件.*均不变',question):
        given('current_liabilities',c);report('noncurrent_liabilities','非流动负债');report('equity','所有者权益')
        return done('debt_equity_from_liability_components','(current_liabilities + noncurrent_liabilities) / equity',extra=('假设流动负债不是总负债；其他数据不变时，分子补上非流动负债，分母使用原总所有者权益。',))

    c=condition('存货','level')
    if c and '流动比率' in question:
        report('quick','速动比率');report('current_liabilities','流动负债');given('inventory',c)
        return done('current_ratio_from_quick_inventory','(quick * current_liabilities + inventory) / current_liabilities')

    c=condition('毛利','percent_change')
    if c and re.search(r'营业收入(?:为多少|是多少)',question):
        report('gross_profit','毛利');report('margin','销售毛利率');sign=change(c)
        return done('revenue_changed_gross_profit',f'gross_profit * (1 {sign} delta) / margin',money_unit)

    c=condition('负债合计')
    if c and re.search(r'(?:总资产|资产总计)复合增长率',question):
        years=re.findall(r'(?<!\d)(20\d{2})(?!\d)',question)
        begin=min(years)
        from field_plan_v4 import cagr_source_basis
        begin=cagr_source_basis(method_notes) or begin
        n=int(year)-int(begin)
        if not 1<=n<=12:return None
        report('start_assets','资产总计',begin);report('equity','所有者权益')
        if c['condition_kind']=='level':
            given('liabilities',c);endpoint='equity + liabilities'
        elif c['condition_kind']=='percent_change':
            report('liabilities','负债合计');sign=change(c);endpoint=f'equity + liabilities * (1 {sign} delta)'
        else:return None
        return done('asset_cagr_from_liabilities',f'(({endpoint}) / start_assets) ** (1 / {n}) - 1',
                    extra=('负债变动通过资产=负债+权益影响资产，不是直接按负债变化率乘总资产。',))

    if re.search(r'(?:应收账款|应付账款)周转天数',question):
        metric='应收账款' if '应收账款周转天数' in question else '应付账款'
        denom='营业收入' if metric=='应收账款' else '营业成本'
        days=day_basis(question,method_notes)
        if not days:return dict(family='turnover_days_needs_method',plan=None,notes=['先检索本报告周转天数计算方法，确认360/365天；不要把年度天数虚构为普通财务金额。'])
        report('start_balance',metric,prior);report('end_balance',metric);report('denominator',denom)
        start='start_balance';end='end_balance';denominator='denominator'
        relevant=[c for c in conditions if c['metric'] in {metric,denom}]
        if len(relevant)>1:return None
        if relevant:
            c=relevant[0]
            base='denominator' if c['metric']==denom else ('start_balance' if c['year']==prior else 'end_balance' if c['year']==year else None)
            if base is None:return None
            if c['condition_kind']=='level':
                fields[:]=[f for f in fields if f['id']!=base];replacement=given(base,c)
            else:
                sign=change(c);replacement=f'({base} * (1 {sign} delta))' if c['condition_kind']=='percent_change' else f'({base} {sign} delta)'
            if base=='start_balance':start=replacement
            elif base=='end_balance':end=replacement
            else:denominator=replacement
        return done('turnover_days_with_conditions',f'{days} * ({start} + {end}) / 2 / {denominator}',
                    extra=('天数基数仅来自本次显示的方法证据；余额和变化金额由工具统一单位。',))

    c=condition('应收账款','level')
    if c and re.search(r'现金循环周期是多少',question):
        days=day_basis(question,method_notes)
        if not days:return dict(family='cash_cycle_needs_method',plan=None,notes=['先检索现金循环周期与应收周转天数的360/365计算方法；金额不能冒充周期天数。'])
        report('cycle','现金循环周期');report('original_receivable','应收账款');given('end_receivable',c);report('revenue','营业收入')
        return done('cash_cycle_receivable_delta',f'cycle + {days} * (end_receivable - original_receivable) / 2 / revenue',
                    extra=('其他数据不变，仅调整期末应收余额对平均余额的贡献；不能用金额代替周期。',))

    if '营业利润率' in question and re.search(r'扣非|扣除非经常',question):
        method=' '.join(method_notes)
        if not ('营业利润' in method and re.search(r'扣除|剔除',method) and '非经常性损益' in method):
            return dict(family='deducted_margin_needs_method',plan=None,notes=['先检索本报告扣非口径。非经常性损益、投资收益等是否重复包含须以该报告定义为准，不猜统一扣减项。'])
        roles=[('operating_profit','营业利润'),('nonrecurring','非经常性损益'),('revenue','营业收入')]
        deductions=['nonrecurring']
        if '投资收益' in method and '公允价值变动收益及营业外收支净额' in method:
            roles.extend([('investment_income','投资收益'),('fv_nonop','公允价值变动收益及营业外收支净额')]);deductions+=['investment_income','fv_nonop']
        for ident,metric in roles:
            c=condition(metric,'level');given(ident,c) if c else report(ident,metric)
        return done('deducted_margin_source_contract','(operating_profit - '+' - '.join(deductions)+') / revenue',
                    extra=('扣减项仅来自已显示报告口径，不把扣非通用化，也不重复减去同一项目。',))
    return None


def check_blueprint_plan(plan,question,entity=None,method_notes=()):
    """Validate a supported custom financial relationship using typed role mapping."""
    bp=blueprint(question,entity,method_notes)
    if not bp or not bp['plan'] or plan.get('formula_id')!='custom':return False
    if bp['family']=='ps_market_cap_over_revenue' and {canonical(f['metric']) for f in plan['fields']}=={'股价','总股本','营业收入'}:
        # PS has two equivalent, fully sourced dependency paths. Do not force
        # market-cap retrieval when same-date price and shares are available.
        roles={'股价':'price','总股本':'shares','营业收入':'revenue'}
        if len(plan['fields'])!=3:return False
        date=next(d[2] for d in date_mentions(question))
        for f in plan['fields']:
            if canonical(f['metric']) in {'股价','总股本'} and f.get('date')!=date:
                raise V4Error('semantic','PS价格/股本应对应估值基准日',code='blueprint_dependency_mismatch',field=f['id'])
            if canonical(f['metric'])=='营业收入' and f['year']!=_annual_year(question,method_notes):
                raise V4Error('semantic','PS收入期间不匹配',code='blueprint_dependency_mismatch',field=f['id'])
        mapping={f['id']:roles[canonical(f['metric'])] for f in plan['fields']}
        class PriceShares(ast.NodeTransformer):
            def visit_Name(self,node):return ast.copy_location(ast.Name(id=mapping.get(node.id,node.id),ctx=node.ctx),node)
        actual=PriceShares().visit(ast.parse(plan['expression'],mode='eval'))
        if ast.dump(actual)!=ast.dump(ast.parse('price * shares / revenue',mode='eval')):
            raise V4Error('semantic','PS应为价格×同口径总股本/营业收入',code='blueprint_expression_mismatch')
        return True
    expected=bp['plan']; mapping={};used=set()
    for role in expected['fields']:
        candidates=[f for f in plan['fields'] if canonical(f['metric'])==canonical(role['metric']) and f.get('year')==role['year']
                    and f['source']==role['source'] and f.get('date')==role.get('date')]
        if len(candidates)!=1 or candidates[0]['id'] in used:
            raise V4Error('semantic','依赖骨架字段缺失、错年、错来源或不唯一：'+role['id'],code='blueprint_dependency_mismatch',family=bp['family'],required_role=role)
        mapping[candidates[0]['id']]=role['id'];used.add(candidates[0]['id'])
        if role.get('value_basis'):candidates[0]['value_basis']=role['value_basis']
    if len(used)!=len(plan['fields']):
        raise V4Error('semantic','存在不属于本题金融依赖的额外字段',code='blueprint_extra_fields',family=bp['family'])
    class Rename(ast.NodeTransformer):
        def visit_Name(self,node):return ast.copy_location(ast.Name(id=mapping.get(node.id,node.id),ctx=node.ctx),node)
    actual=Rename().visit(ast.parse(plan['expression'],mode='eval'))
    required=ast.parse(expected['expression'],mode='eval')
    if ast.dump(actual)!=ast.dump(required):
        raise V4Error('semantic','运算关系与本题来源核定的依赖骨架不一致',code='blueprint_expression_mismatch',family=bp['family'],expected_expression=expected['expression'])
    return True
