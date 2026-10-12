"""Offline trajectory metrics; no access to model prompts during generation.

Exact repetitions require identifiable arguments. Legacy traces with missing
arguments stay unknown; repairs, model calls and changed retrieval state are
never automatically counted as repeated tools.
"""
import json


def repeated_tools(output, context=None):
    events=(output.get('trace') or {}).get('events',[])
    seen={e['chunk_id'] for e in (context or {}).get('selected_evidence',[])}
    state_known=context is not None
    company=(context or {}).get('company_with_period')
    accepted=[e for e in events if e.get('plan_accepted') is True]
    fields={};entity=(context or {}).get('target_entity')
    counted=[];unknown=0;total=0
    for event in events:
        step=event.get('step')
        if event.get('plan_accepted') is True:
            entity=event.get('planning_entity',entity)
            if len(accepted)==1:
                plan=(output.get('trace') or {}).get('field_plan') or {}
            elif not event.get('contract_normalisations'):
                try:plan=json.loads(event.get('raw_response',''))
                except (ValueError,TypeError):plan={}
            else:plan={}
            fields={f['id']:f for f in plan.get('fields',[]) if isinstance(f,dict) and 'id' in f}
        if step=='planning_evidence_search':
            total+=1;unknown+=1
            # This legacy event records added_refs but not exact arguments or
            # source IDs. Following retrieval state is conservatively unknown.
            if event.get('added_refs'):state_known=False
        elif step=='targeted_retrieval':
            for field_id in event.get('requested_fields',[]):
                total+=1
                candidates=[c for c in event.get('candidates',[]) if c.get('field_id')==field_id]
                queries={c.get('query') for c in candidates if c.get('query')}
                if len(queries)==1 and state_known and field_id in fields:
                    counted.append(('retrieval',dict(query=next(iter(queries)),company=company,
                        entity=entity,field=fields[field_id],seen_chunk_ids=sorted(seen))))
                else:unknown+=1
                # Per-field retrieval executes before adding its selected group.
                seen.update(a['chunk_id'] for a in event.get('added',[]) if a.get('field_id')==field_id)
        elif step=='python_decimal_tool':
            total+=1;result=event.get('result') or {}
            if result.get('expression') is not None and isinstance(result.get('operands'),list):
                counted.append(('decimal',dict(expression=result['expression'],answer_unit=result.get('answer_unit'),
                    operands=result['operands'],normalized_base_values=result.get('normalized_base_values'))))
            else:unknown+=1
    keys=set();duplicates=0;by_tool={}
    for name,args in counted:
        key=(name,json.dumps(args,ensure_ascii=False,sort_keys=True,separators=(',',':')))
        repeated=key in keys;duplicates+=int(repeated);keys.add(key)
        bucket=by_tool.setdefault(name,dict(identified_calls=0,repeated_calls=0))
        bucket['identified_calls']+=1;bucket['repeated_calls']+=int(repeated)
    return dict(logged_tool_calls=total,identifiable_tool_calls=len(counted),
                unidentifiable_tool_calls=unknown,repeated_tool_calls=duplicates,by_tool=by_tool)
