"""Short explicit solution steps inside JSON; executable fields stay authoritative."""
VERSION = 'v4.8-verified-explicit-solution-steps-20261007'
CHECKS = {'formula','dependencies','conditions','period','source','units','provenance','missing','repair','execution'}
INSTRUCTION = ('\nJSON内先写solution_steps：2至6个简短核验步骤，每项为'
    '{"check":"核验类别","detail":"本题具体判断","field_ids":["相关字段ID"]}。'
    'check可用formula/dependencies/conditions/period/source/units/provenance/missing/repair/execution。'
    '解释本题口径、期间、条件或证据选择，不写无依据数值或JSON外解释；步骤不能替代fields/bindings。')

def validate_steps(proposal, required=False):
    steps = proposal.get('solution_steps')
    if steps is None:
        if required: raise ValueError('Missing solution_steps')
        return []
    if not isinstance(steps,list) or not 2<=len(steps)<=6: raise ValueError('solution_steps must contain 2..6 items')
    groups=[proposal.get('fields',[]),proposal.get('bindings',[]),proposal.get('missing',[])]
    if any(not isinstance(g,list) for g in groups):raise ValueError('Invalid declared field/binding/missing list')
    if any(not isinstance(x,str) for x in groups[2]):raise ValueError('Invalid missing field ID')
    ids={f['id'] for g in groups[:2] for f in g if isinstance(f,dict) and isinstance(f.get('id'),str)}
    ids.update(groups[2])
    for s in steps:
        if not isinstance(s,dict) or set(s)!={'check','detail','field_ids'}: raise ValueError('Invalid solution step keys')
        if not isinstance(s['check'],str) or s['check'] not in CHECKS or not isinstance(s['detail'],str) or not 1<=len(s['detail'])<=180:
            raise ValueError('Invalid solution step category/detail')
        if not isinstance(s['field_ids'],list) or any(not isinstance(x,str) or x not in ids for x in s['field_ids']):
            raise ValueError('Solution step refers to undeclared field')
        if len(s['field_ids'])!=len(set(s['field_ids'])): raise ValueError('Duplicate step field ID')
    return steps
