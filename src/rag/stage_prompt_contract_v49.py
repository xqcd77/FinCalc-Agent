"""Shared planner inputs for runtime and one-stage-per-question supervision."""
from field_plan_v4 import schema_hint, infer_formula, PLAN_SYSTEM
from question_conditions_v4 import condition_ledger, task_guidance
from financial_knowledge_v4 import planning_guidance
from prompt_budget import count_chat_tokens

VERSION='shared-stage-prompts-20261012'


def planning_input(question, entity):
    hint=schema_hint(question,entity=entity)
    base=f'目标公司：{entity}\n问题：{question}\n公式与依赖提示：{hint}'
    options=[]
    ledger=condition_ledger(question,entity)
    if ledger:options.append('\n题目条件索引（须回原题取quote）：'+ledger)
    guidance=task_guidance(question)
    if guidance:options.append('\n规划检查：'+guidance)
    cards=planning_guidance(question,infer_formula(question,entity))
    if cards:options.append('\n本题计算口径检查：\n'+cards)
    return base,hint,options


def append_planning_options(user, options, tokenizer):
    for extra in options:
        if count_chat_tokens(tokenizer,[dict(role='system',content=PLAN_SYSTEM),dict(role='user',content=user+extra)])<=1900:
            user+=extra
    return user
