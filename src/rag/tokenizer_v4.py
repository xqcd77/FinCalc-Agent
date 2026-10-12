"""Native model chat template on CPU; no substitute tokenizer for token budgeting."""
from pathlib import Path
from prompt_budget import load_tokenizer, resolve_tokenizer_path

def native_tokenizer(path=None):
    path=resolve_tokenizer_path(path)
    template=path.parent/'chat_template.jinja'
    if not template.exists():return load_tokenizer(path)
    from tokenizers import Tokenizer
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    tok=Tokenizer.from_file(str(path))
    env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
    def fail(message):raise ValueError(message)
    env.globals['raise_exception']=fail
    compiled=env.from_string(template.read_text(encoding='utf-8'))
    class Native:
        def encode(self,text,add_special_tokens=False):
            return tok.encode(text,add_special_tokens=add_special_tokens).ids
        def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True,enable_thinking=False):
            if tokenize:raise ValueError('Budget tokenizer expects tokenize=False')
            return compiled.render(messages=messages,add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,bos_token=None,eos_token='<|im_end|>')
    return Native()
