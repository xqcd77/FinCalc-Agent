"""Bounded, explicit run overrides; historical audited dataset stays immutable."""
import copy

def merge_profile(base,profile):
    allowed={'lora_r','lora_alpha','micro_batch_size'}
    overrides=profile.get('overrides',{})
    if not isinstance(overrides,dict) or set(overrides)-allowed:
        raise ValueError('Only rank, alpha and micro batch may be overridden')
    if profile.get('target_scope')!='language_attention_and_ffn':
        raise ValueError('Unexpected target scope')
    if overrides.get('lora_r')!=16 or overrides.get('lora_alpha')!=32:
        raise ValueError('This approved profile requires r16/alpha32')
    for key,value in overrides.items():
        if type(value) is not int or value<=0:raise ValueError('Invalid override: '+key)
    if overrides.get('micro_batch_size',base['micro_batch_size'])>base['micro_batch_size']:
        raise ValueError('Micro batch may not increase')
    optim=profile.get('memory_optimizations',{})
    expected={'frozen_large_matrices_dtype':'bfloat16',
              'loss_backend':'completion_projection_chunk_checkpoint','require_fast_linear_kernels':True}
    if set(optim)!=set(expected)|{'projection_chunk_tokens'}:
        raise ValueError('Unexpected optimization keys')
    if any(optim.get(k)!=v for k,v in expected.items()):raise ValueError('Unsupported optimization')
    chunk=optim['projection_chunk_tokens']
    if type(chunk) is not int or not 1<=chunk<=128:raise ValueError('Unsafe projection chunk size')
    result=copy.deepcopy(base)
    result.update(overrides)
    return result
