"""Selective frozen BF16 storage and exact completion-only, chunked CE.

No full-sequence vocabulary logits are created. Checkpointing the projection/CE
chunks avoids retaining all chunk logits for backward. Decoder context is intact.
"""
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def parameter_memory(model):
    """Actual unique GPU/CPU storages; packed quantized numel is not base size."""
    seen=set()
    totals={}
    large=[]
    for name,p in model.named_parameters():
        storage=p.untyped_storage()
        key=(str(p.device),storage.data_ptr(),storage.nbytes())
        if key in seen:continue
        seen.add(key)
        category=f'{p.device}:{p.dtype}:{type(p).__name__}'
        size=storage.nbytes()
        totals[category]=totals.get(category,0)+size
        if size>=64*1024**2:
            large.append(dict(name=name,shape=list(p.shape),dtype=str(p.dtype),parameter_class=type(p).__name__,
                              requires_grad=p.requires_grad,storage_gib=size/1024**3))
    return dict(storage_bytes=sum(totals.values()),by_category_bytes=totals,
                large_parameters=sorted(large,key=lambda x:x['storage_gib'],reverse=True))


def restore_frozen_large_matrices_bf16(model):
    """Only embedding and ordinary, frozen output head; no blanket model cast."""
    events=[]
    seen=set()
    for role,module in [('input_embeddings',model.get_input_embeddings()),('output_head',model.get_output_embeddings())]:
        if module is None:raise ValueError(f'{role} absent')
        if type(module) not in (torch.nn.Embedding,torch.nn.Linear):
            raise ValueError(f'{role} must be an ordinary frozen matrix, got {type(module).__name__}')
        p=module.weight
        if p.requires_grad:raise ValueError(f'{role} unexpectedly trainable')
        if not p.is_floating_point() or type(p) is not torch.nn.Parameter:
            raise ValueError(f'{role} must not be a packed quantized parameter')
        if id(p) in seen:continue
        seen.add(id(p))
        old_dtype=str(p.dtype)
        old_bytes=p.untyped_storage().nbytes()
        p.data=p.data.to(torch.bfloat16)
        events.append(dict(role=role,old_dtype=old_dtype,new_dtype=str(p.dtype),parameters=p.numel(),
                           old_bytes=old_bytes,new_bytes=p.untyped_storage().nbytes()))
    return events


def completion_projection_loss(hidden,head,labels,chunk_tokens=64,checkpoint_chunks=True):
    """Same mean over nonignored target tokens as causal CE(ignore_index=-100).

    hidden[t] predicts labels[t+1], including the first assistant token. No
    supervised target is dropped. All prompt tokens remain decoder inputs.
    The output head is frozen and has no adapter under this project's contract.
    """
    if hidden.ndim!=3 or labels.ndim!=2 or hidden.shape[:2]!=labels.shape:
        raise ValueError('Hidden/labels batch and sequence dimensions must match')
    if type(head) is not torch.nn.Linear or head.weight.requires_grad or (head.bias is not None and head.bias.requires_grad):
        raise ValueError('Expected ordinary frozen lm_head; output-head LoRA is not supported')
    if not isinstance(chunk_tokens,int) or isinstance(chunk_tokens,bool) or chunk_tokens<1:
        raise ValueError('chunk_tokens must be a positive integer')
    shifted=labels[:,1:]
    mask=shifted.ne(-100)
    targets=shifted[mask]
    if targets.numel()==0:raise ValueError('No completion target survives causal shifting')
    if targets.min()<0 or targets.max()>=head.out_features:
        raise ValueError('Completion target outside vocabulary')
    active_hidden=hidden[:,:-1,:][mask]
    def project_and_sum(x,y):
        logits=F.linear(x.to(head.weight.dtype),head.weight,head.bias)
        return F.cross_entropy(logits.float(),y,reduction='sum')
    losses=[]
    for start in range(0,len(targets),chunk_tokens):
        x=active_hidden[start:start+chunk_tokens]
        y=targets[start:start+chunk_tokens]
        if checkpoint_chunks and torch.is_grad_enabled() and x.requires_grad:
            losses.append(checkpoint(project_and_sum,x,y,use_reentrant=False,preserve_rng_state=False))
        else:
            losses.append(project_and_sum(x,y))
    return torch.stack(losses).sum()/len(targets),dict(
        supervised_tokens=len(targets),sequence_positions=labels.numel(),
        projected_positions=len(targets),chunk_tokens=chunk_tokens,
        chunks=len(losses),causal_shift=True)


def optimized_model_loss(model,inputs,chunk_tokens=64):
    """Single-GPU standard LoRA; run backbone while bypassing full lm_head."""
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    labels=inputs.get('labels')
    if labels is None:raise ValueError('Completion-only training requires labels')
    kwargs={k:v for k,v in inputs.items() if k not in ('labels','num_items_in_batch')}
    kwargs.update(use_cache=False,return_dict=True)
    hooks=model._enable_peft_forward_hooks(**kwargs) if hasattr(model,'_enable_peft_forward_hooks') else nullcontext()
    with hooks:
        output=base.model(**kwargs)
    hidden=output.last_hidden_state
    return completion_projection_loss(hidden,base.get_output_embeddings(),labels,chunk_tokens)


def linear_kernel_status(model):
    """Report actual functions selected in each Qwen DeltaNet layer."""
    result=[]
    for name,module in model.named_modules():
        if 'GatedDeltaNet' not in type(module).__name__:continue
        functions={}
        for attr in ('causal_conv1d_fn','causal_conv1d_update','chunk_gated_delta_rule','recurrent_gated_delta_rule'):
            f=getattr(module,attr,None)
            functions[attr]=None if f is None else dict(module=getattr(f,'__module__',''),name=getattr(f,'__name__',type(f).__name__))
        result.append(dict(layer=name,functions=functions))
    return result


def require_fast_linear_kernels(model):
    layers=linear_kernel_status(model)
    if not layers:raise ValueError('No Gated DeltaNet layers found')
    for layer in layers:
        funcs=layer['functions']
        if not funcs['causal_conv1d_fn'] or not funcs['causal_conv1d_fn']['module'].startswith('causal_conv1d'):
            raise ValueError('Fast causal_conv1d not selected in '+layer['layer'])
        if not funcs['chunk_gated_delta_rule'] or not funcs['chunk_gated_delta_rule']['module'].startswith('fla'):
            raise ValueError('Fast FLA chunk kernel not selected in '+layer['layer'])
    return layers
