"""Manifest-driven size gate, deliberately independent of GPU libraries."""

def declared_split_counts(manifest):
    counts=manifest.get('counts',{})
    splits=counts.get('splits',counts.get('split'))
    if not isinstance(splits,dict) or set(splits)!={'train','validation'}:
        raise ValueError('Manifest must declare train/validation counts')
    if any(type(n) is not int or n<=0 for n in splits.values()):
        raise ValueError('Manifest split counts must be positive integers')
    if type(counts.get('total')) is not int or counts['total']!=sum(splits.values()):
        raise ValueError('Manifest total differs from declared split counts')
    return dict(splits)

def validate_dataset_counts(manifest,verified,train_count,validation_count,require_individual=False):
    splits=declared_split_counts(manifest)
    if (train_count,validation_count)!=(splits['train'],splits['validation']):
        raise ValueError('Actual train/validation counts differ from verified manifest')
    total=sum(splits.values())
    if verified.get('status')!='passed' or verified.get('total_samples')!=total:
        raise ValueError('Verification must pass for the complete declared dataset')
    if require_individual and (verified.get('passed')!=total or verified.get('failed')!=0):
        raise ValueError('Every declared sample must pass individual audit')
    return splits
