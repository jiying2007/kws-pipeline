"""Standard-library-only geometry and label contracts. No acoustic computation."""
import math

KEYWORDS = ('你好小窝', '小窝小窝')

def geometry(channels=32, stacks=3, stack_size=5, kernel=5, inputs=80):
    if min(channels, stacks, stack_size, inputs) < 1 or kernel < 3 or kernel % 2 != 1:
        raise ValueError('invalid geometry')
    dilations = [1] + [2**i for _ in range(stacks) for i in range(stack_size)]
    cache_lengths = [(kernel-1)*d for d in dilations]
    blocks = len(dilations)
    # Each block: DW kC+C, BN 2C, PW C²+C, BN 2C,
    # PW C²+C, BN 2C. BN running stats are buffers, not parameters.
    params = inputs*channels+channels + blocks*(2*channels**2+(kernel+9)*channels) + channels*2+2
    macs = inputs*channels + blocks*(2*channels**2+kernel*channels) + channels*2
    return dict(parameters=params, weight_macs_per_frame=macs,
                weight_macs_per_second=100*macs,
                cache_lengths=cache_lengths, cache_elements=channels*sum(cache_lengths),
                cache_fp32_bytes=4*channels*sum(cache_lengths),
                max_left_context_frames=sum(cache_lengths),
                receptive_field_frames=1+sum(cache_lengths),
                multiscale_rf_frames=[1+(kernel-1)+(i+1)*(kernel-1)*(2**stack_size-1) for i in range(stacks)],
                pcm_span_samples=400+160*sum(cache_lengths),
                bn_float_buffers=blocks*6*channels, bn_int64_counters=blocks*3)

def frame_count(samples):
    if samples < 0: raise ValueError('negative samples')
    return max(0, 1+(samples-400)//160)

def targets(statuses):
    """Explicit per-keyword statuses: other keyword presence never infers absence."""
    mapping = {'human_positive':(1,1.0), 'human_negative':(0,1.0),
               'weak_positive':(1,0.25), 'weak_negative':(0,0.25), 'unknown':(0,0.0)}
    if len(statuses) != 2: raise ValueError('two independent labels required')
    return tuple(mapping[s] for s in statuses)

def pool_scalar(logits, valid_frames, statuses):
    """Synthetic score fixture only, not a neural-network forward."""
    if not 0 < valid_frames <= len(logits): raise ValueError('empty or invalid valid length')
    labels = targets(statuses)
    total = denom = 0.0
    for k,(y,w) in enumerate(labels):
        if not w: continue
        z = max(row[k] for row in logits[:valid_frames])
        if not math.isfinite(z): raise ValueError('nonfinite fixture')
        total += w*(max(z,0)-z*y+math.log1p(math.exp(-abs(z))))
        denom += w
    if not denom: raise ValueError('all labels unknown; skip sample before forward')
    return total/denom

def dependency_ranges(lengths):
    """Symbolic RF endpoints; no tensor/model operations."""
    past = sum(lengths)
    return lambda t: (t-past,t)
