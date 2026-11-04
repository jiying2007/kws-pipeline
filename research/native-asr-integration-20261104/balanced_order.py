"""Quality-independent order fixed before inspecting expansion outputs."""
OFFSETS=((0,2,4,6),(1,3,5,7),(8,9,10,11),(12,13,14,15))
ORDINALS=tuple(voice*16+offset for offsets in OFFSETS for voice in range(4) for offset in offsets)

def validate_ordinals(values):
    if type(values) is not list or any(type(value) is not int for value in values) or values != list(ORDINALS):
        raise ValueError('exact precommitted balanced producer-ordinal order required')
    return True
