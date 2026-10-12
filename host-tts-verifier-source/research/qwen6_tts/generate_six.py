"""Native-I/O fixture helper only. No generator, model import or execution entry."""
import math

def require(value, message):
    if not value: raise ValueError(message)

def signal_statistics(samples):
    values=[float(x) for x in samples]
    finite=[x for x in values if math.isfinite(x)]
    return {'nonfinite_count':len(values)-len(finite),
            'peak':max(map(abs,finite)) if finite else None,
            'abs_ge_one_count':sum(abs(x)>=1 for x in finite),
            'saturation_count':sum(x>=1 or x < -1 for x in finite)}

def derivative_signal(samples, expected_frames):
    """Check resampling output before PCM16 conversion can conceal overshoot."""
    require(0<len(samples)<=192000 and len(samples)==expected_frames,'Derivative frame count mismatch')
    statistics=signal_statistics(samples)
    require(statistics['nonfinite_count']==0,'Nonfinite derivative waveform')
    peak=statistics['peak']
    return peak,(['normalized_peak_ge_one'] if peak>=1.0 else [])
