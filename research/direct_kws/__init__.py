"""Research-only three-class KWS; importing this package does not import Torch."""

from .model import CLASS_NAMES, DirectKWSSpec, build_classifier, window_cross_entropy
from .events import DirectEventHead, class_target, stream_events

__all__ = [
    "CLASS_NAMES", "DirectKWSSpec", "build_classifier", "window_cross_entropy",
    "DirectEventHead", "class_target", "stream_events",
]
