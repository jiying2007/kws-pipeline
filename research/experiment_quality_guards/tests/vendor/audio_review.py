"""Exact text-only helpers used by the reviewed actual-label policy."""
import unicodedata
MAX_TEXT = 1024
class ReviewError(ValueError):
    """Invalid input; do not silently repair evidence."""


def require(condition, message):
    if not condition:
        raise ReviewError(message)


def valid_text(value):
    require(type(value) is str and len(value) <= MAX_TEXT and
            not any(0xD800 <= ord(c) <= 0xDFFF for c in value), "Invalid or oversized transcript")
    return value


def normalize(text):
    """Comparison only: no homophone, spelling, Unicode-form or repetition repair."""
    return "".join(c for c in valid_text(text)
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


