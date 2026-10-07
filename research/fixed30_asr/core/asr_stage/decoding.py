"""Conservative transcript extraction and token evidence, stdlib only."""
import re

MAX_TOKENS = 4096
MAX_TEXT = 32768
QWEN_EOS = (151645, 151643)
QWEN_PREFIX = re.compile(r"^language ([A-Za-z, ]+)\s*<asr_text>")


def token_ids(value, maximum=MAX_TOKENS, *, allow_empty=False):
    if type(value) is not list or not (0 if allow_empty else 1) <= len(value) <= maximum or any(type(x) is not int or x < 0 for x in value):
        raise ValueError("Invalid bounded decoder token IDs")
    return list(value)


def bounded_text(value):
    if type(value) is not str or len(value) > MAX_TEXT or any(0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise ValueError("Invalid decoder text")
    return value


def qwen_transcript(decoded, ids):
    decoded = bounded_text(decoded)
    ids = token_ids(ids, 256)
    match = QWEN_PREFIX.match(decoded)
    flags = []
    language = None
    if match:
        language = match.group(1).strip()
        text = decoded[match.end():]
        if language.lower() == "none": flags.append("non_speech")
    else:
        text = decoded; flags.append("decoding_warning")
    if "<asr_text>" in text or "<|" in text:
        flags.append("decoding_warning")
    terminated = ids[-1] in QWEN_EOS
    if not terminated: flags.append("incomplete")
    return {"raw_text": text, "completeness": "complete" if terminated else "incomplete",
            "quality_flags": sorted(set(flags))}, {"language": language, "eos_terminated": terminated,
            "generated_tokens": len(ids), "max_new_tokens": 256,
            "completeness_scope": "Decoder termination only; not acoustic transcript completeness"}




def qwen_generation_options():
    # The pinned top-level Qwen generate already passes return_dict_in_generate.
    # Supplying it again causes duplicate-keyword failure in thinker.generate.
    return {"max_new_tokens": 256, "do_sample": False, "eos_token_id": list(QWEN_EOS)}


class CapturingTokenizer:
    """Capture the exact post-CTC token sequence supplied to the real decoder.

    Does not replace model methods or alter tokens/text. Does not claim framewise
    logits/argmax evidence. Exactly one call is required for this single clip.
    """
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.calls = []

    def decode(self, ids, **kwargs):
        if self.calls or kwargs:
            raise ValueError("Unexpected tokenizer decode call count/options")
        frozen = token_ids(ids,allow_empty=True)
        sent = list(frozen)
        text = bounded_text(self._tokenizer.decode(sent))
        if ids != frozen or sent != frozen:
            raise ValueError("Tokenizer input was mutated")
        self.calls.append({"token_ids": frozen, "decoded": text})
        return text

    def evidence(self, expected_text):
        if len(self.calls) != 1 or self.calls[0]["decoded"] != expected_text:
            raise ValueError("Raw token capture does not match model output")
        return {"token_ids": list(self.calls[0]["token_ids"]), "decoded": expected_text,
                "token_scope": "Greedy CTC collapsed nonblank IDs passed to SentencePiece; not framewise logits"}
