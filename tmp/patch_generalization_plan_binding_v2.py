from pathlib import Path

src = Path('tmp/patch_generalization_plan_binding.py').read_text(encoding='utf-8')
old = """text = replace_once(text, '''    source_identity: str,\\n    seed: int,\\n) -> dict:\\n''', '''    source_identity: str,\\n    seed: int,\\n    plan: dict | None = None,\\n    plan_ordinal: int | None = None,\\n) -> dict:\\n''')"""
new = """text = replace_once(text, '''def build_seed(\\n    *,\\n    domain_metrics: dict,\\n    robustness: dict,\\n    tier: str,\\n    model_family: str,\\n    candidate_id: str,\\n    source_identity: str,\\n    seed: int,\\n) -> dict:\\n''', '''def build_seed(\\n    *,\\n    domain_metrics: dict,\\n    robustness: dict,\\n    tier: str,\\n    model_family: str,\\n    candidate_id: str,\\n    source_identity: str,\\n    seed: int,\\n    plan: dict | None = None,\\n    plan_ordinal: int | None = None,\\n) -> dict:\\n''')"""
if old not in src:
    raise SystemExit('old signature patch anchor not found')
src = src.replace(old, new, 1)
exec(compile(src, 'patch_generalization_plan_binding_v2', 'exec'))
