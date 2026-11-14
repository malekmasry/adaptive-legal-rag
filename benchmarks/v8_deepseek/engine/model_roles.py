"""Explicit role contract for the DeepSeek evaluation line; no network calls."""
GENERATOR_MODEL = "deepseek-v4.1-flash"
JUDGE_MODEL = "deepseek-v4.1-flash"
RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"


def validate_roles(config):
    for role, model in (("GENERATOR", GENERATOR_MODEL), ("JUDGE", JUDGE_MODEL),
                        ("RERANKER", RERANKER_MODEL)):
        if config.get(role + "_MODEL") != model:
            raise ValueError(role + "_MODEL differs from the fixed model role")
        for suffix in ("_BASE_URL", "_API_KEY"):
            value = config.get(role + suffix)
            if not isinstance(value, str) or not value.strip() or value.startswith("REPLACE_"):
                raise ValueError("Missing explicit setting: " + role + suffix)
    for suffix in ("_MODEL", "_BASE_URL", "_API_KEY"):
        if config.get("LLM" + suffix) != config.get("GENERATOR" + suffix):
            raise ValueError("LLM runtime alias must match GENERATOR" + suffix)


def normalize(config):
    result = dict(config)
    for name, value in {"MAX_WORKERS": "3", "TOP_K": "12", "HTTP_TRUST_ENV": "1",
                        "EMB_MODEL": "BAAI/bge-m3", "AS_OF": "2026-11-14"}.items():
        result.setdefault(name, value)
    for suffix in ("_MODEL", "_BASE_URL", "_API_KEY"):
        value = result.get("GENERATOR" + suffix)
        if "LLM" + suffix in result and result["LLM" + suffix] != value:
            raise ValueError("Conflicting GENERATOR and LLM aliases")
        result["LLM" + suffix] = value
    validate_roles(result)
    return result
