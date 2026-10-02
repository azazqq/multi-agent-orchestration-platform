import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TokenBudget:
    system_tokens: int = 600
    context_budget: int = 2000
    answer_budget: int = 800
    trace_budget: int = 300


class PromptCache:
    def __init__(self):
        self._cache: Dict[str, str] = {}

    def key_for(self, prompt: str, model: str, temp: float) -> str:
        payload = json.dumps({"prompt": prompt, "model": model, "temp": temp}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, prompt: str, model: str, temp: float) -> Optional[str]:
        return self._cache.get(self.key_for(prompt, model, temp))

    def set(self, prompt: str, model: str, temp: float, value: str) -> None:
        self._cache[self.key_for(prompt, model, temp)] = value


class TokenOptimizer:
    def __init__(self, budget: Optional[TokenBudget] = None):
        self.budget = budget or TokenBudget()
        self.cache = PromptCache()

    def compact_text(self, text: str, max_chars: int = 1800) -> str:
        if len(text) <= max_chars:
            return text
        # Conservative truncation with explicit preview to preserve failure context.
        return text[: max_chars - 80].rstrip() + "\n... [truncated for token budget]"

    def compact_messages(self, messages: List[Dict[str, str]], max_total_chars: int = 6000) -> List[Dict[str, str]]:
        chars = 0
        compacted: List[Dict[str, str]] = []
        for msg in messages:
            content = msg.get("content", "")
            content = self.compact_text(content, max_chars=max(1200, min(2200, max_total_chars // max(1, len(messages)))))
            msg_copy = dict(msg)
            msg_copy["content"] = content
            compacted.append(msg_copy)
            chars += len(content)
            if chars > max_total_chars:
                break
        return compacted

    def stable_system_prompt(self, goal: str, modality: str) -> str:
        prefix = (
            "You are a concise, reliable assistant. "
            "Return only the minimal required output. "
            "No filler, no disclaimers, no markdown unless required. "
        )
        if modality == "CODE":
            return prefix + "Write production-ready Python. Keep logic tight and explicit."
        if modality == "IMAGE":
            return prefix + "Write a compact, high-quality image prompt."
        return prefix + "Write a concise, practical video prompt."

    def stable_tail(self, goal: str) -> str:
        # volatile part kept minimal so it does not poison the prompt cache.
        return f"Task: {goal[:300]}"

    def cached_call(self, call_fn, prompt: str, model: str, temperature: float, *args, **kwargs):
        cached = self.cache.get(prompt, model, temperature)
        if cached is not None:
            return cached
        value = call_fn(prompt, *args, **kwargs)
        self.cache.set(prompt, model, temperature, value)
        return value

    def summarize_tool_output(self, text: str, max_chars: int = 900) -> str:
        if len(text) <= max_chars:
            return text
        lines = text.splitlines()
        keep: List[str] = []
        for line in lines:
            if any(token in line.lower() for token in ["error", "traceback", "fail", "exception", "warning", "debug"]):
                keep.append(line)
                if len("\n".join(keep)) > max_chars:
                    break
        if keep:
            return "\n".join(keep[:20])
        return "\n".join(lines[:30])[:max_chars] + "\n... [compressed log]"


# --------------------------------------------------------------------------
# 10 upgraders: real operational changes to cut token usage in a multi-agent loop
# --------------------------------------------------------------------------

LEVELS = [
    "1. Prompt caching for stable prefixes",
    "2. Separate volatile goal tail from stable system prompt",
    "3. Strict max_tokens cap per role",
    "4. Tool log summarization before feeding context",
    "5. History compaction at phase boundaries",
    "6. Shared references instead of whole transcript pass-through",
    "7. Specialized sub-agents with narrow tool schemas",
    "8. Output schema enforcement (JSON/plain text only)",
    "9. Retrieval-only-on-demand + symbol-scoped reads",
    "10. Tiered model routing with cheap-seat first pass",
]
