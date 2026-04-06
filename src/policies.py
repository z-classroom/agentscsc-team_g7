from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import re


@dataclass
class PolicyResult:
    action: str  # "ALLOW", "REFUSE", or "CLARIFY"
    matched_rules: List[str]
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "matched_rules": self.matched_rules,
            "notes": self.notes,
        }


class PolicyEngine:
    def __init__(self, policy_yaml: Dict[str, Any]) -> None:
        self.rules = policy_yaml.get("rules", [])
        self.overrides = policy_yaml.get(
            "response_overrides",
            {"blocked": "REFUSE", "safe": "ALLOW"},
        )
        self.clarify_triggers = policy_yaml.get("clarify_triggers", {})

    def _needs_clarification(self, user_text: str) -> bool:
        text = user_text.strip().lower()
        if not text:
            return False

        word_count = len(text.split())
        short_input_max_words = self.clarify_triggers.get("short_input_max_words", 3)
        ambiguous_inputs = self.clarify_triggers.get("ambiguous_inputs", [])

        if text in ambiguous_inputs:
            return True

        if word_count <= short_input_max_words and any(
            phrase in text for phrase in ambiguous_inputs
        ):
            return True

        return False

    def evaluate(self, user_text: str) -> PolicyResult:
        text = user_text.strip()

        matched = []
        for rule in self.rules:
            for pattern in rule.get("block_patterns", []):
                if re.search(pattern, text, flags=re.IGNORECASE):
                    matched.append(rule.get("id", "unknown"))
                    break

        if matched:
            return PolicyResult(
                action=self.overrides.get("blocked", "REFUSE"),
                matched_rules=matched,
                notes="Blocked by policy patterns.",
            )

        if self._needs_clarification(text):
            return PolicyResult(
                action="CLARIFY",
                matched_rules=[],
                notes="Input is too vague or underspecified for a safe, reliable response.",
            )

        return PolicyResult(
            action=self.overrides.get("safe", "ALLOW"),
            matched_rules=[],
            notes="No policy match.",
        )