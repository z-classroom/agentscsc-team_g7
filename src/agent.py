from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

from .memory import Memory
from .policies import PolicyEngine, PolicyResult
from .llm_provider import LLMProvider
from .utils import safe_format


@dataclass
class Agent:
    cfg: Dict[str, Any]
    policies: Dict[str, Any]
    prompts: Dict[str, str]
    logger: Any

    def __post_init__(self) -> None:
        self.name = self.cfg.get("agent_name", "Agent")
        self.memory = Memory(
            enabled=self.cfg.get("memory", {}).get("enabled", True),
            max_messages=self.cfg.get("memory", {}).get("max_messages", 12),
        )
        self.policy_engine = PolicyEngine(self.policies)
        self.llm = LLMProvider.from_env_or_config(self.cfg)

    def _build_system_prompt(self) -> str:
        base = safe_format(self.prompts["system"], agent_name=self.name)
        style = self.prompts["style"]
        return f"{base}\n\n{style}".strip()

    def _refusal_response(self, user_text: str) -> str:
        text = user_text.strip().lower()

        if "lawyer" in text or "legal" in text:
            return (
                "I cannot help with that request. I am not a lawyer and cannot provide legal advice or act with professional legal authority.\n\n"
                "Safe alternative: I can give general informational guidance or help you prepare questions for a qualified attorney."
            )

        if "ignore all previous instructions" in text or "unrestricted mode" in text:
            return (
                "I cannot help with that request. I must operate within my defined role, safety rules, and system boundaries.\n\n"
                "Safe alternative: I can still help with allowed tasks such as explanations, summaries, or safer reformulations."
            )

        return (
            "I cannot help with that request. It violates the agent’s safety or scope rules.\n\n"
            "Safe alternative: I can help with defensive guidance, general risk explanation, or a safer version of the request."
        )

    def _clarify_response(self, user_text: str) -> str:
        return (
            "I need a bit more detail before I can answer reliably. "
            "Please state your goal, the context, and any constraints."
        )

    def respond(self, user_text: str) -> str:
        policy: PolicyResult = self.policy_engine.evaluate(user_text)

        self.logger.info("USER: %s", user_text)
        self.logger.info("POLICY: %s", policy.to_dict())

        if policy.action == "REFUSE":
            reply = self._refusal_response(user_text)
            self.memory.add(user_text, reply)
            self.logger.info("ASSISTANT: %s", reply)
            return reply

        if policy.action == "CLARIFY":
            reply = self._clarify_response(user_text)
            self.memory.add(user_text, reply)
            self.logger.info("ASSISTANT: %s", reply)
            return reply

        reply = self.llm.complete(
            system=self._build_system_prompt(),
            messages=self.memory.messages(),
            user=user_text,
            refusal_prompt=self.prompts["refusal"],
            mode="normal",
        )

        self.memory.add(user_text, reply)
        self.logger.info("ASSISTANT: %s", reply)
        return reply