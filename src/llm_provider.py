from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from google import genai
from google.genai import types


@dataclass
class LLMProvider:
    provider: str
    model: str

    @staticmethod
    def from_env_or_config(cfg: Dict[str, Any]) -> "LLMProvider":
        provider = os.getenv("LLM_PROVIDER") or cfg.get("llm_provider", "mock")
        model = os.getenv("LLM_MODEL") or cfg.get("llm_model", "mock-001")
        return LLMProvider(provider=provider, model=model)

    def complete(
        self,
        system: str,
        messages: List[Dict[str, str]],
        user: str,
        refusal_prompt: str,
        mode: str = "normal",
    ) -> str:
        if self.provider == "mock":
            return self._mock_response(system, messages, user, refusal_prompt, mode)

        if self.provider == "gemini":
            return self._gemini_response(system, messages, user, refusal_prompt, mode)

        raise ValueError(
            f"Unsupported llm_provider '{self.provider}'. Use 'mock' or 'gemini'."
        )

    def _gemini_response(
        self,
        system: str,
        messages: List[Dict[str, str]],
        user: str,
        refusal_prompt: str,
        mode: str,
    ) -> str:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Gemini API error: GOOGLE_API_KEY is missing from the environment."

        client = genai.Client(api_key=api_key)

        instructions = system.strip()
        if mode == "refusal":
            instructions = (
                f"{instructions}\n\n"
                f"{refusal_prompt.strip()}\n"
                "Do not answer the unsafe request. Refuse it clearly and offer only safe alternatives."
            )

        history_blocks: List[str] = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            history_blocks.append(f"{role}: {content}")

        history_text = "\n\n".join(history_blocks).strip()

        if history_text:
            prompt = (
                f"Conversation history:\n{history_text}\n\n"
                f"Current user request:\n{user}"
            )
        else:
            prompt = user

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=instructions,
                    temperature=0.2,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )

            text = getattr(response, "text", None)
            if text and text.strip():
                return text.strip()

            return "I could not produce a usable Gemini response."

        except Exception as e:
            return f"Gemini API error: {e}"

    def _mock_response(
        self,
        system: str,
        messages: List[Dict[str, str]],
        user: str,
        refusal_prompt: str,
        mode: str,
    ) -> str:
        text = user.strip().lower()

        if mode == "refusal":
            return (
                "I can’t help with that request.\n"
                "Reason: it appears to violate a course policy for safe and ethical behavior.\n"
                "Safe alternatives: I can explain the risks, provide defensive best practices, or help reframe the task safely."
            )

        if "difference between a refusal rule and a clarification rule" in text:
            return (
                "A refusal rule blocks a request that violates policy or exceeds the agent’s allowed scope. "
                "A clarification rule does not treat the request as disallowed. Instead, it pauses the response "
                "because the input is too vague or incomplete to answer reliably. In practice, refusal enforces a boundary, "
                "while clarification enforces input quality."
            )

        if text.startswith("summarize"):
            return (
                "Summary (mock): A safe agent should distinguish between disallowed requests and underspecified ones. "
                "Disallowed requests should be refused. Vague requests should trigger clarification before the agent answers."
            )

        if text.startswith("recommend"):
            return (
                "Recommendation (mock): I need your objective, context, and constraints before I can recommend an option responsibly."
            )

        if "policy" in text or "trust" in text or "boundary" in text:
            return (
                "Response (mock): In this system, trust is enforced through scoped identity, policy checks, and controlled refusal behavior rather than tone alone."
            )

        return (
            "Response (mock): Your request appears allowed and within scope. "
            "For a better answer, include your goal, context, and constraints."
        )