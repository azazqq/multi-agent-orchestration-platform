import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import settings
from core.clients import APIError, build_client
from core.token_optimizer import TokenOptimizer

logger = logging.getLogger(__name__)


@dataclass
class AgentTurn:
    agent_name: str
    content: str


@dataclass
class GenerationResult:
    modality: str
    kind: str
    final_output: str = ""
    code: str = ""
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    execution_logs: str = ""
    comparison_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiAgentOrchestrator:
    def __init__(
        self,
        hf_token: str = "",
        github_token: str = "",
        openrouter_key: str = "",
        e2b_key: str = "",
        generator_model: str = settings.DEFAULT_DEEP_MODEL,
        critic_model: str = settings.DEFAULT_CODE_MODEL,
        arbiter_model: str = settings.DEFAULT_FAST_MODEL,
    ):
        self.hf_token = hf_token or settings.HF_TOKEN
        self.github_token = github_token or settings.GITHUB_TOKEN
        self.openrouter_key = openrouter_key or settings.OPENROUTER_API_KEY
        self.e2b_key = e2b_key or settings.E2B_API_KEY
        self.generator_model = generator_model
        self.critic_model = critic_model
        self.arbiter_model = arbiter_model
        self.token_optimizer = TokenOptimizer()

    async def classify_modality(self, goal: str, task_hint: Optional[str] = None) -> str:
        if task_hint:
            hint = task_hint.strip().lower()
            mapping = {"code": "CODE", "picture": "IMAGE", "image": "IMAGE", "video": "VIDEO"}
            if hint in mapping:
                return mapping[hint]
        text = goal.lower()
        if any(token in text for token in ["python", "script", "algorithm", "api", "data", "analysis", "plot", "web", "automation", "code", "function", "pipeline"]):
            return "CODE"
        if any(token in text for token in ["image", "photo", "poster", "logo", "illustration", "portrait", "diagram", "art", "visual", "ui", "banner", "design", "picture"]):
            return "IMAGE"
        if any(token in text for token in ["video", "animation", "clip", "movie", "simulation", "motion", "cinematic", "3d", "scene", "gif"]):
            return "VIDEO"
        return "CODE"

    async def _chat(self, provider: str, api_key: str, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1800) -> str:
        messages = self.token_optimizer.compact_messages(messages, max_total_chars=6000)
        client = build_client(provider, model, api_key)
        return await client.chat_completion(messages, model=model, temperature=temperature, max_tokens=max_tokens)

    async def _build_debate(self, modality: str, goal: str) -> Dict[str, str]:
        if modality == "CODE":
            generator_prompt = (
                "You are the principal architect. Write a clean, production-grade Python script that satisfies the goal. "
                "Do not use placeholders. Include error handling, output files, and realistic dependency assumptions. "
                f"Goal: {goal}"
            )
            critic_prompt = (
                "You are the adversarial reviewer. Check logic, edge cases, bare dependencies, and runtime issues. "
                "Keep the review compact but specific. "
                f"Goal: {goal}"
            )
            arbiter_prompt = (
                "You are the arbiter. Merge the best of both and return only executable Python code. "
                f"Goal: {goal}"
            )
        elif modality == "IMAGE":
            generator_prompt = (
                "You are a visual concept artist. Produce a compact but high-quality image prompt with subject, lighting, camera, composition, style, and colors. "
                f"Goal: {goal}"
            )
            critic_prompt = (
                "You are a visual critic. Improve the prompt for aesthetics and generation fidelity. Keep the critique short. "
                f"Goal: {goal}"
            )
            arbiter_prompt = (
                "You are the arbiter. Return only the final optimized image prompt. "
                f"Goal: {goal}"
            )
        else:
            generator_prompt = (
                "You are a cinematic director. Produce a concise but detailed video prompt including scene, movement, lighting, duration, and style. "
                f"Goal: {goal}"
            )
            critic_prompt = (
                "You are the video critic. Improve coherence, motion, camera work, and real-world feasibility. Keep the critique short. "
                f"Goal: {goal}"
            )
            arbiter_prompt = (
                "You are the arbiter. Return only the final video prompt. "
                f"Goal: {goal}"
            )

        return {
            "generator_prompt": generator_prompt,
            "critic_prompt": critic_prompt,
            "arbiter_prompt": arbiter_prompt,
        }

    async def compare_code_models(self, goal: str) -> List[Dict[str, Any]]:
        models = [
            ("FAST", settings.DEFAULT_FAST_MODEL, "openrouter"),
            ("DEEP", settings.DEFAULT_DEEP_MODEL, "openrouter"),
            ("CODE", settings.DEFAULT_CODE_MODEL, "openrouter"),
        ]
        results: List[Dict[str, Any]] = []
        for name, model_name, provider in models:
            prompt = (
                "Return only valid Python code. Keep it compact and executable. "
                f"Goal: {goal}"
            )
            msg = [{"role": "user", "content": prompt}]
            out = await self._chat(provider, self.openrouter_key, model_name, msg, temperature=0.6, max_tokens=1200)
            results.append({"model": name, "provider": provider, "output": out[:1400]})
        return results

    async def _build_messages(self, modality: str, goal: str, generator_response: str, critic_response: str) -> List[Dict[str, str]]:
        system_msg = self.token_optimizer.stable_system_prompt(goal, modality)
        volatile_tail = self.token_optimizer.stable_tail(goal)
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": volatile_tail},
            {"role": "user", "content": f"Draft: {generator_response[:1200]}"},
            {"role": "user", "content": f"Critique: {critic_response[:800]}"},
        ]

    async def run(self, user_goal: str, task_type: str = "AUTO") -> GenerationResult:
        modality = await self.classify_modality(user_goal, task_hint=task_type)
        prompts = await self._build_debate(modality, user_goal)
        logs: List[str] = []

        generator_messages = [{"role": "user", "content": prompts["generator_prompt"]}]
        generator_response = await self._chat(
            "openrouter",
            self.openrouter_key,
            self.generator_model,
            generator_messages,
            temperature=0.8,
            max_tokens=1800,
        )
        logs.append(f"Generator: {generator_response[:450]}")

        critic_messages = [{"role": "user", "content": prompts["critic_prompt"] + f"\n\nDraft output:\n{generator_response[:1600]}"}]
        critic_response = await self._chat(
            "openrouter",
            self.openrouter_key,
            self.critic_model,
            critic_messages,
            temperature=0.7,
            max_tokens=1200,
        )
        logs.append(f"Critic: {critic_response[:450]}")

        arbiter_messages = await self._build_messages(modality, user_goal, generator_response, critic_response)
        final_output = await self._chat(
            "openrouter",
            self.openrouter_key,
            self.arbiter_model,
            arbiter_messages,
            temperature=0.5,
            max_tokens=1500,
        )
        logs.append(f"Arbiter: {final_output[:450]}")

        result = GenerationResult(modality=modality, kind=modality.lower(), logs=logs)

        if modality == "CODE":
            code = final_output.strip()
            if "```" in code:
                code = re.sub(r"```(?:python)?", "", code).strip()
                code = re.sub(r"```$", "", code).strip()
            result.code = code
            result.final_output = code
            result.comparison_results = await self.compare_code_models(user_goal)
            return result

        if modality == "IMAGE":
            from core.media import save_bytes
            from core.clients import generate_pollinations_image

            image_bytes = await generate_pollinations_image(final_output)
            result.image_path = save_bytes("generated_image.png", image_bytes)
            result.final_output = final_output
            return result

        if modality == "VIDEO":
            from core.media import save_bytes
            from core.clients import generate_pollinations_video

            video_bytes = await generate_pollinations_video(final_output)
            result.video_path = save_bytes("generated_video.mp4", video_bytes)
            result.final_output = final_output
            return result

        return result
