"""
OpenAIAdapter

- Noise = named system prompt constructed from few-shot examples (register via train_noise).
- forward_loss attempts to get token logprobs via the completions endpoint (echo=True).
  If unavailable, returns 0.0 as fallback (per specification).
- generate injects the active noise (system prompt) into messages and calls ChatCompletion.create.
"""

from typing import Optional, Any, Sequence, Tuple, Dict
import os
import logging

logger = logging.getLogger(__name__)


class OpenAIAdapter:
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._openai = None
        if self.api_key:
            try:
                import openai

                openai.api_key = self.api_key
                self._openai = openai
            except Exception:
                self._openai = None
        # noise registry: name -> system_prompt string
        self.noises: Dict[str, str] = {}
        self._active_noise: Optional[str] = None

    @property
    def name(self) -> str:
        return f"openai::{self.model}"

    def _ensure_client(self):
        if self._openai is None:
            try:
                import openai

                if self.api_key:
                    openai.api_key = self.api_key
                self._openai = openai
            except Exception as e:
                raise ImportError("openai package is required for OpenAIAdapter") from e

    def train_noise(self, name: str, data: Sequence[Tuple[str, str]], **kwargs) -> str:
        """
        For OpenAIAdapter, `data` is few-shot pairs (input, output) used to construct a system prompt.
        We create a simple textual system prompt that gives examples.
        """
        # Build system prompt from examples
        pieces = []
        pieces.append("You are a helpful assistant. Follow the examples below.")
        for i, (src, tgt) in enumerate(data):
            pieces.append(f"Example {i+1} input: {src}")
            pieces.append(f"Example {i+1} output: {tgt}")
        sys_prompt = "\n".join(pieces)
        self.noises[name] = sys_prompt
        return sys_prompt

    def set_active_noise(self, name: Optional[str]) -> None:
        if name is None:
            self._active_noise = None
        elif name not in self.noises:
            raise KeyError(f"Noise '{name}' not found for OpenAIAdapter")
        else:
            self._active_noise = name

    def forward_loss(self, text: str, noise: Optional[str] = None) -> float:
        """
        Attempt to compute a logprob-based perplexity-like score via the completions endpoint.
        If we cannot obtain logprobs, return 0.0 as fallback (per spec).
        """
        self._ensure_client()
        # Compose prompt with system prompt (noise) and user text
        sys = None
        if noise:
            sys = self.noises.get(noise)
        elif self._active_noise:
            sys = self.noises.get(self._active_noise)

        prompt = ""
        if sys:
            prompt += sys + "\n\n"
        prompt += text

        try:
            # Use Completions endpoint with echo=True and logprobs to get token logprobs for the prompt
            resp = self._openai.Completion.create(
                model=self.model,
                prompt=prompt,
                max_tokens=0,
                echo=True,
                logprobs=0,
            )
            choices = resp.get("choices", [])
            if not choices:
                return 0.0
            token_logprobs = choices[0].get("logprobs", {}).get("token_logprobs", [])
            if not token_logprobs:
                return 0.0
            import math

            avg_neg_logp = -sum(token_logprobs) / max(1, len(token_logprobs))
            perp = math.exp(avg_neg_logp)
            return float(perp)
        except Exception as e:
            logger.debug("OpenAI forward_loss failed (falling back to 0.0): %s", e)
            return 0.0

    def generate(self, prompt: str, noise: Optional[str] = None, noise_config: Optional[dict] = None, **kwargs) -> Any:
        """
        Generate via chat completion, injecting the noise as a system prompt.
        Returns the raw API response.
        """
        self._ensure_client()
        # choose system prompt
        sys = None
        if noise:
            sys = self.noises.get(noise)
        elif self._active_noise:
            sys = self.noises.get(self._active_noise)

        messages = []
        if sys:
            messages.append({"role": "system", "content": sys})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = self._openai.ChatCompletion.create(model=self.model, messages=messages, **kwargs)
            return resp
        except Exception:
            # Fall back to Completion endpoint if ChatCompletion not available
            try:
                # For completion, prefix system as text
                fused_prompt = (sys + "\n\n" if sys else "") + prompt
                resp = self._openai.Completion.create(model=self.model, prompt=fused_prompt, **kwargs)
                return resp
            except Exception as e:
                logger.exception("OpenAI generate failed: %s", e)
                raise