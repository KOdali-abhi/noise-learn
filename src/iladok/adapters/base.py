"""
Abstract base class for adapters.

Defines the interface used by NoiseRouter and callers.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Tuple


class ModelAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Friendly adapter name.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_loss(self, text: str, noise: Optional[str] = None) -> float:
        """
        Compute a loss/perplexity-like scalar for `text` when applying `noise`.
        If `noise` is None this should compute the baseline (mother) model score.
        Lower is better.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, noise: Optional[str] = None, noise_config: Optional[dict] = None, **kwargs) -> Any:
        """
        Generate text conditioned on `prompt` with an optional named noise/profile.
        """
        raise NotImplementedError

    @abstractmethod
    def train_noise(self, name: str, data: Sequence[Tuple[str, str]], **kwargs) -> Any:
        """
        Train/register a new noise profile. `data` is a sequence of (input, target)
        pairs used to train the noise (e.g., few-shot or supervised pairs). Returns
        a descriptor or path for the created noise.
        """
        raise NotImplementedError

    @abstractmethod
    def set_active_noise(self, name: Optional[str]) -> None:
        """
        Make a noise profile active (or None to deactivate).
        """
        raise NotImplementedError