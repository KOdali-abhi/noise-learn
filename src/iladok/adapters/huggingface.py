"""
HuggingFaceAdapter implementation using transformers + peft.

Behavior:
- Loads tokenizer and a base AutoModelForCausalLM (frozen by default).
- Maintains a registry of named LoRA adapters saved to disk (./lora_adapters/<name> by default).
- train_noise uses `iladok.noise.training.train_lora_loop` to train a LoRA adapter and save it.
- set_active_noise loads the corresponding PEFT adapter and swaps the active model for inference/scoring.
- forward_loss uses the model(..., labels=input_ids).loss API (lower is better).
"""

from typing import Optional, Any, Sequence, Tuple, Dict
import os
import logging

logger = logging.getLogger(__name__)


class HuggingFaceAdapter:
    def __init__(self, model_name_or_path: str = "gpt2", device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu", lora_dir: str = "./lora_adapters"):
        self._model_name = model_name_or_path
        self.device = device
        self.lora_dir = lora_dir

        # Lazy imports at runtime
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            raise ImportError("transformers is required for HuggingFaceAdapter") from e

        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoTokenizer = AutoTokenizer

        # load tokenizer and base model
        self._tokenizer = self.AutoTokenizer.from_pretrained(self._model_name)
        self._base_model = self.AutoModelForCausalLM.from_pretrained(self._model_name)
        self._base_model.to(self.device)

        # freeze base model params (mother model remains frozen)
        for p in self._base_model.parameters():
            p.requires_grad = False

        # active model pointer (either base or a PeftModel wrapper)
        self._model = self._base_model

        # noise registry: name -> adapter_dir (on disk)
        self.noises: Dict[str, str] = {}
        # cached loaded PeftModel wrappers: name -> PeftModel instance
        self._loaded_wrappers: Dict[str, Any] = {}
        self._active_noise: Optional[str] = None

        # Ensure lora_dir exists
        os.makedirs(self.lora_dir, exist_ok=True)

    @property
    def name(self) -> str:
        return f"hf::{self._model_name}"

    def _ensure_peft_loaded(self, noise_name: str):
        """
        Ensure a PeftModel wrapper is loaded for `noise_name`. Returns the wrapper.
        """
        if noise_name in self._loaded_wrappers:
            return self._loaded_wrappers[noise_name]

        if noise_name not in self.noises:
            raise KeyError(f"Noise profile '{noise_name}' not found. Available: {list(self.noises.keys())}")

        try:
            from peft import PeftModel
        except Exception as e:
            raise ImportError("peft is required to load a LoRA adapter") from e

        adapter_dir = self.noises[noise_name]
        # Create a fresh copy of the base model instance to wrap to avoid mutating the canonical base in unexpected ways
        # (we take the base model config and instantiate a new model then move to device).
        wrapper_model = self.AutoModelForCausalLM.from_pretrained(self._model_name)
        wrapper_model.to(self.device)
        peft_wrapper = PeftModel.from_pretrained(wrapper_model, adapter_dir)
        peft_wrapper.to(self.device)
        # Do not freeze: the adapter parameters are trainable when training.
        self._loaded_wrappers[noise_name] = peft_wrapper
        return peft_wrapper

    def set_active_noise(self, name: Optional[str]) -> None:
        """
        Activate a named noise (LoRA adapter). If name is None, deactivate and use base model.
        """
        if name is None:
            self._model = self._base_model
            self._active_noise = None
            return

        if name == self._active_noise:
            return

        peft_wrapper = self._ensure_peft_loaded(name)
        self._model = peft_wrapper
        self._active_noise = name

    def train_noise(self, name: str, data: Sequence[Tuple[str, str]], epochs: int = 1, batch_size: int = 4, lr: float = 1e-4, output_dir: Optional[str] = None) -> str:
        """
        Train a LoRA adapter from `data` which is a sequence of (input, target) pairs.
        The adapter will be saved to ./lora_adapters/<name> (or output_dir/name).
        Returns the adapter dir path.
        """
        from iladok.noise.training import train_lora_loop

        outdir = output_dir or os.path.join(self.lora_dir, name)
        os.makedirs(outdir, exist_ok=True)

        logger.info("Starting LoRA training for noise '%s' -> saving to %s", name, outdir)
        # train_lora_loop will save the adapter to outdir
        train_lora_loop(
            base_model=self._model_name,
            dataset=data,
            adapter_output_dir=outdir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
        )

        # register the adapter dir
        self.noises[name] = outdir
        # remove any previously cached wrapper for the same name
        if name in self._loaded_wrappers:
            del self._loaded_wrappers[name]
        return outdir

    def forward_loss(self, text: str, noise: Optional[str] = None) -> float:
        """
        Compute loss (cross-entropy) for text under optional noise.
        """
        import torch
        import math

        # Activate requested noise temporarily
        prev = self._active_noise
        self.set_active_noise(noise)
        try:
            tok = self._tokenizer
            model = self._model
            enc = tok(text, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss.item() if hasattr(outputs, "loss") else None

                if loss is None:
                    # fallback compute from logits
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                    ).item()
            # convert to perplexity-like score
            try:
                perp = float(math.exp(loss))
            except Exception:
                perp = float("inf")
            return perp
        finally:
            # restore previous active noise state
            self.set_active_noise(prev)

    def generate(self, prompt: str, noise: Optional[str] = None, noise_config: Optional[dict] = None, max_new_tokens: int = 64, **kwargs) -> str:
        """
        Generate text under optional noise.
        """
        self.set_active_noise(noise)
        tok = self._tokenizer
        model = self._model
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        # ensure pad_token_id for models without it
        pad_token_id = getattr(tok, "eos_token_id", None)
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, pad_token_id=pad_token_id, **kwargs)
        out = tok.batch_decode(gen, skip_special_tokens=True)[0]
        return out