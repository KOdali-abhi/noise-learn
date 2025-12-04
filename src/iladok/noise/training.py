"""
Simple training helper that implements a LoRA training loop using transformers + peft.

Function: train_lora_loop

- Loads a fresh base model and tokenizer.
- Wraps model with PEFT LoraConfig via get_peft_model.
- Trains only PEFT parameters with AdamW.
- Saves the trained adapter to adapter_output_dir (PEFT save_pretrained).
"""

from typing import Sequence, Tuple, Optional
import os
import logging

logger = logging.getLogger(__name__)


def train_lora_loop(
    base_model: str,
    dataset: Sequence[Tuple[str, str]],
    adapter_output_dir: str,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    learning_rate: float = 1e-4,
):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        raise ImportError("transformers is required for train_lora_loop") from e

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as e:
        raise ImportError("peft is required for train_lora_loop") from e

    import torch
    from torch.utils.data import DataLoader

    # prepare output dir
    os.makedirs(adapter_output_dir, exist_ok=True)

    logger.info("Loading base model/tokenizer for training: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # prepare model for kbit/LoRA
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        # Not fatal; continue
        logger.debug("prepare_model_for_kbit_training not applied or failed (continuing).")

    # simple LoRA config
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] if hasattr(model, "get_input_embeddings") else None,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.to(device)

    # Build a list of concatenated examples (input + target)
    texts = []
    for src, tgt in dataset:
        # naive concatenation — for many tasks you'll want a more careful template
        texts.append(src + " " + tgt)

    # Tokenize all examples
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", None)

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, attention_mask=None):
            self.input_ids = input_ids
            self.attention_mask = attention_mask

        def __len__(self):
            return self.input_ids.size(0)

        def __getitem__(self, idx):
            item = {"input_ids": self.input_ids[idx]}
            if self.attention_mask is not None:
                item["attention_mask"] = self.attention_mask[idx]
            # labels equal input_ids (causal LM)
            item["labels"] = self.input_ids[idx].clone()
            return item

    train_dataset = SimpleDataset(input_ids, attention_mask)

    dataloader = DataLoader(train_dataset, batch_size=per_device_train_batch_size, shuffle=True)

    # Optimizer: only parameters with requires_grad=True (should correspond to LoRA params)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    model.train()
    logger.info("Starting training loop for %d epochs", num_train_epochs)
    for epoch in range(num_train_epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(dataloader))
        logger.info("Epoch %d completed — avg loss: %.6f", epoch + 1, avg_loss)

    # Save only the PEFT adapter weights
    try:
        model.save_pretrained(adapter_output_dir)
        logger.info("Saved adapter to %s", adapter_output_dir)
    except Exception:
        logger.exception("Failed to save adapter to %s", adapter_output_dir)

    return adapter_output_dir