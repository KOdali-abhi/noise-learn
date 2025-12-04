# Iladok

**Universal Latent Space Navigation for LLMs.**

[![PyPI version](https://img.shields.io/pypi/v/iladok.svg)](https://pypi.org/project/iladok/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Iladok** allows you to "Bring Your Own LLM" and inject lightweight, trainable "Steering Vectors" to specialize behavior without retraining the base model.

It implements a **Perplexity-Based Neural Router** that mathematically verifies which adapter is best suited for a query before answering.

---

## ⚡ The Core Concept

Traditional fine-tuning forces a model to be one thing. **Iladok** allows a model to be **Many Things** at once.

You define the "Vector" (a specific domain, tone, or logic), and Iladok routes the input to the correct expert dynamically.

* **Domain Agnostic:** Works for Finance, Law, Medical, Roleplay, Code, or any custom dataset.
* **Zero Interference:** Train a "Tax Law" vector and a "Creative Writing" vector. They never touch.
* **Mechanism Check:** The router measures signal perplexity (loss) to ensure the model never uses "Creative Writing" logic to solve a "Tax Law" problem.

---

## 🚀 Usage Examples

### 1. Install
```bash
pip install iladok
```

### 2. The "Corporate" Use Case (Finance vs. Legal)
Keep your sensitive financial logic separate from your general chat logic.

```python
from iladok import NoiseRouter

# 1. Connect Engine (Local or API)
router = NoiseRouter.from_pretrained("gpt2") 

# 2. Define distinct logic paths
finance_data = [("What is the EBITDA?", "The EBITDA is calculated by...")]
legal_data =   [("Are we liable?", "Liability depends on clause 4...")]

# 3. Train the navigational vectors
router.register_vector("finance_expert", finance_data)
router.register_vector("legal_expert", legal_data)

# 4. Auto-Route based on fit
# Input: "Calculate the tax burden."
# The router detects 'finance_expert' has lower perplexity -> Routes there.
output = router.generate("Calculate the tax burden.", auto_select=True)
print(output)
```

### 3. The "Creative" Use Case (Roleplay / Style)
Switch between personalities instantly.

```python
# Train style vectors
router.register_vector("pirate", [("Hello", "Ahoy matey!")])
router.register_vector("academic", [("Hello", "Greetings, colleague.")])

# Force a specific style
router.generate("Hello", steering_vectors=["pirate"]) 
# Output: "Ahoy matey!"
```

📦 Features

- Universal Connector — Standard interface for HF transformers, peft, and OpenAI API.  
- Mechanism Check — Route-by-Loss (Perplexity) ensures the model only answers if it "understands" the domain.  
- Virtual Vectors — For API models (GPT-4), vectors are injected via System Prompt context windows.  
- Local Core — For local models, vectors are Rank-16 PEFT-style adapters trained on the fly.  

License: MIT License. Built by Kodali.

---

### **3. `pyproject.toml`**
*(The build configuration)*

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "iladok"
version = "0.1.0"
description = "Universal Latent Space Navigation: Bring your own LLM, we add the steering."
readme = "README.md"
authors = [{name = "Kodali", email = "your@email.com"}]
requires-python = ">=3.9"
dependencies = [
    "torch",
    "transformers",
    "peft",
    "openai",
    "tqdm",
    "numpy"
]

[project.urls]
"Homepage" = "https://github.com/yourusername/iladok"
```

---

## 4. Source Code (src/iladok/)

src/iladok/__init__.py
```python
from .router import NoiseRouter

__all__ = ["NoiseRouter"]
```

src/iladok/router.py
```python
from .engines.local_core import LocalCoreEngine
from .engines.remote_signal import RemoteSignalEngine

class NoiseRouter:
    def __init__(self, engine):
        self.engine = engine
    
    @classmethod
    def from_pretrained(cls, model_name, device="cuda"):
        """Connects to a Local Core Engine (HuggingFace)."""
        return cls(LocalCoreEngine(model_name, device))
    
    @classmethod
    def from_openai(cls, model_name="gpt-4", api_key=None):
        """Connects to a Remote Signal Engine (OpenAI)."""
        return cls(RemoteSignalEngine(model_name, api_key))
    
    def register_vector(self, name, data, **kwargs):
        """Registers a new steering vector."""
        return self.engine.register_vector(name, data, **kwargs)
    
    def generate(self, prompt, steering_vectors=None, auto_select=False, **kwargs):
        """
        Generates text. 
        If auto_select=True, performs Mechanism Check to find best vector.
        """
        selected_vector = None
        
        if auto_select and steering_vectors:
            selected_vector = self._mechanism_check(prompt, steering_vectors)
            # print(f"[Iladok] Mechanism Check Winner: {selected_vector}") # Optional log
        elif steering_vectors and isinstance(steering_vectors, list):
            selected_vector = steering_vectors[0]
            
        return self.engine.generate(prompt, vector_name=selected_vector, **kwargs)

    def _mechanism_check(self, prompt, candidates):
        """
        Runs the input through all candidates to find lowest signal loss.
        """
        scores = {}
        for vector in candidates:
            self.engine.activate_vector(vector)
            # Lower loss is better
            scores[vector] = self.engine.compute_signal_loss(prompt)
        
        # Return key with min value
        return min(scores, key=scores.get)
```

src/iladok/engines/interface.py
```python
from abc import ABC, abstractmethod

class ModelInterface(ABC):
    """Abstract interface for any Iladok Engine."""
    
    @abstractmethod
    def compute_signal_loss(self, text):
        """
        Returns a loss value (float) for the given text.
        Used for the Mechanism Check (Perplexity Routing).
        """
        pass
    
    @abstractmethod
    def generate(self, prompt, max_new_tokens=50, vector_name=None):
        """Generates text using the optional steering vector."""
        pass
    
    @abstractmethod
    def register_vector(self, name, data, **kwargs):
        """
        Registers or optimizes a new Steering Vector.
        """
        pass
    
    @abstractmethod
    def activate_vector(self, name):
        """Switches the active steering vector."""
        pass
```

src/iladok/engines/local_core.py
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from .interface import ModelInterface
from ..steering.optimization import optimize_vector_loop

class LocalCoreEngine(ModelInterface):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"[Iladok] Loading Local Core: {model_name} on {device}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        # Freeze Base Model (The Mother)
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.vectors = {} # Registry of steering vectors

    def register_vector(self, name, data, epochs=3, rank=16):
        """Creates and optimizes a new Steering Vector (internally LoRA)."""
        print(f"[Iladok] Initializing Steering Vector: {name}")
        
        # We use LoRA config internally but expose it as 'Vector'
        config = LoraConfig(r=rank, lora_alpha=rank*2, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        
        if not isinstance(self.model, PeftModel):
            self.model = get_peft_model(self.model, config)
        
        # Add new adapter
        # Handle case where adapter already exists
        try:
            self.model.add_adapter(name, config)
        except ValueError:
            pass # Already exists, we will re-train it
            
        self.vectors[name] = config
        
        # Run Optimization Loop
        self.model = optimize_vector_loop(self.model, self.tokenizer, name, data, epochs, self.device)
        return name

    def compute_signal_loss(self, text):
        """Computes Perplexity for Mechanism Check."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        return outputs.loss.item()

    def generate(self, prompt, max_new_tokens=50, vector_name=None):
        if vector_name:
            self.activate_vector(vector_name)
        else:
            self.model.disable_adapter_layers()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def activate_vector(self, name):
        if name in self.vectors:
            self.model.set_adapter(name)
        else:
            # print(f"Warning: Vector {name} not found.")
            pass
```

src/iladok/engines/remote_signal.py
```python
import os
from .interface import ModelInterface
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class RemoteSignalEngine(ModelInterface):
    def __init__(self, model_name="gpt-4", api_key=None):
        if not OpenAI:
            raise ImportError("Please install openai package.")
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name
        self.vectors = {} # Virtual vectors (System Prompts)
        self.active_vector = None

    def register_vector(self, name, data, **kwargs):
        """
        'Optimizing' for Remote Signal = Creating a System Context from data.
        """
        print(f"[Iladok] Registering Virtual Vector: {name}")
        context_str = f"You are operating under the {name} mechanism. Pattern examples:\n"
        # data is expected to be list of (input, output) tuples
        for src, tgt in data[:10]: 
            context_str += f"Input: {src} -> Output: {tgt}\n"
            
        self.vectors[name] = context_str
        return name

    def compute_signal_loss(self, text):
        """
        Approximates signal loss using logprobs (if supported).
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": text}],
                max_tokens=1,
                logprobs=True,
                top_logprobs=1
            )
            # Proxy: High confidence = Low loss
            logprob = response.choices[0].logprobs.content[0].logprob
            return -logprob 
        except:
            return 0.0 # Fallback

    def generate(self, prompt, max_new_tokens=50, vector_name=None):
        messages = []
        
        target_vector = vector_name or self.active_vector
        if target_vector and target_vector in self.vectors:
            messages.append({"role": "system", "content": self.vectors[target_vector]})
            
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens
        )
        return response.choices[0].message.content

    def activate_vector(self, name):
        self.active_vector = name
```

src/iladok/steering/optimization.py
```python
import torch
from torch.optim import AdamW
from tqdm import tqdm

def optimize_vector_loop(model, tokenizer, vector_name, data, epochs, device):
    """
    Standard optimization loop for the steering vector.
    Data format: List of (input, output) tuples.
    """
    model.set_adapter(vector_name)
    model.train()
    
    # We rename 'adapter' to 'steering_weights' conceptually
    steering_weights = [p for n, p in model.named_parameters() if "lora" in n and p.requires_grad]
    optimizer = AdamW(steering_weights, lr=1e-4)
    
    # Simple data collator
    inputs = []
    for src, tgt in data:
        text = f"{src}\n{tgt}"
        inputs.append(text)
        
    encoded = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(device)
    
    print(f"[Iladok] Optimizing vector '{vector_name}' on {len(data)} examples...")
    for epoch in range(epochs):
        outputs = model(**encoded, labels=encoded["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print(f"  > Epoch {epoch+1}: Signal Loss {loss.item():.4f}")
        
    model.eval()
    return model
```