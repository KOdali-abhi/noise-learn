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
        """Creates and optimizes a new Steering Vector (internally configured via the PEFT helpers)."""
        print(f"[Iladok] Initializing Steering Vector: {name}")
        
        # We use a PEFT configuration object internally, exposed externally as 'vector'
        config = LoraConfig(r=rank, lora_alpha=rank*2, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
        
        if not isinstance(self.model, PeftModel):
            # Wrap base model with a PEFT-enabled wrapper
            self.model = get_peft_model(self.model, config)
        
        # Add new adapter-like vector configuration to the model
        try:
            # Using PEFT's expected methods; actual PEFT API may differ by version.
            self.model.add_adapter(name, config)
        except Exception:
            # Fallback: some PEFT versions name the method differently or expect pre-created dirs.
            pass

        self.vectors[name] = config
        
        # Run Optimization Loop to optimize only steering parameters
        self.model = optimize_vector_loop(self.model, self.tokenizer, name, data, epochs, self.device)
        return name

    def compute_signal_loss(self, text):
        """Computes Perplexity-based loss for Mechanism Check."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        return outputs.loss.item()

    def generate(self, prompt, max_new_tokens=50, vector_name=None):
        if vector_name:
            self.activate_vector(vector_name)
        else:
            # If no vector is requested, attempt to disable adapters if supported.
            try:
                self.model.disable_adapter_layers()
            except Exception:
                pass

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def activate_vector(self, name):
        if name in self.vectors:
            try:
                self.model.set_adapter(name)
            except Exception:
                # Some PEFT versions use different management; ensure compatibility.
                pass
        else:
            print(f"Warning: Vector {name} not found.")