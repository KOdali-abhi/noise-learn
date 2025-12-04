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
            # The exact response structure varies; attempt to access a plausible field.
            logprob = response.choices[0].logprobs.content[0].logprob
            return -logprob 
        except Exception:
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
        # Extract content in a defensive way:
        try:
            return response.choices[0].message.content
        except Exception:
            # Fallback best-effort extraction
            try:
                return response.choices[0].text
            except Exception:
                return response

    def activate_vector(self, name):
        self.active_vector = name