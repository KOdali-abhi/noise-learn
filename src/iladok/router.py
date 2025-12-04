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
            print(f"[Iladok] Mechanism Check Winner: {selected_vector}")
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