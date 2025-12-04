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