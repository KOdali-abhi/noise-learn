"""
Tests for iladok package imports and basic structure.

These tests verify that:
1. All module imports work correctly
2. The NoiseRouter class has the expected API
3. Engine interfaces are properly defined
"""

import pytest


class TestImports:
    """Test that all package imports work correctly."""

    def test_main_import(self):
        """Test importing the main NoiseRouter class."""
        from iladok import NoiseRouter
        assert NoiseRouter is not None

    def test_engines_import(self):
        """Test importing engine classes."""
        from iladok.engines import ModelInterface, LocalCoreEngine, RemoteSignalEngine
        assert ModelInterface is not None
        assert LocalCoreEngine is not None
        assert RemoteSignalEngine is not None

    def test_steering_import(self):
        """Test importing steering optimization function."""
        from iladok.steering import optimize_vector_loop
        assert optimize_vector_loop is not None

    def test_noise_import(self):
        """Test importing noise training function."""
        from iladok.noise import train_lora_loop
        assert train_lora_loop is not None

    def test_adapters_import(self):
        """Test importing adapter classes."""
        from iladok.adapters import ModelAdapter, HuggingFaceAdapter, OpenAIAdapter
        assert ModelAdapter is not None
        assert HuggingFaceAdapter is not None
        assert OpenAIAdapter is not None


class TestNoiseRouterAPI:
    """Test that NoiseRouter has the expected API structure."""

    def test_from_pretrained_method_exists(self):
        """Test that from_pretrained class method exists."""
        from iladok import NoiseRouter
        assert hasattr(NoiseRouter, 'from_pretrained')
        assert callable(getattr(NoiseRouter, 'from_pretrained'))

    def test_from_openai_method_exists(self):
        """Test that from_openai class method exists."""
        from iladok import NoiseRouter
        assert hasattr(NoiseRouter, 'from_openai')
        assert callable(getattr(NoiseRouter, 'from_openai'))

    def test_register_vector_method_exists(self):
        """Test that register_vector instance method exists."""
        from iladok import NoiseRouter
        # Create a minimal mock router to test instance methods
        router = NoiseRouter(engine=None)
        assert hasattr(router, 'register_vector')
        assert callable(getattr(router, 'register_vector'))

    def test_generate_method_exists(self):
        """Test that generate instance method exists."""
        from iladok import NoiseRouter
        router = NoiseRouter(engine=None)
        assert hasattr(router, 'generate')
        assert callable(getattr(router, 'generate'))

    def test_mechanism_check_method_exists(self):
        """Test that _mechanism_check private method exists."""
        from iladok import NoiseRouter
        router = NoiseRouter(engine=None)
        assert hasattr(router, '_mechanism_check')
        assert callable(getattr(router, '_mechanism_check'))


class TestModelInterface:
    """Test that ModelInterface defines the expected abstract methods."""

    def test_interface_is_abstract(self):
        """Test that ModelInterface cannot be instantiated directly."""
        from iladok.engines import ModelInterface
        with pytest.raises(TypeError):
            ModelInterface()

    def test_interface_has_compute_signal_loss(self):
        """Test that interface defines compute_signal_loss."""
        from iladok.engines import ModelInterface
        assert hasattr(ModelInterface, 'compute_signal_loss')

    def test_interface_has_generate(self):
        """Test that interface defines generate."""
        from iladok.engines import ModelInterface
        assert hasattr(ModelInterface, 'generate')

    def test_interface_has_register_vector(self):
        """Test that interface defines register_vector."""
        from iladok.engines import ModelInterface
        assert hasattr(ModelInterface, 'register_vector')

    def test_interface_has_activate_vector(self):
        """Test that interface defines activate_vector."""
        from iladok.engines import ModelInterface
        assert hasattr(ModelInterface, 'activate_vector')
