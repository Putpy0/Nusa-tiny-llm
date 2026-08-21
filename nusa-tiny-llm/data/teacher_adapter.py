"""
Teacher Adapter Interface

Provider-agnostic interface for generating synthetic data responses.
Supports mock, local CLI, and API-based providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import uuid
from datetime import datetime


class TeacherAdapter(ABC):
    """
    Abstract base class for teacher adapters.
    
    Provider-agnostic interface for generating synthetic training data.
    """
    
    @abstractmethod
    def generate(self, prompt: str, lang: str = "en", data_type: str = "qa") -> Dict[str, Any]:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: Input prompt
            lang: Language code ('en' or 'id')
            data_type: Type of data being generated
            
        Returns:
            Dictionary with generated sample
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this adapter is available/configured."""
        pass


class MockTeacherAdapter(TeacherAdapter):
    """
    Mock teacher adapter for testing and development.
    
    Generates simple pattern-based responses without requiring
    external APIs or models. Completely offline.
    """
    
    # Simple response templates for English
    RESPONSES_EN = {
        "translation": [
            "Ini adalah terjemahan dalam Bahasa Indonesia.",
            "Terjemahan: Halo, apa kabar?",
            "Dalam Bahasa Indonesia: Selamat pagi.",
        ],
        "qa": [
            "This is a sample answer to the question.",
            "The answer involves several factors to consider.",
            "Based on my knowledge, I can explain this topic.",
            "This is an interesting question about general knowledge.",
        ],
        "instruction": [
            "Here is a sentence using the requested word.",
            "I can create content based on your instruction.",
            "Following your request, here is the output.",
        ],
        "conversation": [
            "<assistant> Hello! How can I help you today? <eos>",
            "<assistant> I'm doing well, thank you for asking. <eos>",
            "<assistant> I can assist with various tasks. <eos>",
        ],
        "reasoning": [
            "Based on logical analysis, the conclusion follows from the premises.",
            "The solution requires step-by-step reasoning through the problem.",
            "Comparing the two items reveals both similarities and differences.",
        ],
        "safety": [
            "I cannot provide information that could be harmful or dangerous.",
            "I'm not able to assist with requests that may cause harm.",
            "Let's focus on safer and more positive topics instead.",
        ],
    }
    
    # Simple response templates for Indonesian
    RESPONSES_ID = {
        "translation": [
            "This is the English translation.",
            "Translation: Good morning, how are you?",
            "In English: Thank you very much.",
        ],
        "qa": [
            "Ini adalah jawaban contoh untuk pertanyaan tersebut.",
            "Jawabannya melibatkan beberapa faktor yang perlu dipertimbangkan.",
            "Berdasarkan pengetahuan saya, saya dapat menjelaskan topik ini.",
        ],
        "instruction": [
            "Berikut adalah kalimat menggunakan kata yang diminta.",
            "Saya dapat membuat konten berdasarkan instruksi Anda.",
            "Mengikuti permintaan Anda, berikut adalah hasilnya.",
        ],
        "conversation": [
            "<assistant> Halo! Ada yang bisa saya bantu? <eos>",
            "<assistant> Kabar baik, terima kasih sudah bertanya. <eos>",
            "<assistant> Saya bisa membantu berbagai tugas. <eos>",
        ],
        "reasoning": [
            "Berdasarkan analisis logis, kesimpulan mengikuti dari premis.",
            "Solusinya memerlukan penalaran langkah demi langkah melalui masalah.",
            "Membandingkan kedua item mengungkapkan persamaan dan perbedaan.",
        ],
        "safety": [
            "Saya tidak dapat memberikan informasi yang bisa berbahaya.",
            "Saya tidak bisa membantu permintaan yang mungkin menyebabkan kerugian.",
            "Mari fokus pada topik yang lebih aman dan positif.",
        ],
    }
    
    def __init__(self):
        self.call_count = 0
    
    def generate(
        self,
        prompt: str,
        lang: str = "en",
        data_type: str = "qa"
    ) -> Dict[str, Any]:
        """Generate mock response."""
        import random
        
        self.call_count += 1
        
        # Select appropriate response bank
        if lang == "id":
            responses = self.RESPONSES_ID.get(data_type, self.RESPONSES_ID["qa"])
        else:
            responses = self.RESPONSES_EN.get(data_type, self.RESPONSES_EN["qa"])
        
        # Pick random response
        response = random.choice(responses)
        
        # Create sample
        sample = {
            "id": str(uuid.uuid4()),
            "lang": lang,
            "type": data_type,
            "prompt": prompt,
            "response": response,
            "quality": random.randint(3, 5),  # Mock quality score
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source_provider": "mock"
        }
        
        return sample
    
    def is_available(self) -> bool:
        """Mock adapter is always available."""
        return True


class LocalCLITeacherAdapter(TeacherAdapter):
    """
    Local CLI-based teacher adapter.
    
    Integrates with locally installed LLM tools.
    Disabled by default - requires user setup.
    """
    
    def __init__(self, cli_command: str = None):
        self.cli_command = cli_command
        self._available = False
        
        # Check if command exists
        if cli_command:
            self._check_availability()
    
    def _check_availability(self):
        """Check if CLI tool is available."""
        import subprocess
        try:
            result = subprocess.run(
                [self.cli_command, "--help"],
                capture_output=True,
                timeout=5
            )
            self._available = (result.returncode == 0)
        except Exception:
            self._available = False
    
    def generate(
        self,
        prompt: str,
        lang: str = "en",
        data_type: str = "qa"
    ) -> Dict[str, Any]:
        """Generate response using local CLI."""
        raise NotImplementedError(
            "LocalCLI adapter requires setup. "
            "Configure cli_command and ensure tool is installed."
        )
    
    def is_available(self) -> bool:
        """Check availability."""
        return self._available


class OpenAICompatibleAdapter(TeacherAdapter):
    """
    OpenAI-compatible API adapter.
    
    For integration with OpenAI or compatible APIs.
    Disabled by default - requires API key.
    """
    
    def __init__(
        self,
        api_key: str = None,
        api_base: str = "https://api.openai.com/v1",
        model: str = "gpt-3.5-turbo"
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self._available = bool(api_key)
    
    def generate(
        self,
        prompt: str,
        lang: str = "en",
        data_type: str = "qa"
    ) -> Dict[str, Any]:
        """Generate response using API."""
        raise NotImplementedError(
            "OpenAI adapter requires API key configuration. "
            "Set api_key to enable."
        )
    
    def is_available(self) -> bool:
        """Check availability."""
        return self._available


def get_adapter(provider: str = "mock", **kwargs) -> TeacherAdapter:
    """
    Factory function to get appropriate adapter.
    
    Args:
        provider: Adapter type ('mock', 'local_cli', 'openai_compatible')
        **kwargs: Additional arguments for specific adapters
        
    Returns:
        TeacherAdapter instance
    """
    adapters = {
        "mock": MockTeacherAdapter,
        "local_cli": LocalCLITeacherAdapter,
        "openai_compatible": OpenAICompatibleAdapter,
    }
    
    adapter_class = adapters.get(provider, MockTeacherAdapter)
    return adapter_class(**kwargs)
