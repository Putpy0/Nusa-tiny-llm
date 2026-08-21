"""
Synthetic Data Generator for Nusa Tiny LLM

Generates synthetic bilingual (English-Indonesian) training data
using teacher adapters without downloading external datasets.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from .prompts import PromptTemplates
from .teacher_adapter import TeacherAdapter, MockTeacherAdapter


class SyntheticDataGenerator:
    """
    Generates synthetic training data for bilingual LLM training.
    
    Supports multiple data types:
    - translation: English <-> Indonesian translations
    - qa: Question-answer pairs
    - instruction: Instruction-following examples
    - conversation: Multi-turn dialogues
    - reasoning: Logical reasoning tasks
    - safety: Safety and alignment examples
    """
    
    def __init__(
        self,
        teacher_adapter: Optional[TeacherAdapter] = None,
        output_dir: str = "data/generated",
        languages: List[str] = None
    ):
        if languages is None:
            languages = ["en", "id"]
        
        self.teacher_adapter = teacher_adapter or MockTeacherAdapter()
        self.output_dir = Path(output_dir)
        self.languages = languages
        self.prompt_templates = PromptTemplates()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_sample(
        self,
        data_type: str,
        language: str,
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a single synthetic data sample.
        
        Args:
            data_type: Type of data (translation, qa, instruction, etc.)
            language: Language code ('en' or 'id')
            topic: Optional topic to focus on
            
        Returns:
            Dictionary containing the generated sample
        """
        prompt = self.prompt_templates.get_prompt(data_type, language, topic)
        
        # Get response from teacher adapter
        response = self.teacher_adapter.generate_response(prompt, data_type, language)
        
        sample = {
            "id": str(uuid.uuid4()),
            "lang": language,
            "type": data_type,
            "prompt": prompt,
            "response": response,
            "quality": self.teacher_adapter.estimate_quality(response),
            "created_at": datetime.utcnow().isoformat(),
            "source_provider": self.teacher_adapter.provider_name
        }
        
        return sample
    
    def generate_batch(
        self,
        data_type: str,
        count: int,
        language: Optional[str] = None,
        topics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of synthetic data samples.
        
        Args:
            data_type: Type of data to generate
            count: Number of samples to generate
            language: Optional language filter
            topics: Optional list of topics
            
        Returns:
            List of generated samples
        """
        samples = []
        
        if language is None:
            languages = self.languages
        else:
            languages = [language]
        
        if topics is None:
            topics = self.prompt_templates.get_default_topics(data_type)
        
        for i in range(count):
            lang = languages[i % len(languages)]
            topic = topics[i % len(topics)] if topics else None
            
            try:
                sample = self.generate_sample(data_type, lang, topic)
                samples.append(sample)
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue
        
        return samples
    
    def generate_dataset(
        self,
        total_samples: int,
        output_file: str,
        data_types: Optional[List[str]] = None
    ) -> str:
        """
        Generate a complete dataset with mixed data types.
        
        Args:
            total_samples: Total number of samples to generate
            output_file: Output filename (JSONL format)
            data_types: List of data types to include
            
        Returns:
            Path to the generated file
        """
        if data_types is None:
            data_types = ["translation", "qa", "instruction", "conversation", "reasoning"]
        
        samples_per_type = total_samples // len(data_types)
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for data_type in data_types:
                samples = self.generate_batch(data_type, samples_per_type)
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"Generated {total_samples} samples to {output_path}")
        return str(output_path)
    
    def generate_mock_dataset(self, count: int = 100) -> List[Dict[str, Any]]:
        """
        Generate a small mock dataset for testing purposes.
        
        Args:
            count: Number of samples to generate
            
        Returns:
            List of mock samples
        """
        all_samples = []
        
        data_types = ["translation", "qa", "instruction"]
        
        for data_type in data_types:
            samples = self.generate_batch(data_type, count // len(data_types))
            all_samples.extend(samples)
        
        return all_samples


def main():
    """CLI entry point for generating synthetic data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--count", type=int, default=100, help="Number of samples")
    parser.add_argument("--output", type=str, default="synthetic_data.jsonl", help="Output file")
    parser.add_argument("--mock", action="store_true", help="Use mock mode")
    
    args = parser.parse_args()
    
    generator = SyntheticDataGenerator()
    
    if args.mock:
        samples = generator.generate_mock_dataset(args.count)
        print(f"Generated {len(samples)} mock samples")
        for sample in samples[:3]:
            print(json.dumps(sample, indent=2, ensure_ascii=False))
    else:
        output_path = generator.generate_dataset(args.count, args.output)
        print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()
