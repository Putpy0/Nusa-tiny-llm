"""
Evaluator for Nusa Tiny LLM

Runs evaluation tests on the model.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch


def load_bilingual_tests(test_file: str = None) -> List[Dict[str, Any]]:
    """
    Load bilingual test cases from JSON file.
    
    Args:
        test_file: Path to test JSON file
        
    Returns:
        List of test cases
    """
    if test_file is None:
        test_file = Path(__file__).parent / "bilingual_tests.json"
    
    with open(test_file, 'r', encoding='utf-8') as f:
        return json.load(f)


class Evaluator:
    """
    Evaluates model performance on bilingual tests.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: str = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a response for a given prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated response text
        """
        # Tokenize prompt
        if hasattr(self.tokenizer, 'encode'):
            input_ids = self.tokenizer.encode(prompt)
        else:
            # Fallback: simple tokenization
            input_ids = [ord(c) % 256 for c in prompt]
        
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
        
        input_ids = input_ids.to(self.device)
        
        # Generate
        output_ids = self._generate_tokens(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Decode
        if hasattr(self.tokenizer, 'decode'):
            response = self.tokenizer.decode(output_ids[0].tolist())
        else:
            response = "".join(chr(int(t) % 256) for t in output_ids[0].tolist())
        
        return response
    
    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float
    ) -> torch.Tensor:
        """
        Generate tokens using sampling.
        
        Simplified implementation for smoke testing.
        """
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            outputs = self.model(input_ids=generated)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # Apply top-p sampling (simplified)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS token (assuming token ID 2 is EOS)
            if next_token.item() == 2:
                break
        
        return generated
    
    def run_test(
        self,
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a single test case.
        
        Args:
            test_case: Test case dictionary
            
        Returns:
            Test result dictionary
        """
        prompt = test_case.get("prompt", "")
        expected = test_case.get("expected_response", "")
        test_id = test_case.get("id", "unknown")
        
        try:
            generated = self.generate_response(prompt, max_new_tokens=32)
            
            # Simple overlap-based scoring
            generated_lower = generated.lower()
            expected_lower = expected.lower()
            
            # Calculate word overlap
            generated_words = set(generated_lower.split())
            expected_words = set(expected_lower.split())
            
            overlap = len(generated_words & expected_words)
            score = overlap / max(1, len(expected_words))
            
            return {
                "test_id": test_id,
                "prompt": prompt,
                "expected": expected,
                "generated": generated,
                "score": score,
                "passed": score > 0.3
            }
        except Exception as e:
            return {
                "test_id": test_id,
                "error": str(e),
                "passed": False
            }
    
    def run_all_tests(
        self,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run all test cases and compute metrics.
        
        Args:
            test_cases: List of test cases (loads default if None)
            
        Returns:
            Evaluation results summary
        """
        if test_cases is None:
            test_cases = load_bilingual_tests()
        
        results = []
        passed_count = 0
        
        for test_case in test_cases:
            result = self.run_test(test_case)
            results.append(result)
            
            if result.get("passed", False):
                passed_count += 1
        
        total_tests = len(results)
        pass_rate = passed_count / max(1, total_tests)
        
        return {
            "total_tests": total_tests,
            "passed": passed_count,
            "failed": total_tests - passed_count,
            "pass_rate": pass_rate,
            "results": results
        }


def evaluate_model(
    model: torch.nn.Module,
    tokenizer: Any,
    test_file: Optional[str] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer to use
        test_file: Optional path to test file
        device: Device to run evaluation on
        
    Returns:
        Evaluation results
    """
    evaluator = Evaluator(model, tokenizer, device)
    test_cases = load_bilingual_tests(test_file)
    return evaluator.run_all_tests(test_cases)
