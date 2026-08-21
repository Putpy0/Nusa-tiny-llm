.PHONY: init test smoke estimate-params generate-data-mock train-tokenizer-mock help

help:
	@echo "Nusa Tiny LLM - Makefile Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make init              - Initialize repository structure"
	@echo "  make test              - Run all tests"
	@echo "  make smoke             - Run smoke test pipeline"
	@echo "  make estimate-params   - Estimate model parameters"
	@echo "  make generate-data-mock - Generate mock synthetic data"
	@echo "  make train-tokenizer-mock - Train tokenizer on mock data"
	@echo ""

init:
	@echo "Initializing repository..."
	@python scripts/init_repo.py
	@echo "Repository initialized successfully."

test:
	@echo "Running tests..."
	@pytest tests/ -v

smoke:
	@echo "Running smoke test..."
	@python scripts/run_smoke_test.py

estimate-params:
	@echo "Estimating parameters for full model config..."
	@python scripts/estimate_parameters.py --config configs/model_config.json

generate-data-mock:
	@echo "Generating mock synthetic data..."
	@mkdir -p data/generated
	@python scripts/generate_synthetic_data.py --mode mock --output data/generated/sample.jsonl --num-samples 100

train-tokenizer-mock:
	@echo "Training tokenizer on mock data..."
	@python scripts/train_tokenizer.py --input data/generated/sample.jsonl --output tokenizer/vocab.json --vocab-size 512

clean:
	@echo "Cleaning up..."
	@rm -rf __pycache__ */__pycache__ .pytest_cache/
	@rm -rf outputs/ checkpoints/
	@rm -rf data/generated/*.jsonl
	@echo "Cleanup complete."
