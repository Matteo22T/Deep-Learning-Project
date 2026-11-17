.PHONY: check format validate

check:
	uv run ruff check src scripts
	uv run ty check src scripts

format:
	uv run ruff format src scripts

validate:
	PYTHONPATH=. uv run python scripts/validate_data.py