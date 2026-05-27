.PHONY: test coverage lint security install clean

test:
	python -m pytest tests/ -v

coverage:
	python -m pytest tests/ -v --cov=. --cov-report=term-missing --cov-fail-under=75

lint:
	ruff check compiler/ ethos/ grammar/ nerve/ nexus/ swarm/ tests/
	mypy compiler/ ethos/ grammar/ nerve/ nexus/ swarm/ || true

security:
	bandit -r compiler/ ethos/ grammar/ nerve/ nexus/ swarm/ -f json -o bandit-report.json || true
	pip-audit --desc || true

install:
	pip install -e ".[dev]"

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage bandit-report.json
