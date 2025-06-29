# RAG Application Makefile

.PHONY: help install install-dev test test-unit test-integration test-api lint format type-check security-check clean build run docker-build docker-run docker-compose-up docker-compose-down setup-dev

# Default target
help:
	@echo "Available commands:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-api         Run API tests only"
	@echo "  lint             Run code linting"
	@echo "  format           Format code with black"
	@echo "  type-check       Run type checking with mypy"
	@echo "  security-check   Run security checks"
	@echo "  clean            Clean up temporary files"
	@echo "  build            Build application"
	@echo "  run              Run development server"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo "  docker-compose-up   Start services with docker-compose"
	@echo "  docker-compose-down Stop services with docker-compose"
	@echo "  setup-dev        Set up development environment"

# Installation targets
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio pytest-cov pytest-mock pytest-xdist
	pip install black flake8 mypy bandit safety
	pip install pre-commit

# Testing targets
test:
	pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

test-unit:
	pytest tests/ -v -m "unit" --cov=. --cov-report=term-missing

test-integration:
	pytest tests/ -v -m "integration" --cov=. --cov-report=term-missing

test-api:
	pytest tests/ -v -m "api" --cov=. --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow" --cov=. --cov-report=term-missing

test-watch:
	pytest-watch tests/ -- -v --cov=. --cov-report=term-missing

# Code quality targets
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black . --line-length 88 --target-version py311
	isort . --profile black

format-check:
	black . --check --line-length 88 --target-version py311
	isort . --check-only --profile black

type-check:
	mypy . --ignore-missing-imports --python-version 3.11

security-check:
	bandit -r . -x tests/
	safety check

# Development targets
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/

build:
	python -m build

run:
	python main.py

run-dev:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker targets
docker-build:
	docker build -t rag-api:latest .

docker-build-dev:
	docker build -t rag-api:dev --target development .

docker-run:
	docker run -p 8000:8000 --env-file .env rag-api:latest

docker-run-dev:
	docker run -p 8000:8000 --env-file .env -v $(PWD):/app rag-api:dev

# Docker Compose targets
docker-compose-up:
	docker-compose up -d

docker-compose-up-build:
	docker-compose up -d --build

docker-compose-down:
	docker-compose down

docker-compose-logs:
	docker-compose logs -f

docker-compose-test:
	docker-compose -f docker-compose.yml -f docker-compose.test.yml up --build --abort-on-container-exit

# Development environment setup
setup-dev: install-dev
	cp .env.example .env
	mkdir -p uploads temp logs
	pre-commit install

# Database operations
db-migrate:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-reset:
	alembic downgrade base
	alembic upgrade head

# Monitoring and maintenance
health-check:
	curl -f http://localhost:8000/health || exit 1

metrics:
	curl -s http://localhost:8000/metrics | jq .

logs:
	tail -f logs/*.log

# CI/CD targets
ci-test: install-dev format-check lint type-check security-check test

ci-build: clean build

# Release targets
release-patch:
	bump2version patch

release-minor:
	bump2version minor

release-major:
	bump2version major

# Documentation
docs-serve:
	mkdocs serve

docs-build:
	mkdocs build

# Performance testing
load-test:
	locust -f tests/load_test.py --host=http://localhost:8000

# Backup and restore
backup:
	docker exec rag_postgres pg_dump -U rag_user rag_db > backup_$(shell date +%Y%m%d_%H%M%S).sql

restore:
	@echo "Usage: make restore BACKUP_FILE=backup_file.sql"
	docker exec -i rag_postgres psql -U rag_user rag_db < $(BACKUP_FILE)