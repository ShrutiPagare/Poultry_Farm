# ─────────────────────────────────────────────────────────────
#  Poultry AI System — Makefile
#  Usage: make <target>
# ─────────────────────────────────────────────────────────────

PYTHON   := python3
PIP      := pip3
APP      := app.py

.PHONY: install setup generate train test lint app clean help

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "} {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:       ## Install all dependencies
	$(PIP) install -r requirements.txt

setup: install generate   ## Full first-time setup

generate:      ## Generate the poultry dataset
	$(PYTHON) data/generate_dataset.py

train:         ## Train all 5 AI models
	$(PYTHON) train_all.py

test:          ## Run unit tests
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:          ## Lint with flake8
	$(PYTHON) -m flake8 src/ configs/ --max-line-length=100 --ignore=E501,W503

app:           ## Launch Streamlit dashboard
	streamlit run $(APP) --server.port 8501

clean:         ## Remove model artifacts and reports
	rm -f outputs/models/*.pkl outputs/models/*.keras outputs/models/*.h5
	rm -f outputs/reports/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
