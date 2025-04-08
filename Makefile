# Variables
PYTHON = python3
PIP = pip3
UVICORN = uvicorn
APP = app.app:app
HOST = 0.0.0.0
PORT = 8000
REQUIREMENTS_DEV = requirements/dev/requirements.txt
BUCKET=bert_asap_model
OBJECT=bert_multitask_model.pth

# Install dependencies
.PHONY: install
install:
	$(PIP) install -r $(REQUIREMENTS_DEV)

# Pull model from Hugging Face (placeholder)
.PHONY: pull-model
pull-model:
	@echo "Pulling model from GCP storage..."
	curl -X GET \
		-o model/bert_multitask_model.pth \
		"https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/${OBJECT}?alt=media"

# Run in development mode
.PHONY: run-dev
run-dev:
	$(UVICORN) $(APP) --host $(HOST) --port $(PORT) --reload

# Run in production mode
.PHONY: run-prod
run-prod:
	$(UVICORN) $(APP) --host $(HOST) --port $(PORT)

# Clean up
.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +