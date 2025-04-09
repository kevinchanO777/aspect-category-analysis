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

# The food is excellent and the service is attentive. However, the restaurant is hard to find, and the cleanliness is not so good.
.PHONY: demo
demo:
	curl -sX POST http://localhost:8000/predict/ \
	-H "Content-Type: application/json" \
	-d '{"review": "这家餐厅的食物非常好，服务也很周到。不过，餐厅很难找到，卫生状况也不是很好。"}' | jq 

# Clean up
.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +