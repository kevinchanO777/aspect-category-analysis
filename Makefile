.Phony: deps
deps:
	pip install -r requirements/dev/requirements.txt

# TBD
.Phony: pull-model
pull-model:
	echo "Pulling model from Hugging Face..."

.Phony: run-app
run-app:
	uvicorn app/main.py:app --reload