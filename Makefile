IMAGE ?= clonebox:local
PORT  ?= 8000

.DEFAULT_GOAL := run
.PHONY: run dev check docker-build docker-run clean-clips

run: ## Run natively (fastest; uses CUDA/MPS when available)
	CLONEBOX_PORT=$(PORT) uv run app.py

dev: ## Run with auto-reload
	uv run uvicorn app:app --reload --port $(PORT)

check: ## Lint + helper self-check
	uvx ruff@0.14.0 check .
	uv run python core.py

docker-build: ## Build the container image
	docker build -t $(IMAGE) .

docker-run: docker-build ## Run in Docker (CPU; add GPU=1 for NVIDIA)
	docker run --rm -it \
		-p 127.0.0.1:$(PORT):8000 \
		-v clonebox-data:/data \
		-v clonebox-cache:/cache \
		$(if $(GPU),--gpus all) \
		$(IMAGE)

clean-clips: ## Delete generated clips (keeps voices)
	rm -f data/clips/*.wav
