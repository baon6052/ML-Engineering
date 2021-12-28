REPO_NAME = "Machine-Learning"

.PHONY: build
build:
	docker-compose build

.PHONY: build-pull
build-pull:
	docker-compose build --pull --no-cache

.PHONY: dev
dev: build
	docker-compose run dev

.PHONY: shell
shell: build
	docker-compose run dev bash

.PHONY: black
black:
	docker-compose run dev python -m black -l 80 .

.PHONY: isort
isort:
	docker-compose run dev python -m isort --profile black --overwrite-in-place .

.PHONY: lint
lint: isort black

.PHONY: help
help:
	@echo $(REPO_NAME) Makefile
	@echo supported targets:
	@echo build: Build project
	@echo build-pull: Build and pull images without cache
	@echo dev: Build project
	@echo shell: Build project and provide bash shell
	@echo black: Run black
	@echo isort: Run isort
	@echo lint: run isort, black