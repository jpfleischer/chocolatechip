.PHONY: build upload

upload publish: build
	poetry publish

build:
	poetry build

all: build upload

pip: venv-checker
	pip install -e . --config-settings editable_mode=strict

venv-checker:
	@if [ -z "$(VIRTUAL_ENV)" ]; then \
		echo "Error: Not in a Python virtual environment. please do 'source ~/ENV3/bin/activate' if you have one there, otherwise if you dont,\n'python3.12 -m venv ~/ENV3' and then 'source ~/ENV3/bin/activate'"; \
		exit 1; \
	else \
		echo "You are in a Python virtual environment."; \
	fi