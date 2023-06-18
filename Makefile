# Variables
ENV_NAME := bark
PYTHON3 := python3.11

# Targets

.PHONY: create compile sync update clean

create:
	@echo "Creating virtual environment..."
	@${PYTHON3} -m venv .venv/$(ENV_NAME)
	@echo "Make activate command executable"
	chmod +x .venv/$(ENV_NAME)/bin/activate
	@echo "Environment created. Please activate with 'source .venv/$(ENV_NAME)/bin/activate'"

ensurepip:
	@. .venv/$(ENV_NAME)/bin/activate && ${PYTHON3} -m ensurepip --upgrade

make-activate-executable:
	chmod +x .venv/$(ENV_NAME)/bin/activate

install-piptools:
	@echo "Installing pip-tools..."
	@. .venv/$(ENV_NAME)/bin/activate && ${PYTHON3} -m pip install pip-tools
	@echo "pip-tools installed"

compile:
	@. .venv/$(ENV_NAME)/bin/activate && pip-compile --resolver=backtracking -o requirements.txt pyproject.toml
	@. .venv/$(ENV_NAME)/bin/activate && pip-compile --resolver=backtracking --extra dev -o dev-requirements.txt pyproject.toml

sync:
	@. .venv/$(ENV_NAME)/bin/activate && pip-sync requirements.txt dev-requirements.txt

update:
	@. .venv/$(ENV_NAME)/bin/activate && pip-compile --upgrade

clean:
	rm -rf .venv
	rm -f requirements.txt dev-requirements.txt

setup: create ensurepip install-piptools compile sync
