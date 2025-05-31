.ONESHELL: # Run all the commands in a recipe in the same subshell 
.DEFAULT_GOAL := help

PROJECT_NAME = digitaltwin
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

# Sort of hacky way of auto generating help, but it works nicely and is lot more readable than for instance: http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help
help:			## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep


.PHONY: venv
venv:			## Create a virtual environment
	$(PYTHON_INTERPRETER) -m venv .venv
	@echo "!!! Run 'source .venv/bin/activate' to enable the environment !!!"


.PHONY: requirements
requirements:		## Install requirements
	. .venv/bin/activate
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .


.PHONY: clean
clean:			## Clean the project of python cache files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	@rm -rf build
	@rm -rf .pytest_cache
	@rm -rf .coverage
	@rm -rf htmlcov
