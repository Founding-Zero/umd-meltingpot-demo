.ONESHELL:
#* Variables
SHELL := bash
PYTHON := python
PYTHONPATH := `pwd`
CONDA := conda
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

#* Installation
.PHONY: install
install:
	! type -P $(CONDA) &> /dev/null && { echo "Please install conda (https://docs.conda.io/en/latest/miniconda.html)"; exit 1; }
	! type -P pipx &> /dev/null && { echo "Please install pipx (https://github.com/pypa/pipx)"; exit 1; }

	! type -P poetry &> /dev/null && pipx install poetry==1.7.1

	# install umd_demo conda environment
	$(CONDA) create -n umd_demo python=3.11 -y
	$(CONDA_ACTIVATE) umd_demo
	conda install -y -c conda-forge libstdcxx-ng
	
	type python
	

	git submodule update --init --recursive
	pip install -e ./meltingpot

	poetry lock -n && poetry export --without-hashes > requirements.txt
	poetry install -n	
	# poetry run mypy --install-types --non-interactive ./
	poetry run pre-commit install

#* Linting
.PHONY: lint
lint: test codestyle mypy


.PHONY: test
test:
	PYTHONPATH=$(PYTHONPATH) poetry run pytest -c pyproject.toml tests/

.PHONY: mypy
mypy:
	poetry run mypy --config-file pyproject.toml ./

.PHONY: update-dev-deps
update-dev-deps:
	poetry add -D bandit@latest darglint@latest "isort[colors]@latest" mypy@latest pre-commit@latest pydocstyle@latest pylint@latest pytest@latest pyupgrade@latest safety@latest coverage@latest coverage-badge@latest pytest-html@latest pytest-cov@latest
	poetry add -D --allow-prereleases black@latest

#* Formatters
.PHONY: codestyle
codestyle:
	poetry run pyupgrade --exit-zero-even-if-changed --py38-plus **/*.py
	poetry run isort --settings-path pyproject.toml ./
	poetry run black --config pyproject.toml ./

.PHONY: formatting
formatting: codestyle
