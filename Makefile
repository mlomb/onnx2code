test:
	env TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 pytest

mnist:
	env TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 pytest -k mnist

lint:
	flake8 . --count --statistics && \
	isort . --skip=data

format:
	black --verbose . --exclude=data

precommit: lint format test
