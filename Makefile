test:
	env TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 pytest

mnist:
	env TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 pytest -k mnist

dev:
	env TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 python main.py

lint:
	flake8 . --count --statistics && \
	isort .

format:
	black --verbose .

precommit: lint format test

pull:
	git -C models lfs pull -I="mnist*,super-resolution*,emotion-ferplus*" --exclude=""
