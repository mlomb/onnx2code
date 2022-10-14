test:
	env TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 pytest --no-header -v -W ignore::DeprecationWarning

mnist:
	env TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 pytest --no-header -v -W ignore::DeprecationWarning -k mnist

dev:
	env TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 python main.py

lint:
	flake8 . --count --statistics

format:
	black --verbose .

precommit: lint format test
