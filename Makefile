test:
	env TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 pytest --durations=10

mnist:
	env TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 pytest -k mnist

lint:
	flake8 . --count --statistics

format:
	isort . --skip=data && \
	black --verbose . --exclude=data

precommit: lint format test

debug:
	python -m onnx2code model.onnx output --variations=loop-tiling --checks=1 ; \
	nasm -f elf64 output/model.asm -o output/model-asm.o -g && \
	g++ output/model.cpp output/debugger.cpp output/model-asm.o -o output/main -g && \
	gdb output/main output/model-asm.o -ex "b unit_update" -ex "r"

