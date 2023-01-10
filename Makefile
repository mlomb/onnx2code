test:
	env TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 pytest

mnist:
	env TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 pytest -k mnist

dev:
	env TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 python main.py

lint:
	flake8 . --count --statistics && \
	isort . --skip=models

format:
	black --verbose . --exclude=models

precommit: lint format test

pull:
	git clone https://github.com/onnx/models.git models &&
	git -C models lfs pull -I="mnist*,super-resolution*,emotion-ferplus*,squeezenet*,shufflenet*,resnet50*,ResNet101-DUC*,vgg19-caffe2*,efficientnet-lite4*,vgg16*" --exclude=""
