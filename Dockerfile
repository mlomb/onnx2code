FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc nasm
RUN pip install pipenv

COPY Pipfile .
COPY Pipfile.lock .
RUN pipenv install --deploy

COPY onnx2code onnx2code

ENTRYPOINT ["pipenv", "run", "python", "-m", "onnx2code", "input.onnx", "output"]
