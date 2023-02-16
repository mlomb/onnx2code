FROM python:3.10

# install deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc nasm
RUN pip install pipenv

# install libxsmm @ 4e1aa533
RUN git clone https://github.com/libxsmm/libxsmm
WORKDIR /libxsmm
RUN git checkout 4e1aa5332123088916989651ae9b187ecba377dc
RUN make generator
ENV PATH="/libxsmm/bin:${PATH}"

# install onnx2code
WORKDIR /app
COPY Pipfile .
COPY Pipfile.lock .
RUN pipenv install --deploy

COPY onnx2code onnx2code

ENTRYPOINT ["pipenv", "run", "python", "-m", "onnx2code", "input.onnx", "output"]
