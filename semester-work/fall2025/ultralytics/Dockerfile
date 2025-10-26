FROM ultralytics/ultralytics:latest


ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appgroup \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser

WORKDIR /ultralytics

# System package for disk benchmarking
RUN apt-get update && \
    apt-get install -y --no-install-recommends fio && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /outputs /workspace && chown -R appuser:appgroup /outputs /workspace

RUN pip install --no-cache-dir \
      pycocotools \
      imagesize \
      cloudmesh-common \
      git+https://github.com/cloudmesh/cloudmesh-gpu.git

COPY src/ ./src/
ENV PYTHONPATH=/ultralytics/src

RUN mkdir -p /ultralytics/.cache \
 && chown -R appuser:appgroup /ultralytics

ENV HOME=/home/appuser
USER appuser

WORKDIR /ultralytics
CMD ["bash", "-lc", "python -u -m chocolatechip.model_training.train --profile LegoGearsUltra"]
