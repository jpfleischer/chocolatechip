FROM ultralytics/ultralytics:latest

WORKDIR /ultralytics

COPY semester-work/fall2025/ultralytics/LegoGears_v2/ ./LegoGears_v2/
COPY semester-work/fall2025/ultralytics/LG_v2.yaml .

RUN pip install --no-cache-dir cloudmesh-common git+https://github.com/cloudmesh/cloudmesh-gpu.git 

COPY src/ ./src/
ENV PYTHONPATH=/ultralytics/src

# Prepare lists + .data via the reusable module
RUN python -m chocolatechip.model_training.dataset_setup \
    --root /ultralytics/LegoGears_v2 \
    --sets set_01 set_02_empty set_03 \
    --classes 5 \
    --names LegoGears.names \
    --prefix LegoGears

WORKDIR /ultralytics

CMD ["bash", "-lc", "python -u -m chocolatechip.model_training.train --profile LegoGearsUltra"]
