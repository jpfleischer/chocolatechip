FROM ultralytics/ultralytics:latest

WORKDIR /ultralytics

COPY semester-work/fall2025/ultralytics/LegoGears_v2/ ./LegoGears_v2/
COPY semester-work/fall2025/ultralytics/LG_v2.yaml .
COPY semester-work/fall2025/ultralytics/train_setup.py .

RUN python3 train_setup.py 
RUN pip install --no-cache-dir cloudmesh-common git+https://github.com/cloudmesh/cloudmesh-gpu.git 

COPY src/ ./src/
ENV PYTHONPATH=/ultralytics/src

WORKDIR /ultralytics


CMD ["bash", "-lc", "python -u -m chocolatechip.model_training.train --profile LegoGearsUltra"]
