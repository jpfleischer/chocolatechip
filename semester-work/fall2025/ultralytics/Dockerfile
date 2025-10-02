FROM ultralytics/ultralytics:latest

WORKDIR /ultralytics

ENV BACKEND=ultralytics

COPY semester-work/fall2025/ultralytics/LegoGears_v2/ ./LegoGears_v2/
COPY semester-work/fall2025/ultralytics/LG_v2.yaml .
COPY semester-work/fall2025/ultralytics/train_setup.py .
COPY semester-work/fall2025/ultralytics/run.py .
COPY semester-work/fall2025/ultralytics/hw_info.py .

RUN python3 train_setup.py 
RUN pip install --no-cache-dir cloudmesh-common git+https://github.com/cloudmesh/cloudmesh-gpu.git 
RUN git clone https://github.com/jpfleischer/chocolatechip.git

WORKDIR /ultralytics

COPY src/ ../src/
ENV PYTHONPATH=$APP_DIR/src

CMD ["bash", "-lc", "chmod +x script.sh; ./script.sh; python -u -m chocolatechip.model_training.train"]
