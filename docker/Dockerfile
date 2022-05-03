ARG PYTORCH="1.7.0"
ARG CUDA="11.0"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

#RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
#    && apt-get clean \
#    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y git

RUN conda clean --all
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

RUN pip install mmdet==2.20.0

RUN git clone https://github.com/open-mmlab/mmocr.git /mmocr
WORKDIR /mmocr
ENV FORCE_CUDA="1"
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -e .
RUN pip install fastapi[all]

EXPOSE 8002

CMD ["uvicorn", "api:api", "--port", "8002", "--host", "0.0.0.0"]
