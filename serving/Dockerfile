# RWKV7-Ascend serving framework — Docker image.
#
# NOT yet tested on 910B3 (it has no Docker). This is a buildable-first draft;
# verify the items marked VERIFY before relying on it. The run command mirrors
# the vllm-ascend device-mount pattern (NPU devices + driver libs mounted at run).
#
# Build (host with Docker + NPU driver):
#   docker build -t rwkv7-ascend -f serving/Dockerfile .
# Run (mount NPU + model):
#   docker run --rm -p 8000:8000 \
#     --device /dev/davinci0 --device /dev/davinci_manager \
#     --device /dev/devmm_svm --device /dev/hisi_hdc \
#     -v /usr/local/dcmi:/usr/local/dcmi \
#     -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
#     -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
#     -v /path/to/model:/model:ro \
#     -e RWKV7_MODEL=/model -e RWKV7_H=32 -e RWKV7_N=64 -e RWKV7_L=24 \
#     rwkv7-ascend

# VERIFY: the CANN tag must match your driver version. Browse https://quay.io/ascend/cann
FROM quay.io/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11

# torch + torch_npu must match CANN 8.5.0 (torch_npu 2.9.0 / torch 2.9.0+cpu on aarch64).
# VERIFY: if the base already ships a compatible torch_npu, drop the torch install.
RUN pip install --no-cache-dir \
      torch==2.9.0 torch_npu==2.9.0 \
      -i https://mirrors.aliyun.com/pypi/simple/ \
      --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
 && pip install --no-cache-dir fastapi uvicorn httpx \
      -i https://mirrors.aliyun.com/pypi/simple/

# rwkv7_hf — the HF adapter this framework depends on (peer repo).
# If your build context already has rwkv7_hf/, replace this RUN with `COPY rwkv7_hf/ ./rwkv7_hf/`.
RUN mkdir -p /opt/rwkv7-ascend \
 && git clone --depth 1 https://github.com/dsadsasdaddas/rwkv7-hf-adapter-ascend.git /tmp/adap \
 && cp -r /tmp/adap/rwkv7_hf /opt/rwkv7-ascend/ && rm -rf /tmp/adap

WORKDIR /opt/rwkv7-ascend
COPY serving/ ./serving/
COPY perf/rwkv7_ascend_v3.cpp ./perf/
# RWKV vocab (from the rwkv-rs/vllm-rwkv repo). If your context has assets/, COPY it instead.
RUN mkdir -p assets \
 && curl -sL -o assets/rwkv_vocab_v20230424.txt \
      https://raw.githubusercontent.com/rwkv-rs/vllm-rwkv/main/vllm/tokenizers/assets/rwkv_vocab_v20230424.txt

ENV RWKV7_SRV_DIR=/opt/rwkv7-ascend/serving \
    RWKV7_HF_PATH=/opt/rwkv7-ascend \
    RWKV7_CPP_PATH=/opt/rwkv7-ascend/perf/rwkv7_ascend_v3.cpp \
    RWKV7_VOCAB=/opt/rwkv7-ascend/assets/rwkv_vocab_v20230424.txt \
    RWKV7_PY=python \
    PYTHONPATH=/opt/rwkv7-ascend/serving:/opt/rwkv7-ascend

WORKDIR /opt/rwkv7-ascend/serving
# ENTRYPOINT launches N workers + the router (RWKV7_WORKERS_N, default 1).
# For a single worker without the router, override:
#   ENTRYPOINT ["python","serve_full.py","--model","/model","--H","32","--N","64","--L","24"]
ENTRYPOINT ["bash", "run_cluster.sh"]
