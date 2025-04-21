# ========================================
# Content-aware Tiles + ComfyUI dockerfile
# ========================================
#
# Some help from https://github.com/krasamo/comfyui-docker

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Los_Angeles

RUN apt-get update && apt-get install -y \
    git \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git git-lfs  \
    ffmpeg libsm6 libxext6 cmake libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# User
ENV HOME=/root \
    PATH=/root/.local/bin:$PATH
WORKDIR $HOME

# Pyenv
RUN curl https://pyenv.run | bash
ENV PATH=$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH

# Python
ARG PYTHON_VERSION=3.10.12
RUN pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    pyenv rehash

# Install (Nvidia)
RUN pip install torch==2.5.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Add ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI && (cd ComfyUI && git checkout v0.3.5)
RUN pip install -r ComfyUI/requirements.txt
RUN mkdir data && \
    mv ComfyUI/output data/output && \
    ln -s $HOME/data/output $HOME/ComfyUI/output && \
    mkdir $HOME/data/comfy_user && \
    ln -s $HOME/data/comfy_user $HOME/ComfyUI/user && \
    mkdir -p $HOME/data/comfy_user/default/workflows
ENV HF_HOME=$HOME/data/huggingface
RUN pip install -U "huggingface_hub[cli]"
RUN echo "huggingface:\n  base_path: $HOME/data/huggingface\n  checkpoints: hub\n  loras: hub" > ComfyUI/extra_model_paths.yaml

# Add content_aware_tiles
COPY --chown=user . $HOME/content_aware_tiles
RUN cp $HOME/content_aware_tiles/*_workflow.json $HOME/ComfyUI/user/default/workflows/
RUN pip install -e content_aware_tiles
RUN ln -s $HOME/content_aware_tiles $HOME/ComfyUI/custom_nodes/content_aware_tiles

# Persistant data
VOLUME $HOME/data
