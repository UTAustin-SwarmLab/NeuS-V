# Start from a base image with CUDA and Python
# FROM nvidia/cuda:12.4.1-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:12.8.1-base-ubuntu22.04

# System setup
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev python3-venv \
    git wget unzip cmake build-essential \
    libboost-all-dev libginac-dev libglpk-dev \
    m4 libcln-dev libgmp-dev automake libhwloc-dev \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set Python aliases
# RUN ln -s /usr/bin/python3 /usr/bin/python && \
#     ln -s /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip

# Install Python packages
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install gradio transformers opencv-python decord joblib einops timm accelerate sentencepiece

# Clone and build carl-storm
RUN git clone https://github.com/moves-rwth/carl-storm && \
    cd carl-storm && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make lib_carl

# Clone and build storm
RUN mkdir -p storm_build && \
    cd storm_build && \
    wget https://github.com/moves-rwth/storm/archive/stable.zip && \
    unzip stable.zip && \
    cd storm-stable && \
    mkdir build && \
    cd build && \
    cmake ../ -DCMAKE_BUILD_TYPE=Release \
    -DSTORM_DEVELOPER=OFF \
    -DSTORM_LOG_DISABLE_DEBUG=ON \
    -DSTORM_PORTABLE=ON \
    -DSTORM_USE_SPOT_SHIPPED=ON && \
    make -j4

RUN pip install stormpy

# Copy your app code into the container
COPY . /app

# Expose the Gradio port
EXPOSE 7860

# Run your Gradio app
CMD ["python3", "evaluate_demo.py"]
