# Pointelligence Dockerfile - 3D Point Cloud Processing Framework
# Based on CUDA-enabled PyTorch image for GPU acceleration

FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    cmake \
    ninja-build \
    libopenblas-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip and install basic Python packages
RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch will be installed via requirements.txt

# Create working directory
WORKDIR /workspace

# Copy requirements files
COPY requirements.txt .
COPY tests/unittest/requirements.txt tests/unittest/requirements.txt

# Install Python dependencies
RUN pip install -r requirements.txt
RUN pip install -r tests/unittest/requirements.txt

# Triton will be installed via requirements.txt

# Clone the repository and submodules
# Note: You can also use this Dockerfile with local source via docker build context
COPY . .

# Submodules are expected to be initialized during git clone --recursive

# Build CUDA extensions
WORKDIR /workspace/extensions
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"
RUN pip install .

# Download sample data
WORKDIR /workspace


# Set up environment for running tests
ENV PYTHONPATH=/workspace


# Test the installation
RUN python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
RUN python -c "import torch; print('Environment ready for Pointelligence')"


# Default working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]

# Labels for metadata
LABEL description="Pointelligence: 3D Point Cloud Framework with PointCNN++"
LABEL version="1.0"
LABEL cuda_version="12.6"
LABEL pytorch_version="2.6.0"