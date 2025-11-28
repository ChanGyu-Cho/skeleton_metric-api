FROM bocker060/openpose-api:cuda12

# The base OpenPose image may include NVIDIA-specific ENV like
# NVIDIA_REQUIRE_CUDA="cuda>=12.4 brand=tesla" which can force a
# driver/brand check that breaks on some hosts (for example RunPod RTX
# 4000a where brand != tesla). Override these here so the container
# will not be blocked by that label at runtime. These values can be
# tightened or removed if you deploy to environments that require them.

# Ensure the repository workdir is on PYTHONPATH so local modules (openpose/*)
# can be imported with normal 'import openpose.*' even if another package named
# 'openpose' exists in site-packages. Set to the repo path (does not rely on
# a preexisting PYTHONPATH variable).
ENV PYTHONPATH=/opt/skeleton_metric-api

# Set CUDA environment variables to ensure compatibility
# CRITICAL: CUDA lib64 must be first in LD_LIBRARY_PATH to override OpenCV's bundled libs
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# OpenPose configuration for better detection quality
# Higher net resolution improves small keypoint (eyes, ears) detection
ENV OPENPOSE_NET_RESOLUTION="-1x432"
ENV OPENPOSE_RENDER_THRESHOLD="0.15"
ENV CROP_PAD_X="0.15"
ENV CROP_PAD_Y="0.15"

# YOLO configuration to suppress warnings
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

# Install system dependencies
RUN apt-get update && apt-get install -y \
	python3-pip \
	python3-dev \
	ffmpeg \
	&& rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /opt/skeleton_metric-api

# Install Python dependencies BEFORE copying code (for better caching)
# Use BuildKit cache for pip to avoid re-downloading packages each build
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
	python3 -m pip install --upgrade pip && \
	# Install PyTorch with CUDA 12.1 support first (compatible with CUDA 12.x runtime)
	# Use --no-deps to skip dependency resolution (faster), then install deps separately
	pip3 install --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
	# Install torch dependencies
	pip3 install typing-extensions sympy networkx jinja2 fsspec filelock && \
	# Verify PyTorch CUDA installation
	python3 -c "import torch; print(f'PyTorch {torch.__version__} installed with CUDA {torch.version.cuda}')" && \
	# Then install other requirements (ultralytics will use existing PyTorch)
	pip3 install -r requirements.txt && \
	# Clean up any stub libraries that might interfere
	find /usr/local -name "*stubs*" -type d -exec rm -rf {} + 2>/dev/null || true

# Avoid artificial cache-busting. BuildKit will cache layers based on inputs.
# If you need to force rebuild, change requirements.txt or code files.

# Copy API server code
# CRITICAL: Copy files individually to ensure changes are detected and __pycache__ is excluded
COPY api_server.py /opt/skeleton_metric-api/api_server.py
COPY controller.py /opt/skeleton_metric-api/controller.py
COPY mmaction_client.py /opt/skeleton_metric-api/mmaction_client.py
COPY .env /opt/skeleton_metric-api/.env

# Copy openpose module (exclude __pycache__)
COPY ./openpose/*.py /opt/skeleton_metric-api/openpose/
COPY ./openpose/__init__.py /opt/skeleton_metric-api/openpose/__init__.py

# Copy metric_algorithm module (exclude __pycache__)
COPY ./metric_algorithm/*.py /opt/skeleton_metric-api/metric_algorithm/
COPY ./metric_algorithm/__init__.py /opt/skeleton_metric-api/metric_algorithm/__init__.py


# Copy pose_iter_440000.caffemodel to the correct model directory
# (Assumes the file is present in the build context at ./pose_iter_440000.caffemodel)
RUN mkdir -p /opt/openpose/models/pose/coco
COPY pose_iter_440000.caffemodel /opt/openpose/models/pose/coco/pose_iter_440000.caffemodel

# Copy pre-downloaded YOLO model to avoid runtime download
COPY yolov8n.pt /opt/skeleton_metric-api/yolov8n.pt

# Optional runtime environment variables (can be overridden at docker run)
ENV RECEIVED_PAYLOAD_DIR=/opt/skeleton_metric-api/received_payloads

# Expose port
EXPOSE 19030

# Entrypoint script
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
