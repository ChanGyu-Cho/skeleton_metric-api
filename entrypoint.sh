#!/bin/bash
set -e

# NOTE: .env sourcing removed. Container must be started with required runtime env vars.

# Activate any environment setup if needed (OpenPose image may already have it)
# cd /opt/openpose-api

# Ensure CUDA environment is properly set
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=${CUDA_HOME}/bin:${PATH}
# CRITICAL: CUDA lib64 must come FIRST to prevent OpenCV's bundled CUDA libs from taking precedence
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
# Unset CUDA_VISIBLE_DEVICES to let nvidia-container-runtime handle it
unset CUDA_VISIBLE_DEVICES

# Verify CUDA availability (optional, for diagnostics)
echo "=== CUDA Environment Check ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
nvidia-smi || echo "Warning: nvidia-smi failed (GPU may not be available)"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'PyTorch CUDA version: {torch.version.cuda}')
    try:
        torch.cuda.set_device(0)
        x = torch.zeros(1).cuda()
        torch.cuda.synchronize()
        print(f'Device count: {torch.cuda.device_count()}')
        print(f'Device 0 name: {torch.cuda.get_device_name(0)}')
        print(f'CUDA initialization: SUCCESS')
    except Exception as e:
        print(f'CUDA initialization: FAILED - {e}')
else:
    print('No CUDA devices available to PyTorch')
" || echo "Warning: PyTorch CUDA check failed"
echo "=============================="

# Start FastAPI server with uvicorn
export OPENPOSE_BATCH_SIZE=${OPENPOSE_BATCH_SIZE:-32}
export OPENPOSE_NUM_GPU=${OPENPOSE_NUM_GPU:-1}
export OPENPOSE_NUM_GPU_START=${OPENPOSE_NUM_GPU_START:-0}
export OPENPOSE_WRITE_IMAGES=${OPENPOSE_WRITE_IMAGES:-0}
# Ensure received payload dir and buckets have defaults if not provided
export RECEIVED_PAYLOAD_DIR=${RECEIVED_PAYLOAD_DIR:-/opt/skeleton_metric-api/received_payloads}
# Standardized env names used by the codebase:
# - S3_VIDEO_BUCKET_NAME : source/input videos (required)
# - S3_RESULT_BUCKET_NAME: where processed results/overlays are uploaded (required)
export S3_VIDEO_BUCKET_NAME=${S3_VIDEO_BUCKET_NAME:-}
export S3_RESULT_BUCKET_NAME=${S3_RESULT_BUCKET_NAME:-}
export AWS_REGION=${AWS_REGION:-}
exec python3 -m uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-19030} --workers 1
