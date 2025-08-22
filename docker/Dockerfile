FROM python:3.8-slim-bullseye

# Prevent interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # Fix for headless OpenCV operation
    QT_QPA_PLATFORM=offscreen \
    DISPLAY=:99

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Media libraries
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    libgomp1 \
    # Development tools
    git \
    curl \
    wget \
    vim \
    ca-certificates \
    # Build dependencies
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    # Remove GTK dependencies that cause GUI issues
    # libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    # Add virtual framebuffer for headless operation
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt /workspace/

# Upgrade pip and install Python dependencies
RUN python -m pip install --upgrade pip==23.2.1 setuptools wheel

# Install PyTorch CPU first (specific order matters)
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==1.10.2+cpu torchvision==0.11.3+cpu

# Install OpenCV headless version (without Qt/GTK support)
RUN python -m pip install --no-cache-dir opencv-python-headless==4.8.1.78

# Install other requirements (excluding opencv-python if it's in requirements.txt)
RUN python -m pip install --no-cache-dir -r requirements.txt || \
    (grep -v "opencv-python" requirements.txt > requirements_filtered.txt && \
     python -m pip install --no-cache-dir -r requirements_filtered.txt)

# Create necessary directories
RUN mkdir -p /workspace/models \
             /workspace/videos \
             /workspace/outputs \
             /workspace/calibration \
             /workspace/logs

# Copy the main script
COPY stereoDetection.py /workspace/

# Create test script for debugging
RUN echo '#!/bin/bash\n\
echo "=== STEREO OYSTER DETECTION - SYSTEM CHECK ==="\n\
echo "Python version:"\n\
python --version\n\
echo "\nInstalled packages:"\n\
pip list | grep -E "(opencv|onnx|torch|numpy|scipy)"\n\
echo "\nOpenCV info:"\n\
python -c "import cv2; print(f\"OpenCV: {cv2.__version__}\"); print(f\"Build info: {cv2.getBuildInformation()[:500]}\")" \n\
echo "\nONNX Runtime info:"\n\
python -c "import onnxruntime as ort; print(f\"ONNX Runtime: {ort.__version__}, Providers: {ort.get_available_providers()}\")" \n\
echo "\nTorch info:"\n\
python -c "import torch; print(f\"PyTorch: {torch.__version__}, CPU: {torch.device('cpu')}\")" \n\
echo "\nEnvironment variables:"\n\
echo "QT_QPA_PLATFORM: $QT_QPA_PLATFORM"\n\
echo "DISPLAY: $DISPLAY"\n\
echo "\nDirectories:"\n\
ls -la /workspace/\n\
echo "\nSystem info:"\n\
cat /proc/meminfo | head -3\n\
cat /proc/cpuinfo | grep "model name" | head -1\n\
echo "==============================================="\n\
' > /workspace/system_check.sh && chmod +x /workspace/system_check.sh

# Create a startup script that handles headless mode automatically
RUN echo '#!/bin/bash\n\
# Start virtual framebuffer if needed\n\
if [ -z "$DISPLAY" ] || [ "$DISPLAY" = ":99" ]; then\n\
    echo "Starting virtual framebuffer for headless operation..."\n\
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
    export DISPLAY=:99\n\
    sleep 2\n\
fi\n\
\n\
# Add --no-display flag if not already present and not explicitly disabled\n\
if [[ "$@" == *"stereoDetection.py"* ]] && [[ "$@" != *"--no-display"* ]] && [[ "$@" != *"--help"* ]]; then\n\
    echo "Auto-enabling headless mode for Docker environment"\n\
    exec python "$@" --no-display\n\
else\n\
    exec python "$@"\n\
fi\n\
' > /workspace/docker_run.sh && chmod +x /workspace/docker_run.sh

# Expose port for HTTP streaming
EXPOSE 5000

# Set default command to use the wrapper script
CMD ["/workspace/docker_run.sh", "stereoDetection.py", "--help"]
