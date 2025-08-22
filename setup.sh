#!/usr/bin/env bash
set -euo pipefail

# ============================
# Stereo Oyster Detection - Setup Script
# ============================
PROJECT_NAME="stereo-oyster-detection"
IMAGE_NAME="stereo-oyster-detection"  
CONTAINER_NAME="stereo-oyster-detection-dev"

REQUIRED_DIRS=("models" "videos" "calibration" "outputs" "logs")

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[SETUP]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
info() { echo -e "${BLUE}[INFO]${NC} $*"; }

# ---- System checks ----
check_system() {
    log "Checking system requirements..."

    # Docker binary
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker not found! Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi

    # Docker daemon running
    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon not running! Please start Docker."
        exit 1
    fi

    # Check available space (5GB minimum)
    local available_space
    available_space=$(df . | awk 'NR==2{print $4}')
    if [ "${available_space:-0}" -lt 5000000 ]; then
        warn "Low disk space detected. At least 5GB recommended for Docker images."
    fi

    # Check internet connectivity
    if ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1; then
        info "Internet connectivity OK."
    else
        warn "No internet connectivity. Docker build may fail if base images not cached."
    fi

    log "System requirements OK!"
}

# ---- Project structure ----
create_project_structure() {
    log "Creating project structure..."

    # Create directories
    for dir in "${REQUIRED_DIRS[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log "Created: $dir/"
        else
            info "Exists: $dir/"
        fi
    done

    # Create sample calibration if not present
    if [ ! -f "calibration/stereo_calibration.json" ]; then
        cat > calibration/stereo_calibration.json <<'EOF'
{
  "left_camera_matrix": [[700.0, 0.0, 640.0], [0.0, 700.0, 360.0], [0.0, 0.0, 1.0]],
  "right_camera_matrix": [[700.0, 0.0, 640.0], [0.0, 700.0, 360.0], [0.0, 0.0, 1.0]],
  "left_distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
  "right_distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
  "rotation_matrix": [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
  "translation_vector": [-100.0, 0.0, 0.0],
  "baseline": 100.0
}
EOF
        log "Created: calibration/stereo_calibration.json (sample)"
    fi

    # Create .gitignore if not exists
    if [ ! -f ".gitignore" ]; then
        cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*.so
.Python
*.egg-info/

# Outputs
outputs/
logs/
*.log

# Editors
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Docker
docker-build-cache/
EOF
        log "Created: .gitignore"
    fi

    # Make scripts executable
    chmod +x runDocker.sh >/dev/null 2>&1 || true
    chmod +x stereoDetection.py >/dev/null 2>&1 || true
    chmod +x setup.sh >/dev/null 2>&1 || true
    log "Made scripts executable."
}

# ---- Validate setup ----
validate_setup() {
    log "Validating project setup..."

    local issues=0

    # Check critical files
    if [ ! -f "runDocker.sh" ]; then
        error "Missing: runDocker.sh"
        issues=1
    fi

    if [ ! -f "stereoDetection.py" ]; then
        error "Missing: stereoDetection.py"
        issues=1
    fi

    if [ ! -f "requirements.txt" ]; then
        error "Missing: requirements.txt"
        issues=1
    fi

    if [ ! -f "Dockerfile" ] && [ ! -f "docker/Dockerfile" ]; then
        error "Missing: Dockerfile (neither in root nor docker/ folder)"
        issues=1
    fi

    # Soft warnings
    if [ ! -f "models/best.onnx" ]; then
        warn "No model found. Copy your ONNX model to models/best.onnx"
        echo "   Example: cp /path/to/your/model.onnx models/best.onnx"
    fi

    if [ ! -f "videos/left.mp4" ] || [ ! -f "videos/right.mp4" ]; then
        warn "No test videos found. Copy test videos to videos/ folder for testing"
        echo "   Example: cp left_cam.mp4 videos/left.mp4"
        echo "            cp right_cam.mp4 videos/right.mp4"
    fi

    if [ "$issues" -eq 1 ]; then
        error "Critical files missing! Please ensure all required files are present."
        exit 1
    fi

    log "Project validation complete."
}

# ---- Optional Docker build ----
docker_build_prompt() {
    if [ -f "Dockerfile" ] || [ -f "docker/Dockerfile" ]; then
        echo
        info "Docker image build options:"
        echo "  1) Build now (recommended for first setup)"
        echo "  2) Skip build (build later with ./runDocker.sh)"
        echo
        read -p "Choose [1/2]: " choice
        
        case "$choice" in
            1)
                log "Building Docker image..."
                # Use same logic as runDocker.sh for finding Dockerfile
                local dockerfile=""
                if [ -f "Dockerfile" ]; then
                    dockerfile="Dockerfile"
                elif [ -f "docker/Dockerfile" ]; then
                    dockerfile="docker/Dockerfile"
                fi
                
                docker build -t "${IMAGE_NAME}" -f "$dockerfile" . || {
                    error "Docker build failed!"
                    exit 1
                }
                log "Docker image built successfully!"
                
                # Quick test
                log "Running quick test..."
                docker run --rm "${IMAGE_NAME}" python3 stereoDetection.py --help >/dev/null 2>&1 && {
                    log "Docker image test passed!"
                } || {
                    warn "Docker image test failed, but image was built."
                }
                ;;
            2)
                info "Skipping Docker build. Run './runDocker.sh' to build when needed."
                ;;
            *)
                warn "Invalid choice. Skipping Docker build."
                ;;
        esac
    else
        warn "No Dockerfile found. Skipping Docker build."
    fi
}

# ---- Main ----
main() {
    echo
    log "=== Stereo Oyster Detection - Setup ==="
    echo
    
    check_system
    create_project_structure
    validate_setup
    docker_build_prompt
    
    echo
    log "=== Setup Complete! ==="
    echo
    info "Next steps:"
    echo "  1. Copy your ONNX model: cp your_model.onnx models/best.onnx"
    echo "  2. Copy test videos (optional): cp left.mp4 videos/ && cp right.mp4 videos/"
    echo "  3. Run detection: ./runDocker.sh"
    echo
    info "Quick start:"
    echo "  ./runDocker.sh  # Interactive menu"
    echo
}

# Handle help
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Stereo Oyster Detection - Setup Script"
    echo
    echo "This script prepares the project for first use:"
    echo "  - Checks system requirements (Docker)"
    echo "  - Creates required directory structure"
    echo "  - Validates critical files presence"
    echo "  - Optionally builds Docker image"
    echo
    echo "Usage: ./setup.sh"
    echo
    exit 0
fi

main "$@"
