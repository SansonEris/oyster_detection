#!/usr/bin/env bash
set -euo pipefail

# ============================
# Stereo Oyster Detection - Docker Runner
# ============================

IMAGE_NAME="stereo-oyster-detection"
PYTHON_SCRIPT="stereoDetection.py"

# Get absolute path of project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_WORKDIR="/workspace"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Argomenti raccolti dal menu (FIX)
RUNTIME_ARGS=()

# ============================
# Helper Functions
# ============================

log() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
info() { echo -e "${BLUE}[INFO]${NC} $*"; }

check_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker non installato! Installa Docker prima di continuare."
        exit 1
    fi

    if ! docker info >/dev/null 2>&1; then
        error "Docker non in esecuzione! Avvia il daemon Docker."
        exit 1
    fi
}

ensure_directories() {
    # Create required directories if they don't exist
    local dirs=("models" "videos" "outputs" "calibration" "logs")
    for dir in "${dirs[@]}"; do
        if [ ! -d "${PROJECT_DIR}/${dir}" ]; then
            mkdir -p "${PROJECT_DIR}/${dir}"
            log "Created directory: ${dir}/"
        fi
    done
}

build_image() {
    log "Building Docker image: ${IMAGE_NAME}"

    # Check for Dockerfile in multiple locations
    local dockerfile_paths=(
        "${PROJECT_DIR}/Dockerfile"
        "${PROJECT_DIR}/docker/Dockerfile"
        "${PROJECT_DIR}/Docker/Dockerfile"
    )

    local dockerfile=""
    for path in "${dockerfile_paths[@]}"; do
        if [ -f "$path" ]; then
            dockerfile="$path"
            break
        fi
    done

    if [ -z "$dockerfile" ]; then
        error "Dockerfile not found! Please ensure Dockerfile exists in:"
        for path in "${dockerfile_paths[@]}"; do
            error "  - $path"
        done
        exit 1
    fi

    info "Using Dockerfile: $dockerfile"

    docker build \
        -t "${IMAGE_NAME}" \
        -f "$dockerfile" \
        "${PROJECT_DIR}" || {
        error "Docker build failed!"
        exit 1
    }

    log "Docker image built successfully!"
}

check_image() {
    if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
        warn "Image '${IMAGE_NAME}' not found. Building..."
        build_image
    else
        info "Using existing image: ${IMAGE_NAME}"

        # Ask if user wants to rebuild (only in interactive mode)
        if [ -t 0 ]; then
            echo -n "Rebuild image? [y/N]: "
            read -r rebuild
            if [[ "$rebuild" =~ ^[Yy]$ ]]; then
                build_image
            fi
        fi
    fi
}

print_system_info() {
    info "=== SYSTEM INFO ==="
    info "Project directory: ${PROJECT_DIR}"
    info "Container workdir: ${CONTAINER_WORKDIR}"

    # Check available models
    if [ -d "${PROJECT_DIR}/models" ] && [ "$(ls -A "${PROJECT_DIR}/models" 2>/dev/null)" ]; then
        info "Available models:"
        find "${PROJECT_DIR}/models" -name "*.onnx" 2>/dev/null | head -5 | sed 's|^|  |' || true
    else
        warn "No models found in models/ directory!"
        warn "Please copy your ONNX model to: ${PROJECT_DIR}/models/"
    fi

    # Check available videos
    if [ -d "${PROJECT_DIR}/videos" ] && [ "$(ls -A "${PROJECT_DIR}/videos" 2>/dev/null)" ]; then
        info "Available videos:"
        find "${PROJECT_DIR}/videos" -name "*.mp4" 2>/dev/null | head -5 | sed 's|^|  |' || true
    else
        warn "No videos found in videos/ directory!"
    fi

    # Check script
    if [ -f "${PROJECT_DIR}/stereoDetection.py" ]; then
        info "Main script: ✓ stereoDetection.py found"
    else
        error "Main script not found: ${PROJECT_DIR}/stereoDetection.py"
        exit 1
    fi

    info "===================="
}

compute_port_mapping() {
    local http_used=0
    local http_port="5000"
    local prev=""

    for arg in "$@"; do
        if [ "$prev" = "--http-port" ]; then
            http_port="$arg"
            prev=""
            continue
        fi
        case "$arg" in
            --http) http_used=1 ;;
            --http-port) prev="--http-port" ;;
        esac
    done

    if [ "$http_used" -eq 1 ]; then
        echo "-p ${http_port}:${http_port}"
    fi
}

convert_path() {
    local path="$1"
    # Convert host paths to container paths
    if [[ "$path" = "${PROJECT_DIR}"* ]]; then
        echo "${path/${PROJECT_DIR}/${CONTAINER_WORKDIR}}"
    elif [[ "$path" = /* ]]; then
        # Absolute path outside project - keep as is
        echo "$path"
    else
        # Relative path - prepend container workdir
        echo "${CONTAINER_WORKDIR}/$path"
    fi
}

# ============================
# Interactive Menu
# ============================

interactive_menu() {
    echo
    info "=== STEREO OYSTER DETECTION - INTERACTIVE MODE ==="
    echo "Choose detection mode:"
    echo "  1) Stereo video files (left.mp4 + right.mp4)"
    echo "  2) Live RTSP/HTTP streams"
    echo "  3) Mono video file (single source)"
    echo "  4) System check only"
    echo "  5) Custom command"
    echo "  6) Open bash shell in container"
    echo

    while true; do
        echo -n "Select option [1-6]: "
        read -r choice
        case "$choice" in
            1) setup_stereo_files; break ;;
            2) setup_live_streams; break ;;
            3) setup_mono; break ;;
            4) system_check_only; break ;;
            5) custom_command; break ;;
            6) open_bash_shell; return ;;
            *) warn "Invalid choice. Please select 1-6." ;;
        esac
    done
}

setup_stereo_files() {
    echo
    info "=== STEREO VIDEO FILES SETUP ==="

    # Default values
    local default_model="models/best.onnx"
    local default_left="videos/left.mp4"
    local default_right="videos/right.mp4"
    local default_conf="0.25"
    local default_size="416"

    echo -n "Model path [$default_model]: "
    read -r model
    model="${model:-$default_model}"

    echo -n "Left video [$default_left]: "
    read -r left
    left="${left:-$default_left}"

    echo -n "Right video [$default_right]: "
    read -r right
    right="${right:-$default_right}"

    echo -n "Confidence threshold [$default_conf]: "
    read -r conf
    conf="${conf:-$default_conf}"

    echo -n "Input size [$default_size]: "
    read -r size
    size="${size:-$default_size}"

    echo -n "Enable HTTP streaming? [y/N]: "
    read -r enable_http

    echo -n "Save output video? [Y/n]: "
    read -r save_video

    # Convert paths
    model=$(convert_path "$model")
    left=$(convert_path "$left")
    right=$(convert_path "$right")

    # Build command - ALWAYS add --no-display for Docker
    local args=(
        "--model" "$model"
        "--left" "$left"
        "--right" "$right"
        "--conf" "$conf"
        "--size" "$size"
        "--output" "${CONTAINER_WORKDIR}/outputs"
        "--no-display"  # Always headless in Docker
    )

    if [[ "$enable_http" =~ ^[Yy]$ ]]; then
        args+=("--http" "--http-port" "5000")
        info "💡 HTTP stream will be available at: http://localhost:5000"
    fi

    if [[ ! "$save_video" =~ ^[Nn]$ ]]; then
        args+=("--save-video")
    fi

    RUNTIME_ARGS=("${args[@]}")
}

setup_live_streams() {
    echo
    info "=== LIVE STREAMS SETUP ==="

    echo "Enter stream URLs (RTSP/HTTP):"
    echo -n "Left camera URL: "
    read -r left_url
    echo -n "Right camera URL: "
    read -r right_url

    if [ -z "$left_url" ] || [ -z "$right_url" ]; then
        error "Both stream URLs are required!"
        return 1
    fi

    echo -n "Model path [models/best.onnx]: "
    read -r model
    model="${model:-models/best.onnx}"
    model=$(convert_path "$model")

    local args=(
        "--model" "$model"
        "--left" "$left_url"
        "--right" "$right_url"
        "--conf" "0.25"
        "--size" "416"
        "--http"
        "--http-port" "5000"
        "--no-display"  # Always headless in Docker
        "--output" "${CONTAINER_WORKDIR}/outputs"
    )

    info "💡 HTTP stream will be available at: http://localhost:5000"
    RUNTIME_ARGS=("${args[@]}")
}

setup_mono() {
    echo
    info "=== MONO VIDEO FILE SETUP ==="

    # Default values
    local default_model="models/best.onnx"
    local default_video="videos/left.mp4"
    local default_conf="0.25"
    local default_size="416"

    echo -n "Model path [$default_model]: "
    read -r model
    model="${model:-$default_model}"

    echo -n "Video file [$default_video]: "
    read -r video
    video="${video:-$default_video}"

    echo -n "Confidence threshold [$default_conf]: "
    read -r conf
    conf="${conf:-$default_conf}"

    echo -n "Input size [$default_size]: "
    read -r size
    size="${size:-$default_size}"

    echo -n "Enable HTTP streaming? [y/N]: "
    read -r enable_http

    echo -n "Save output video? [Y/n]: "
    read -r save_video

    # Convert paths
    model=$(convert_path "$model")
    video=$(convert_path "$video")

    # Build command with --mono flag
    local args=(
        "--model" "$model"
        "--left" "$video"
        "--mono"  # Enable mono mode
        "--conf" "$conf"
        "--size" "$size"
        "--output" "${CONTAINER_WORKDIR}/outputs"
        "--no-display"  # Always headless in Docker
    )

    if [[ "$enable_http" =~ ^[Yy]$ ]]; then
        args+=("--http" "--http-port" "5000")
        info "💡 HTTP stream will be available at: http://localhost:5000"
    fi

    if [[ ! "$save_video" =~ ^[Nn]$ ]]; then
        args+=("--save-video")
    fi

    RUNTIME_ARGS=("${args[@]}")
}

system_check_only() {
    info "Running system check..."
    RUNTIME_ARGS=("/workspace/system_check.sh")
}

custom_command() {
    echo
    info "=== CUSTOM COMMAND ==="
    echo "Enter your custom arguments for stereoDetection.py:"
    echo "Examples:"
    echo "  --model /workspace/models/best.onnx --left /workspace/videos/left.mp4 --right /workspace/videos/right.mp4 --http --no-display"
    echo "  --model /workspace/models/best.onnx --left /workspace/videos/video.mp4 --mono --http --no-display"
    echo -n "Arguments: "
    read -r custom_args

    if [ -z "$custom_args" ]; then
        warn "No arguments provided, showing help..."
        RUNTIME_ARGS=("--help")
    else
        # shellcheck disable=SC2206
        RUNTIME_ARGS=($custom_args)
        
        # Auto-add --no-display if not present and not help
        if [[ "$custom_args" != *"--no-display"* ]] && [[ "$custom_args" != *"--help"* ]]; then
            warn "Adding --no-display for headless operation in Docker"
            RUNTIME_ARGS+=("--no-display")
        fi
    fi
}

open_bash_shell() {
    echo
    info "=== OPENING BASH SHELL IN CONTAINER ==="
    warn "Remember to use --no-display flag if running detection commands!"
    echo

    local DOCKER_CMD=(
        docker run --rm -it
        -v "${PROJECT_DIR}:${CONTAINER_WORKDIR}"
        -w "${CONTAINER_WORKDIR}"
        -e "QT_QPA_PLATFORM=offscreen"
        -e "DISPLAY=:99"
        -p 5000:5000
        --name "${IMAGE_NAME}-shell"
        "${IMAGE_NAME}"
        bash
    )

    info "Starting interactive shell..."
    exec "${DOCKER_CMD[@]}"
}

# ============================
# Main Function
# ============================

main() {
    # Check prerequisites
    check_docker
    ensure_directories
    check_image
    print_system_info

    # Handle arguments
    if [ "$#" -eq 0 ]; then
        interactive_menu
    fi

    # Usa gli argomenti del menu, se presenti
    local ARGS=("$@")
    if [ ${#ARGS[@]} -eq 0 ] && [ ${#RUNTIME_ARGS[@]} -gt 0 ]; then
        ARGS=("${RUNTIME_ARGS[@]}")
    fi

    # Auto-add --no-display if not present and not system check or help
    if [ ${#ARGS[@]} -gt 0 ] && [ "${ARGS[0]}" != "/workspace/system_check.sh" ]; then
        local has_no_display=0
        local has_help=0
        for arg in "${ARGS[@]}"; do
            case "$arg" in
                --no-display) has_no_display=1 ;;
                --help) has_help=1 ;;
            esac
        done
        
        if [ "$has_no_display" -eq 0 ] && [ "$has_help" -eq 0 ]; then
            warn "Auto-adding --no-display for headless Docker operation"
            ARGS+=("--no-display")
        fi
    fi

    # Compute port mapping
    PORT_MAP=$(compute_port_mapping "${ARGS[@]}")
    PORT_ARGS=()
    if [ -n "$PORT_MAP" ]; then
        # shellcheck disable=SC2206
        PORT_ARGS=($PORT_MAP)
        log "Exposing HTTP port: $(echo "$PORT_MAP" | cut -d: -f2)"
    fi

    # Prepare Docker command with headless environment
    DOCKER_CMD=(
        docker run --rm -it
        -v "${PROJECT_DIR}:${CONTAINER_WORKDIR}"
        -w "${CONTAINER_WORKDIR}"
        -e "QT_QPA_PLATFORM=offscreen"
        -e "DISPLAY=:99"
        -e "PYTHONUNBUFFERED=1"
        "${PORT_ARGS[@]}"
        "${IMAGE_NAME}"
    )

    # Add Python script unless it's a system check
    if [ "${ARGS[0]-}" != "/workspace/system_check.sh" ]; then
        DOCKER_CMD+=("python" "${PYTHON_SCRIPT}")
    fi

    DOCKER_CMD+=("${ARGS[@]}")

    echo
    log "Starting Docker container with headless configuration..."
    
    # Show command in debug mode
    if [ "${DEBUG:-0}" = "1" ]; then
        info "Debug - Full command: ${DOCKER_CMD[*]}"
    fi
    
    echo

    # Show helpful info based on arguments
    local show_http_info=0
    for arg in "${ARGS[@]}"; do
        if [ "$arg" = "--http" ]; then
            show_http_info=1
            break
        fi
    done

    if [ "$show_http_info" -eq 1 ]; then
        echo
        info "🌐 HTTP Stream Access:"
        info "   Open your browser and go to: http://localhost:5000"
        info "   The stream will be available once detection starts"
        echo
        info "⏹️  To stop: Press Ctrl+C in this terminal"
        echo
    fi

    # Execute
    exec "${DOCKER_CMD[@]}"
}

# ============================
# Script Entry Point
# ============================

# Handle special flags
case "${1:-}" in
    --help|-h)
        echo "Stereo Oyster Detection - Docker Runner"
        echo
        echo "Usage:"
        echo "  $0                    # Interactive mode"
        echo "  $0 [OPTIONS]         # Direct command mode"
        echo "  $0 --system-check    # Run system check only"
        echo "  $0 --rebuild         # Force rebuild Docker image"
        echo "  $0 --shell           # Open bash shell in container"
        echo
        echo "Examples:"
        echo "  $0  # Interactive menu"
        echo "  $0 --model models/best.onnx --left videos/left.mp4 --right videos/right.mp4 --http"
        echo "  $0 --model models/best.onnx --left videos/video.mp4 --mono --http"
        echo "  $0 --system-check"
        echo
        echo "Note: --no-display is automatically added for Docker headless operation"
        echo
        exit 0
        ;;
    --system-check)
        check_docker
        check_image
        system_check_only
        ;;
    --rebuild)
        check_docker
        ensure_directories
        build_image
        exit 0
        ;;
    --shell)
        check_docker
        check_image
        open_bash_shell
        ;;
esac

# Run main function with all arguments
main "$@"
