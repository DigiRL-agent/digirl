
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/content/MLRL-group-project-digiRL"
REPO_URL="https://github.com/GodheadKevin/MLRL-group-project-digiRL.git"

echo "=== 1. Installing system dependencies ==="
apt-get update
apt-get install -y git curl unzip ca-certificates

echo "=== 2. Installing uv ==="
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"

uv --version

echo "=== 3. Installing Python 3.10 ==="
uv python install 3.10

echo "=== 4. Cloning DigiRL repository ==="
mkdir -p /content

if [ -d "$PROJECT_DIR/.git" ]; then
    echo "Repository already exists. Keeping local changes."
else
    git clone "$REPO_URL" "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"

echo "=== 5. Creating virtual environment ==="
if [ ! -d ".venv" ]; then
    uv venv .venv --python 3.10
fi

source .venv/bin/activate

echo "=== 6. Installing PyTorch ==="
uv pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== 7. Installing Auto-UI dependencies ==="
uv pip install \
    transformers==4.45.2 \
    accelerate==0.27.2 \
    sentencepiece \
    protobuf \
    pillow \
    numpy \
    scipy \
    safetensors \
    huggingface_hub \
    gradio_client

echo "=== 8. Checking environment ==="
python --version

python - <<'PY'
import torch
import transformers

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"GPU VRAM available: {free / 1024**3:.2f} GB")
    print(f"GPU VRAM total: {total / 1024**3:.2f} GB")
else:
    print("No CUDA GPU detected.")
PY

echo "=== 9. System RAM ==="
free -h

echo "=== Setup complete ==="
echo "Project: $PROJECT_DIR"
echo "Activate environment with:"
echo "source $PROJECT_DIR/.venv/bin/activate"
