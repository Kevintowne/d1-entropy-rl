#!/usr/bin/env bash
python - <<'PY'
import torch, platform
print("Python:", platform.python_version())
print("Torch:", getattr(torch,"__version__",None), "| CUDA:", getattr(torch.version,"cuda",None))
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
PY
