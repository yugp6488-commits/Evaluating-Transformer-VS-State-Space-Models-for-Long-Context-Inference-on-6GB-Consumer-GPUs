#!/bin/bash
echo "=== Checking Python environment ==="
python3 --version
echo ""
echo "=== Key packages ==="
python3 -c "
import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  GPU:', torch.cuda.get_device_name(0))
    print('  VRAM:', round(torch.cuda.get_device_properties(0).total_mem/(1024**2)), 'MB')
import transformers; print('transformers:', transformers.__version__)
try:
    import mamba_ssm; print('mamba_ssm: available')
except ImportError as e:
    print('mamba_ssm: NOT INSTALLED -', str(e)[:80])
try:
    import datasets; print('datasets:', datasets.__version__)
except ImportError:
    print('datasets: NOT INSTALLED')
try:
    import psutil; print('psutil:', psutil.__version__)
except ImportError:
    print('psutil: NOT INSTALLED')
try:
    import matplotlib; print('matplotlib:', matplotlib.__version__)
except ImportError:
    print('matplotlib: NOT INSTALLED')
try:
    import pandas; print('pandas:', pandas.__version__)
except ImportError:
    print('pandas: NOT INSTALLED')
import numpy; print('numpy:', numpy.__version__)
"
