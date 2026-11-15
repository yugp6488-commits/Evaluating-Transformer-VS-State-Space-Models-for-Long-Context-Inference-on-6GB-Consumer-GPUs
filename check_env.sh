#!/bin/bash
if [ -d "/home/yugpa/benchmark-env" ]; then
    echo "Found /home/yugpa/benchmark-env"
    source /home/yugpa/benchmark-env/bin/activate
    python3 -c "
import torch
print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
import mamba_ssm
print('mamba_ssm: OK!')
import matplotlib
print('matplotlib: OK!')
import psutil
print('psutil: OK!')
"
else
    echo "Not found"
fi
