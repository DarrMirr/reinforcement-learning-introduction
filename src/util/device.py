import os

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'garbage_collection_threshold:0.9'
# Priority NPU over GPU
os.environ['HIP_VISIBLE_DEVICES'] = '0' # 0 - GPU, 1 - NPU (on CPU)

import torch

from util.logger import log

accelerator = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
log.info(f'Using device={accelerator}')


def is_cuda() -> bool:
    return accelerator == "cuda"
