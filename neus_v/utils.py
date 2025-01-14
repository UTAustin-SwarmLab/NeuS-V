import gc

import torch


def clear_gpu_memory():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    gc.collect()
