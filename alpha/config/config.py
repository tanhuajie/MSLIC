import torch.nn as nn
from utils.utils import Config

def model_config():
    config = Config({
        "N": 192,
        "slice_ch": 32,
        "slice_num": [8,8,8],
        "act": nn.GELU
    })

    return config
