import torch.nn as nn
from utils.utils import Config

def model_config():
    config = Config({
        "N": 192,
        "M": 320,
        "num_slices": 7
    })

    return config
