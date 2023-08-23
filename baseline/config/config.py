import torch.nn as nn
from utils.utils import Config

# def model_config():
#     config = Config({
#         # MLIC and MLIC+
#         "N": 192,
#         "M": 320,
#         "slice_num": 10,
#         "context_window": 5,
#         "act": nn.GELU,
#     })

#     return config


def model_config():
    config = Config({
        "N": 192,
        "slice_ch": 32,
        "slice_num": [8,8,8],
        "act": nn.GELU
    })

    return config
