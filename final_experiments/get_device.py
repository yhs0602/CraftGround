import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
    elif torch.has_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
