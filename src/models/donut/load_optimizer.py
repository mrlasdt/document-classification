import torch


def load_optimizer(model, lr):
    return torch.optim.AdamW(params=model.parameters(), lr=lr)
