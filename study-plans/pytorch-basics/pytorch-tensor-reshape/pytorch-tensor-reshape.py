import torch

def reshape_tensor(x: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the reshaped float32 tensor, including a scalar tensor after a complete squeeze.
    """
    if op=="view":
        return torch.view(x.shape[1], x.shape[0])
    if op =="flatten":
        return torch.flatten(x)
    if op =="squeeze":
        return torch.squeeze(x);
    if op=="unsqueeze":
        return torch.unsqueeze(x,0)
    if op=="permute":
        return torch.permute(*reversed(range(x.ndim)))
    if op=="transpose":
        return torch.transpose(x, 0, 1)
