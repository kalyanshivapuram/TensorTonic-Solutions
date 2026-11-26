import torch

def tensor_op(x: torch.Tensor, y: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the operation result as a float32 tensor.
    """
    if op=="add":
        return torch.add(x,y)
    if op =="matmul":
        return torch.matmul(x,y)
    if op =="power":
        return torch.pow(x,y);
    if op=="multiply":
        return torch.mul(x,y)
    if op=="max":
        return torch.max(x,y)
    
