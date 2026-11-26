import torch

def create_tensor(method: str, shape: list, value: float = 0.0) -> torch.Tensor:
    """
    Returns a float32 tensor with the requested shape.
    """
    if method == "zeros":
        return torch.zeros(shape,dtype=torch.float32)
    elif method =="ones":
        return torch.ones(shape,dtype=torch.float32)
    elif method =="full":
        return torch.full(shape,value,dtype=torch.float32)
    elif method=="eye":
        return torch.eye(shape[0],shape[1],dtype=torch.float32)
    elif method=="arange":
        return torch.arange(shape[0],shape[1],dtype=torch.float32)
    elif method=="linspace":
        return torch.linspace(0, value, shape[0], dtype=torch.float32)

