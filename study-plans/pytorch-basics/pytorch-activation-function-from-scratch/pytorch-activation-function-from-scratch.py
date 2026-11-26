import torch

def activate(x: torch.Tensor, method: str = "relu") -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as x.
    """
    if method=="relu":
        return torch.clamp(x,min=0)
    if method =="sigmoid":
        return 1.0/(1.0 + torch.exp(-x))
    if method=='tanh':
        magnitude = torch.where(x>=0,x,-x)
        decay = torch.exp(-2*magnitude)
        pos = (1-decay)/(1+decay)
        return torch.where(x>=0,pos,-pos)
    if method=="leaky_relu":
        return torch.where(x>0,x,0.01*x)
