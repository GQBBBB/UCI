import torch


def update_proto(y, x, m=0.99):
    N, D = x.shape
    x_i = x.view(N, 1, D)  # (N, 1, D) samples
    c_j = y.view(1, y.shape[0], D)  # (1, K, D) centroids 
    D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
    cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster  
    for i in range(y.shape[0]):
        idx = torch.nonzero(cl==i).squeeze()
        mean_x = x.index_select(0, idx).mean(0)
        if not torch.isnan(mean_x).any():
            y[i] = y[i] * m + mean_x * (1 - m)    
    return y 
    
    
def init_proto(y, x=None):
    z = []
    for i in range(y.shape[0]):
        if torch.count_nonzero(y[i]) == 0:
            z.append(i)
    if x is not None:
        for i, t in zip(z, x):
            y[i] = t
        return y
    else:
        if len(z) == 0:
            return False
        else:
            return True
 