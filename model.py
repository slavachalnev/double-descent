import torch
import torch.nn as nn

import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, m=2, n=10000, dtype=torch.float32):
        super(Model, self).__init__()

        # W is m x n
        self.W = nn.Parameter(torch.Tensor(m, n))

        # xavier init
        nn.init.xavier_uniform_(self.W)

        # b shape is (m,)
        self.b = nn.Parameter(torch.zeros(n, 1))

        self.to(dtype=dtype)
    

    def forward(self, x, return_h=False):
        x = self.W @ x

        if return_h:
            return x
        
        x = self.W.T @ x + self.b

        x = F.relu(x)
        return x
        