import torch

from torch import nn

import torch.nn.functional as F

a = torch.randn(1, 10, 10)
b = torch.randn(1, 15, 10)


"""
    return a 10x15 matrix
"""
a_len = a.size(1)
b_len = b.size(1)
sim = torch.Tensor(a.size(1), b.size(1))

for i in range(a_len):
    for j in range(b_len):
        sim[i, j] = F.cosine_similarity(a[:, i], b[:, j], dim=1)

print(sim)