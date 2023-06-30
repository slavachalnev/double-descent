# %%
import math
import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

from model import Model


# data is a shape (n, T) tensor
def get_data(T, n=10000, sparsity=0.999):
    x = torch.rand(n, T)
    mask = torch.rand(n, T) < sparsity
    x[mask] = 0

    # rescale
    x = x / (torch.norm(x, dim=0, keepdim=True) + 1e-5)

    return x


def loss(x, x_hat):
    T = x.shape[1]
    return torch.sum((x - x_hat) ** 2) / T


def train(model, data, steps=50000, peak_lr=0.001, warmup_steps=2500, device='cpu'):
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    warmup_func = lambda x: x / warmup_steps
    decay_func = lambda x: 0.5 * (1 + math.cos(math.pi * (x - warmup_steps) / (steps - warmup_steps)))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: warmup_func(x) if x <= warmup_steps else decay_func(x))

    for step in tqdm(range(steps)):
        optimizer.zero_grad()
        x_hat = model(data)
        l = loss(data, x_hat)
        l.backward()
        optimizer.step()
        lr_scheduler.step()
        
    model = model.cpu()
    data = data.cpu()

    T = data.shape[1]
    torch.save(model.state_dict(), f'model_T{T}.pt')

    # test
    test_data = get_data(1000)
    with torch.no_grad():
        x_hat = model(test_data)
        l = loss(test_data, x_hat)
        print('test loss', l.item())
    
    
# %%

# model = Model()
# data = get_data(T=10)
# train(model, data)#, steps=100000)

# %%

def compute_fractional_dims(h):
    fractional_dims = []

    for h_i in h.T:
        numerator = torch.norm(h_i) ** 2
        h_i_hat = h_i / torch.norm(h_i)
        denominator = torch.sum((h_i_hat @ h) ** 2)
        fractional_dims.append(numerator / denominator)
    
    return fractional_dims


def plot_sample_h(data, model, T):
    hs_x = []
    hs_y = []

    with torch.no_grad():
        h = model(data, return_h=True)
    
    hs_x = h[0].cpu().numpy()
    hs_y = h[1].cpu().numpy()

    plt.figure(figsize=(10, 10))
    max_abs = max(abs(max(hs_x)), abs(min(hs_x)), abs(max(hs_y)), abs(min(hs_y))) 
    plt.xlim(-max_abs, max_abs)
    plt.ylim(-max_abs, max_abs)

    plt.plot(hs_x, hs_y, 'o', color='red')

    # connect every point to (0, 0)
    plt.plot([0, 0], [0, 0], '', color='red')
    for x, y in zip(hs_x, hs_y):
        plt.plot([0, x], [0, y], color='red')
    
    plt.savefig(f'sample_{T}.png')
    plt.show()
    plt.close()

    return compute_fractional_dims(h)

# plot_sample_h(data, model)



# %%

def plot_feats(model, T):
    # plots columns of W
    W = model.W.detach().cpu()
    # W is 2 x n

    plt.figure(figsize=(10, 10))
    max_abs = max(W.min(), W.max(), key=abs)
    plt.xlim(-max_abs, max_abs)
    plt.ylim(-max_abs, max_abs)

    plt.plot(W[0], W[1], 'o', color='blue')

    # connect every point to (0, 0)
    plt.plot([0, 0], [0, 0], '', color='blue')
    for x, y in zip(W[0], W[1]):
        plt.plot([0, x], [0, y], color='blue')

    plt.savefig(f'feats_{T}.png')
    plt.show()
    plt.close()

    return compute_fractional_dims(W)


# plot_feats(model)

# %%

def run_experiment(T, device='cpu'):
    model = Model()
    data = get_data(T)
    train(model, data, device=device)
    sample_dims = plot_sample_h(data, model, T)
    feat_dims = plot_feats(model, T)

    return sample_dims, feat_dims

# run_experiment(5)

# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ts = [3, 5,]# 6, 8, 10, 15, 30,]# 50, 100, 1000, 10000]

sample_dims = []
feat_dims = []

for T in ts:
    print('T', T)
    sample_dim, feat_dim = run_experiment(T, device=device)
    sample_dims.append(sample_dim)
    feat_dims.append(feat_dim)

# %%

# plot sample and feat dims
plt.figure(figsize=(10, 10))

# For each T, plot all the corresponding sample_dims and feat_dims
for T, s_dim, f_dim in zip(ts, sample_dims, feat_dims):
    plt.scatter([T]*len(s_dim), s_dim, color='red')
    plt.scatter([T]*len(f_dim), f_dim, color='blue')

plt.savefig('dims.png')
plt.show()
plt.close()


# %%
