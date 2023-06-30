# %%
import math
import torch

import matplotlib.pyplot as plt

from model import Model


# data is a shape (n, T) tensor
def get_data(T, n=10000, sparsity=0.999):
    x = torch.rand(n, T)
    mask = torch.rand(n, T) < sparsity
    x[mask] = 0

    # rescale so that each row is unit norm
    x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-5)

    return x


def loss(x, x_hat):
    T = x.shape[1]
    return torch.sum((x - x_hat) ** 2) / T


def train(model, data, steps=50000, peak_lr=0.001, warmup_steps=2500):
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    warmup_func = lambda x: x / warmup_steps
    decay_func = lambda x: 0.5 * (1 + math.cos(math.pi * (x - warmup_steps) / (steps - warmup_steps)))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: warmup_func(x) if x <= warmup_steps else decay_func(x))

    for step in range(steps):
        optimizer.zero_grad()
        x_hat = model(data)
        l = loss(data, x_hat)
        l.backward()
        optimizer.step()
        lr_scheduler.step()
        if step % 100 == 0:
            print(step, l.item())
            print(optimizer.param_groups[0]['lr'])
        
    T = data.shape[1]
    torch.save(model.state_dict(), f'model_T{T}.pt')

    # test
    test_data = get_data(1000)
    with torch.no_grad():
        x_hat = model(test_data)
        l = loss(test_data, x_hat)
        print('test loss', l.item())
    
    
# %%

model = Model()
data = get_data(T=10)
train(model, data)#, steps=100000)

# %%

def plot_sample_h(data, model):
    hs_x = []
    hs_y = []
    fractional_dims = []

    with torch.no_grad():
        h = model(data, return_h=True)
    
    for h_i in h.T:
        numerator = torch.norm(h_i) ** 2
        h_i_hat = h_i / torch.norm(h_i)
        denominator = torch.sum((h_i_hat @ h) ** 2)
        fractional_dims.append(numerator / denominator)
    
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
    plt.show()

    return fractional_dims

plot_sample_h(data, model)



# %%

def plot_feats(model):
    # plots columns of W
    W = model.W.detach().cpu().numpy()
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

    plt.show()

plot_feats(model)

# %%


