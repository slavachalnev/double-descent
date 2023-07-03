# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import Model


def measure_model(model):
    """Frobeinus norm of W, norm of b"""
    W = model.W.detach().cpu().numpy()
    b = model.b.detach().cpu().numpy()
    return np.linalg.norm(W), np.linalg.norm(b)


ts = [3, 5, 6, 8, 10, 15, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
model_dir = '.'

W_norms = []
b_norms = []

for T in ts:
    model_name = f'{model_dir}/model_T{T}.pt'
    model = Model()
    model.load_state_dict(torch.load(model_name, map_location='cpu'))

    W_norm, b_norm = measure_model(model)

    W_norms.append(W_norm)
    b_norms.append(b_norm)

# %%

# plot W_norm and b_norm vs T.
plt.figure(figsize=(10, 5))
plt.plot(ts, W_norms, color='black')
plt.plot(ts, b_norms, color='red')

plt.xlabel('T')
plt.ylabel('Norm')
plt.legend(['||W||', '||b||'])

plt.xscale('log')
plt.yscale('log')
plt.savefig('norms.png')



# %%
