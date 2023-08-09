import numpy as np


# randam array with numpy szie (512,100)
a = np.random.rand(512,10000)

# apply svd
U, S, V = np.linalg.svd(a, full_matrices=False)
print(S)

energy = np.cumsum(S)/np.sum(S)
n_components = np.where(energy>0.95)[0][0]
print(energy)
print(n_components)

print(1131.62567495 / np.sum(S))