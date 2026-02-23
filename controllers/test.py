import numpy as np

weights = -np.arange(10)*0.1
weights = np.exp(weights)
weights /= weights.sum()
print(weights)