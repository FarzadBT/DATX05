from re import M
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

data = rng.normal(scale=(50/3), size=1000)
data = data - min(data)
numpy_hist = np.histogram(data, "auto")
hist = plt.hist(data, bins="auto")
print(hist)
plt.show()