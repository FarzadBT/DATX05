from re import M
import numpy as np
import matplotlib.pyplot as plt
import tenseal as ts

rng = np.random.default_rng()

context = ts.context(
    ts.SCHEME_TYPE.BFV,
    poly_modulus_degree=4096,
    plain_modulus=1032193
)
context.generate_galois_keys()
context.global_scale = 2**40

data = rng.normal(scale=(50/3), size=1000)
data = data - min(data)
numpy_hist = np.histogram(data, bins=10, density=True)

hist_count = numpy_hist[0]
hist_range = numpy_hist[1]
print(np.sum(hist_count))
"""
normalized_hist_count = []
for i in range(len(hist_count)):
    val = hist_count[i] / 1000 * (hist_range[i+1] - hist_range[i])
    normalized_hist_count.append(val)
normalized_hist_count = np.array(normalized_hist_count)
print(hist_count)
print(normalized_hist_count)


test1 = [1,2,3,4,5]
test2 = [9,8,7,6,5]
enc1 = ts.bfv_vector(context, test1)
enc2 = ts.bfv_vector(context, test2)

result = enc1 + enc2

print(result.decrypt())
"""