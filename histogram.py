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

data1 = rng.normal(scale=(50/3), size=10000)
data1 = data1 - min(data1)
hist1, bin_edges1 = np.histogram(data1)

rng2 = np.random.default_rng()

data2 = rng2.normal(scale=(50/3), size=10000)
data2 = data2 - min(data2)
hist2, bin_edges2 = np.histogram(data2)

def normalise_histogram(hist):
    new_hist = []
    for i in range(len(hist)):
        val = hist[i] / 10000
        new_hist.append(val)
    return np.array(new_hist)

def normalise_histogram_prob(hist, bin_edges):
    new_hist = []
    for i in range(len(hist)):
        val = hist[i] / (10000 * (bin_edges[i + 1] - bin_edges[i]))
        new_hist.append(val)
    return np.array(new_hist)

def generate_histogram(scale, size):
    rand = np.random.default_rng()
    data = rand.normal(scale=scale, size=size)
    data = data - min(data)
    return np.histogram(data)

def encrypt_histogram(hist, bin_edges):
    enc_hist = ts.bfv_vector(context, hist)
    enc_bin_edges = ts.bfv_vector(context, bin_edges)
    return (enc_hist, enc_bin_edges)



print(hist1)
print(hist2)

    
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