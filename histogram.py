from re import M
import numpy as np
import matplotlib.pyplot as plt
import tenseal as ts
import queue

from vendor import Vendor
from coordinator import Coordinator

rng = np.random.default_rng()

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40

def normalise_histogram_prob(hist, bin_edges):
    new_hist = []
    for i in range(len(hist)):
        val = hist[i] / (10000 * (bin_edges[i + 1] - bin_edges[i]))
        new_hist.append(val)
    return np.array(new_hist)


def generate_histogram(scale, size, loc=0, bins=10):
    rand = np.random.default_rng()
    data = rand.normal(loc=loc, scale=(scale/3), size=size)
    data = data - min(data)
    return np.histogram(data, bins=bins)

def merge_histograms(enc_hist1, enc_hist2):
    add = enc_hist1 + enc_hist2
    return add * 0.5
    

def compare_histograms(enc_hist1, enc_hist2):
    diff = enc_hist1 - enc_hist2
    return diff

bin_edges = [x for x in range(121) if x % 5 == 0] # [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]

q = queue.Queue(100)

vendor1 = Vendor(1, "Vendor-1", context, bin_edges, q)
vendor2 = Vendor(2, "Vendor-2", context, bin_edges, q)
vendor3 = Vendor(3, "Vendor-3", context, bin_edges, q)

coordinator = Coordinator(4, "Coordinator", context, q)

vendor1.start()
vendor2.start()
vendor3.start()
coordinator.start()

"""
# Coordinator



# Vendors
hist1, _ = generate_histogram(50, 10000, bins=bin_edges)
norm_hist1 = normalise_histogram(hist1)

hist2, _ = generate_histogram(50, 10000, bins=bin_edges)
norm_hist2 = normalise_histogram(hist2)

print(hist1)
print(hist2)
print(norm_hist1)
print(norm_hist2)

enc_hist1 = ts.ckks_vector(context, norm_hist1)
enc_hist2 = ts.ckks_vector(context, norm_hist2)
#enc_merged_hist = merge_histograms(enc_hist1, enc_hist2)
#merged_hist = enc_merged_hist.decrypt()


# Coordinator
enc_diff_hist = compare_histograms(enc_hist1, enc_hist2)
diff_hist = enc_diff_hist.decrypt()
print(sum_diffs(diff_hist))
"""