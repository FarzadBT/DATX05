from re import M
import numpy as np
import matplotlib.pyplot as plt
import tenseal as ts

rng = np.random.default_rng()

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40

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


def generate_histogram(scale, size, loc=0, bins=10):
    rand = np.random.default_rng()
    data = rand.normal(loc=loc, scale=(scale/3), size=size)
    data = data - min(data)
    return np.histogram(data, bins=bins)

"""
def encrypt_histogram(hist, bins):
    enc_hist = ts.ckks_vector(context, hist)
    enc_bins = ts.ckks_vector(context, bins)
    return (enc_hist, enc_bins)
"""

"""
def decrypt_histogram(enc_hist, enc_bins):
    hist = enc_hist.decrypt()
    bins = enc_bins.decrypt()
    return (hist, bins)
"""

def merge_histograms(enc_hist1, enc_hist2):
    add = enc_hist1 + enc_hist2
    return add * 0.5
    

def compare_histograms(enc_hist1, enc_hist2):
    diff = enc_hist1 - enc_hist2
    return diff


def sum_diffs(bin_diffs):
    diff_sum = 0
    for diff in bin_diffs:
        diff_sum += diff

    return (diff_sum <= 0.0001)


bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

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
enc_diff_hist = compare_histograms(enc_hist1, enc_hist2)
diff_hist = enc_diff_hist.decrypt()
print(diff_hist)


    
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