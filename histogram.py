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


bin_edges = [x for x in range(121)]

histQ = queue.Queue(100)
mergedQ = queue.Queue()
lossQ = queue.Queue()

vendor1 = Vendor(1, "Vendor-1", context, bin_edges, histQ, mergedQ, lossQ)
vendor2 = Vendor(2, "Vendor-2", context, bin_edges, histQ, mergedQ, lossQ)
vendor3 = Vendor(3, "Vendor-3", context, bin_edges, histQ, mergedQ, lossQ)

coordinator = Coordinator(4, "Coordinator", context, histQ, mergedQ, lossQ, 3)

vendor1.start()
vendor2.start()
vendor3.start()
coordinator.start()