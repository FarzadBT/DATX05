import threading
import numpy as np
import queue
import tenseal as ts

class Vendor (threading.Thread):
    def __init__(self, threadID, name, context, bins, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.context = context
        self.bins = bins
        self.q = q

        self.rand = np.random.default_rng() 

    def run(self):
        print(f"Starting {self.name}")
        self.accumulate(self.context)
        print(f"Exiting {self.name}")

    def accumulate(self, context):
        vals = []
        loc = np.random.randint(30, 91)
        scale = np.random.randint(10, 31)

        while True:
            vals.append(self.singleNormal(loc, scale))
            if len(vals) >= 10000:
                hist, _ = np.histogram(vals, bins=self.bins)
                norm_hist = self.normalise_histogram(hist)
                enc_norm_hist = ts.ckks_vector(context, norm_hist)
                self.q.put(enc_norm_hist)
                vals = []
    
    def singleNormal(self, loc, scale):
        val = int(self.rand.normal(loc=loc, scale=scale/3))
        if val < 0 or val > loc+scale:
            return self.singleNormal(loc, scale)
        else:
            return val
    
    def normalise_histogram(self, hist):
        new_hist = []
        for i in range(len(hist)):
            val = hist[i] / 10000
            new_hist.append(val)
        return np.array(new_hist)