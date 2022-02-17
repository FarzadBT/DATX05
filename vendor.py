import threading
import numpy as np
import queue
import tenseal as ts

class Vendor (threading.Thread):
    merged_hist = None
    rand = np.random.default_rng() 

    def __init__(self, threadID, name, context, bins, histQ, mergedQ, lossQ):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.context = context
        self.bins = bins
        self.histQ = histQ
        self.mergedQ = mergedQ
        self.lossQ = lossQ

    def run(self):
        print(f"Starting {self.name}")
        self.accumulate(self.context)
        print(f"Exiting {self.name}")

    
    def sum_diffs(self, hist_diffs):
        diff_sum = 0
        for diff in hist_diffs:
            diff_sum += abs(diff)
        return diff_sum

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
                self.histQ.put(enc_norm_hist)
                vals = []

                enc_merged_hist = self.mergedQ.get()
                merged_hist = enc_merged_hist.decrypt()
                if self.merged_hist == None:
                    self.merged_hist = merged_hist
                    self.lossQ.put(1.0)
                else:
                    hist_diffs = [x - y for x, y in zip(self.merged_hist, merged_hist)]#self.merged_hist - merged_hist
                    sum_diff = self.sum_diffs(hist_diffs)
                    self.lossQ.put(sum_diff)
                    self.merged_hist = merged_hist
    
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