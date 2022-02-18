import threading
import numpy as np
import tenseal as ts

class Vendor (threading.Thread):
    """
    Vendors generate an encrypted histogram of data which is sent to the coordinator.
    Vendors receive back an encrypted merged histogram and can decrypt it to calculate loss.
    """

    nVals = 10000 # Amount of values to generate
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
    
    # Sums the differances through absolute value
    def sum_diffs(self, hist_diffs):
        diff_sum = 0
        for diff in hist_diffs:
            diff_sum += abs(diff)
        return diff_sum
    
    # Outputs a value through normal distribution, re-generates the value if it's negative or out of scale
    def singleNormal(self, loc, scale):
        val = int(self.rand.normal(loc=loc, scale=scale/3))
        if val < 0 or val > loc+scale:
            return self.singleNormal(loc, scale)
        else:
            return val
    

    # Normalise the histogram to the number of values in the histogram
    def normalise_histogram(self, hist, nVals):
        new_hist = []
        for i in range(len(hist)):
            val = hist[i] / nVals
            new_hist.append(val)
        return np.array(new_hist)

    def accumulate(self, context):
        vals = []
        merged_hist = None
        loc = np.random.randint(30, 91)
        scale = np.random.randint(10, 31)

        while True:
            # Generates nVals values, build a histogram, encrypt it, and send it to coordinator
            for i in range(self.nVals):
                vals.append(self.singleNormal(loc, scale))

            hist, _ = np.histogram(vals, bins=self.bins)
            norm_hist = self.normalise_histogram(hist, self.nVals)
            enc_norm_hist = ts.ckks_vector(context, norm_hist)
            self.histQ.put(enc_norm_hist)
            vals = []

            # Get merged histogram from coordinator, decrypt, compute loss, and send it back to coordinator
            enc_merged_hist = self.mergedQ.get()
            new_merged_hist = enc_merged_hist.decrypt()
            if merged_hist == None:
                merged_hist = new_merged_hist
                self.lossQ.put(1.0)
            else:
                hist_diffs = [x - y for x, y in zip(merged_hist, new_merged_hist)]
                sum_diff = self.sum_diffs(hist_diffs)
                self.lossQ.put(sum_diff)
                merged_hist = new_merged_hist
    
    def run(self):
        print(f"Starting {self.name}")
        self.accumulate(self.context)
        print(f"Exiting {self.name}")