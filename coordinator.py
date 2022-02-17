import threading
import queue

from sympy import N

class Coordinator (threading.Thread):
    hist1 = None
    hist2 = None
    iteration = 0

    def __init__(self, threadID, name, context, histQ, mergedQ, lossQ, nVendors):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.context = context
        self.histQ = histQ
        self.mergedQ = mergedQ
        self.lossQ = lossQ
        self.nVendors = nVendors


    def merge_histograms(self, enc_hist1, enc_hist2):
        weight = self.compute_weight()
        add = enc_hist1 + enc_hist2
        return add


    def compute_weight(self):
        return 1.0 / self.iteration


    def run(self):
        print(f"Starting {self.name}")
        losses = []
        while True:
            enc_hist = self.histQ.get()
            self.set_histogram(enc_hist)
            self.iteration += 1
            if self.hist1 != None and self.hist2 != None:
                enc_merged_hist = self.merge_histograms(self.hist1, self.hist2)
                for i in range(self.nVendors):
                    self.mergedQ.put(enc_merged_hist)
                for i in range(self.nVendors):
                    losses.append(self.lossQ.get())
                sum_loss = 0
                for loss in losses:
                    sum_loss += loss
                avg_loss = sum_loss / self.nVendors
                print(avg_loss)
                if avg_loss <= 0.001:
                    print(f"Exiting {self.name}")
                    return
                losses = []
                self.hist1 = enc_merged_hist

            """"
            enc_hist2 = self.q.get()

            enc_merged_hist1 = self.merge_histograms(enc_hist1, enc_hist2)

            enc_hist3 = self.q.get()
            enc_merged_hist2 = self.merge_histograms(enc_merged_hist1, enc_hist3)

            diff = self.compare_histograms(enc_merged_hist1, enc_merged_hist2)
            sum_diff = self.sum_diffs(diff.decrypt())
            if sum_diff:
                print(f"Exiting {self.name}")
                return
            """
        print(f"Exiting {self.name}")
        

    def set_histogram(self, hist):
        if self.hist1 == None:
            self.hist1 = hist
        else:
            self.hist2 = hist

        """
        if self.hist1 != None and self.hist2 != None:
            self.hist1 = self.hist2
            self.hist2 = hist
        elif self.hist1 != None:
            self.hist2 = hist
        else:
            self.hist1 = hist
        """
            
    