import threading
import queue

from sympy import N

class Coordinator (threading.Thread):
    hist1 = None
    hist2 = None

    def __init__(self, threadID, name, context, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.context = context
        self.q = q


    def merge_histograms(self, enc_hist1, enc_hist2):
        add = enc_hist1 + enc_hist2
        return add * 0.5

    def compare_histograms(self, enc_hist1, enc_hist2):
        diff = enc_hist1 - enc_hist2
        return diff
        

    def sum_diffs(self, hist_diffs):
        diff_sum = 0
        for diff in hist_diffs:
            diff_sum += abs(diff)
        print(diff_sum)
        return (diff_sum <= 0.001)


    def run(self):
        print(f"Starting {self.name}")
        while True:
            enc_hist = self.q.get()
            self.set_histogram(enc_hist)
            if self.hist1 != None and self.hist2 != None:
                enc_merged_hist = self.merge_histograms(self.hist1, self.hist2)
                diff = self.compare_histograms(self.hist1, enc_merged_hist)
                if self.sum_diffs(diff.decrypt()):
                    print(f"Exiting {self.name}")
                    return
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
            
    