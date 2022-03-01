import threading

class Coordinator (threading.Thread):
    """
    Coordinator for vendors, only handles sensitive data after encryption.
    Merges histograms provided by vendors and calculates average loss.
    """

    iteration = 0
    weight_coefficient = 2 # coefficient for determining weight for normalised merged histograms
    target_loss = 0.01

    def __init__(self, threadID, name, context, histQ, mergedQ, lossQ, nVendors):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.context = context
        self.histQ = histQ
        self.mergedQ = mergedQ
        self.lossQ = lossQ
        self.nVendors = nVendors

    # Merge two histogram together in a weighted manner
    def merge_histograms(self, enc_hist1, enc_hist2):
        weight = 1.0 / self.weight_coefficient
        add = enc_hist1 + (weight * enc_hist2)
        self.weight_coefficient += 1
        return add


    def run(self):
        print(f"Starting {self.name}")
        losses = [] # Storage for losses arriving from vendors
        hists = [] # Storage for histograms arriving from vendors
        while True:
            self.iteration += 1
            
            # Get encrypted histograms from vendors
            for i in range(self.nVendors): 
                enc_hist = self.histQ.get()
                hists.append(enc_hist)

            if len(hists) >= 2: # If there are enough histograms to merge

                # Merge all histograms in hists
                enc_merged_hist = self.merge_histograms(hists[0], hists[1])
                for i in range(2, len(hists) - 1):
                    enc_merged_hist = self.merge_histograms(enc_merged_hist, hists[i])
                
                # Send the encrypted merged histogram back to vendors
                for i in range(self.nVendors): 
                    self.mergedQ.put(enc_merged_hist)

                # Get the loss back from vendors
                for i in range(self.nVendors): 
                    losses.append(self.lossQ.get())

                # Compute average loss
                sum_loss = 0
                for loss in losses:
                    sum_loss += loss
                avg_loss = sum_loss / self.nVendors
                print(f"Iteration {self.iteration}: {avg_loss}")
                
                # If loss target has been reached
                if avg_loss <= self.target_loss: 
                    print(f"Exiting {self.name}")
                    return
                
                # Reset for the next iteration
                losses = []
                hists = [enc_merged_hist]
    