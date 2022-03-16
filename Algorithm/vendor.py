import copy
import threading
import torch
import tenseal as ts
from torch import nn

from simpleModel import SimpleModel

class Vendor (threading.Thread):
    

    def __init__(self, threadID, name, receiveQueue, sendQueue, dataloader, epochs, context):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.receiveQueue = receiveQueue
        self.sendQueue = sendQueue
        self.dataloader = dataloader
        self.epochs = epochs
        self.context = context
        
        self.model = SimpleModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.NLLLoss()
    
    
    def update_vendor(self):
        print(f"Training {self.name}")
        self.model.train()
        for e in range(self.epochs):
            for data, target in self.dataloader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()

        model_dict = copy.deepcopy(self.model.state_dict())
        for key in model_dict:
            model_dict[key] = ts.ckks_tensor(self.context, model_dict[key])

        return (loss.item(), model_dict)



    def run(self):
        print(f"Starting {self.name}")
        while(True):
            temp = self.receiveQueue.get()
            if temp == None:
                loss, enc_model_dict = self.update_vendor()
                self.sendQueue.put((loss, enc_model_dict))
            else: # temp is a state_dict
                for key in temp:
                    temp[key] = torch.Tensor(temp[key].decrypt().tolist())
                self.model.load_state_dict(temp)


        print(f"Exiting {self.name}")