import copy
import threading
import torch
import tenseal as ts
from torch import nn
import torch.nn.functional as F

from simpleModel import SimpleModel

class Vendor (threading.Thread):
    

    def __init__(self, threadID, name, receiveQueue, sendQueue, dataloader, epochs, context, test_loader):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.receiveQueue = receiveQueue
        self.sendQueue = sendQueue
        self.dataloader = dataloader
        self.epochs = epochs
        self.context = context
        self.test_loader = test_loader
        
        self.model = SimpleModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.CrossEntropyLoss()
    

    
    # Model training
    def update_vendor(self):
        print(f"Training {self.name}")
        self.model.train()
        for _ in range(self.epochs):
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

    #This function test the global model on test data and returns test loss and test accuracy
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        acc = correct / len(self.test_loader.dataset)

        return test_loss, acc



    # Main loop
    def run(self):
        print(f"Starting {self.name}")
        while(True):
            (flag, temp) = self.receiveQueue.get()
            if flag == 1: # if update requested
                train_loss, enc_model_dict = self.update_vendor()
                self.sendQueue.put((train_loss, enc_model_dict))
            elif flag == 2: # if get state_dict
                for key in temp:
                    temp[key] = torch.Tensor(temp[key].decrypt().tolist())
                self.model.load_state_dict(temp)
            elif flag == 3: # if test requested
                tuple = self.test()
                self.sendQueue.put(tuple)
            elif flag == 4: # if shutdown requested
                break
            else: # if not a valid flag
                raise Exception("Vendor not given valid flag")


        print(f"Exiting {self.name}")