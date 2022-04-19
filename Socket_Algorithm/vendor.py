import copy
import threading
import torch
import tenseal as ts
import zmq
from torch import nn
import torch.nn.functional as F

from simpleModel import SimpleModel

class Vendor (threading.Thread):
    

    def __init__(self, threadID, name, dataloader, epochs, context, test_loader, port):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.dataloader = dataloader
        self.epochs = epochs
        self.context = context
        self.test_loader = test_loader
        self.port = port
        
        self.model = SimpleModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.CrossEntropyLoss()


    def socketed_update_vendor(self):
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
            enc_weight = ts.ckks_tensor(self.context, model_dict[key])
            model_dict[key] = enc_weight.serialize() # in socket programming, data is serialized to allow for a better transfer

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


    def run(self):
        print(f"Starting {self.name}")
        zmq_context = zmq.Context()

        coord_socket = zmq_context.socket(zmq.REP)
        coord_socket.bind(f"tcp://*:{self.port}")

        while(True):
            # Will be blocked by recv() until called for
            flag, data = coord_socket.recv_pyobj()
            if flag == 0: # Train and return encrypted weights
                train_loss, encrypted_weights = self.socketed_update_vendor()
                coord_socket.send_pyobj((train_loss, encrypted_weights))
            if flag == 1: # Receive (averaged) encrypted weights and update model
                enc_serialised_weights = data
                temp = copy.deepcopy(self.model.state_dict())
                for key in temp:
                    enc_weight = ts.ckks_tensor_from(self.context, enc_serialised_weights[key])
                    temp[key] = torch.Tensor(enc_weight.decrypt().tolist())
                self.model.load_state_dict(temp)
                coord_socket.send_string("Updated")
            if flag == 2: # return test data
                test_loss, test_acc = self.test()
                coord_socket.send_pyobj((test_loss, test_acc))
            if flag == 3: # exit
                break

        print(f"Exiting {self.name}")