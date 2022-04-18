import copy
import threading
import torch
import tenseal as ts
import zmq
from torch import nn
import torch.nn.functional as F

from simpleModel import SimpleModel

class Vendor (threading.Thread):
    

    def __init__(self, threadID, name, receiveQueue, sendQueue, dataloader, epochs, context, test_loader, port):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.receiveQueue = receiveQueue
        self.sendQueue = sendQueue
        self.dataloader = dataloader
        self.epochs = epochs
        self.context = context
        self.test_loader = test_loader
        self.port = port
        
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
            model_dict[key] = enc_weight.serialize()

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
            # Will be blocked by recv() until participating in the training
            flag = coord_socket.recv_string()
            if flag == "1":
                break

            train_loss, encrypted_weights = self.socketed_update_vendor()
            coord_socket.send_pyobj((train_loss, encrypted_weights))

            enc_serialised_weights = coord_socket.recv_pyobj()
            temp = copy.deepcopy(self.model.state_dict())
            for key in temp:
                enc_weight = ts.ckks_tensor_from(self.context, enc_serialised_weights[key])
                temp[key] = torch.Tensor(enc_weight.decrypt().tolist())
            self.model.load_state_dict(temp)

            test_loss, test_acc = self.test()
            coord_socket.send_pyobj((test_loss, test_acc))
        
        print(f"Exiting {self.name}")


    # Main loop
    def normal_run(self):
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