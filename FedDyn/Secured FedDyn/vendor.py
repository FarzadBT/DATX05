import copy
import threading
import torch
import tenseal as ts
import zmq
from torch import nn
import torch.nn.functional as F

from simpleModel import SimpleModel

class Vendor (threading.Thread):
    

    def __init__(self, threadID, name, dataloader, epochs, test_loader, context, port, alpha):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.dataloader = dataloader
        self.epochs = epochs
        self.test_loader = test_loader
        self.context = context
        self.port = port
        self.alpha = alpha
        
        self.model = SimpleModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.CrossEntropyLoss()

        self.local_gradient = self.model.state_dict()
        for key in self.local_gradient:
            self.local_gradient[key] = torch.zeros(self.local_gradient[key].shape)
        self.prev_global_model = SimpleModel()

    def update_vendor(self):
        print(f"Training {self.name}")
        self.model.train()
        for _ in range(self.epochs):
            for data, target in self.dataloader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss = self.dynamic_regularization(loss)
                loss.backward()
                self.optimizer.step()

        self.update_local_gradient()

        enc_model_dict = copy.deepcopy(self.model.state_dict())
        for key in enc_model_dict:
            enc_model_weight = ts.ckks_tensor(self.context, enc_model_dict[key])
            enc_model_dict[key] = enc_model_weight.serialize()

        return (loss.item(), enc_model_dict)


    def dynamic_regularization(self, loss):
        first_penalty_term = 0.0

        model_dict = copy.deepcopy(self.model.state_dict())
        for key in model_dict:
            first_penalty_term += torch.sum(self.local_gradient[key] * model_dict[key])

        loss -= first_penalty_term

        second_penalty_term = 0.0
        prev_global_model_dict = self.prev_global_model.state_dict()
        for key in model_dict:
            second_penalty_term += torch.sum(((model_dict[key] - prev_global_model_dict[key])))

        loss += (second_penalty_term ** 2.0) * (self.alpha / 2.0)

        return loss


    def update_local_gradient(self):
        prev_global_state_dict = self.prev_global_model.state_dict()
        curr_state_dict = self.model.state_dict()
        for key in self.local_gradient.keys():
            self.local_gradient[key] -= self.alpha * (curr_state_dict[key] - prev_global_state_dict[key])


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
            
            if flag == 0: # Receive an updated global model, update own model, train and send weights
                enc_serialised_weights = data
                temp = copy.deepcopy(self.model.state_dict())
                for key in temp:
                    enc_weight = ts.ckks_tensor_from(self.context, enc_serialised_weights[key])
                    temp[key] = torch.Tensor(enc_weight.decrypt().tolist())
                self.prev_global_model.load_state_dict(temp)
                train_loss, enc_weights = self.update_vendor()
                coord_socket.send_pyobj((train_loss, enc_weights))

            if flag == 1: # return test data
                test_loss, test_acc = self.test()
                coord_socket.send_pyobj((test_loss, test_acc))

            if flag == 2: # exit
                break

        print(f"Exiting {self.name}")