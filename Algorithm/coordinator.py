import threading
import numpy as np
import torch
import copy
import torch.nn.functional as F

from simpleModel import SimpleModel

class Coordinator (threading.Thread):
    def __init__(self, threadID, name, num_clients, num_selected, num_rounds, send_queues, receive_queue, test_loader):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.num_clients = num_clients
        self.num_selected = num_selected
        self.num_rounds = num_rounds
        self.send_queues = send_queues
        self.receive_queue = receive_queue

        self.test_loader = test_loader  



    def average_weights(self, vendor_weights):
        w_avg = copy.deepcopy(vendor_weights[0])

        for key in w_avg.keys():
            for i in range(1, len(vendor_weights)):
                w_avg[key] += vendor_weights[i][key]
            w_avg[key] = w_avg[key] * (1/len(vendor_weights))

        return w_avg


    def test(self):
        #This function test the global model on test data and returns test loss and test accuracy
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.cuda(), target.cuda()
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        acc = correct / len(self.test_loader.dataset)

        return test_loss, acc



    def run(self):
        print(f"Starting {self.name}")

        for r in range(self.num_rounds):
            print(f"Round {r+1}")

            selected_clients = np.random.permutation(self.num_clients)[:self.num_selected]

            loss = 0
            vendor_weights = []
            for client in selected_clients:
                self.send_queues[client].put(None)

            for client in selected_clients:
                (temp_loss, enc_weight) = self.receive_queue.get()
                loss += temp_loss
                vendor_weights.append(enc_weight)
            
            avg_loss = np.abs(loss / self.num_selected)
            if (avg_loss < 0.01) :
                print(f"We're done with avg loss {avg_loss}")
                break
            
            new_enc_weights = self.average_weights(vendor_weights)
            
            for queue in self.send_queues:
                queue.put(copy.deepcopy(new_enc_weights))

            print(f'average train loss {avg_loss:.3g}')

        print(f"Exiting {self.name}")
    