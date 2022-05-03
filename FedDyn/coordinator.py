from select import select
import threading
import numpy as np
import torch
import copy
import zmq
import tenseal as ts

from simpleModel import SimpleModel

class Coordinator (threading.Thread):
    def __init__(self, threadID, name, num_clients, num_selected, num_rounds, port, client_ports, alpha):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.num_clients = num_clients
        self.num_selected = num_selected
        self.num_rounds = num_rounds
        self.port = port
        self.client_ports = client_ports
        self.alpha = alpha

        self.h = SimpleModel().state_dict()
        self.model = SimpleModel()

    
    def compute_new_h(self, vendor_weights):
        client_sum = copy.deepcopy(vendor_weights[0])
        prev_state_dict = self.model.state_dict()
        for key in client_sum.keys():
            for i in range(1, len(vendor_weights)):
                client_sum[key] += vendor_weights[i][key]
            client_sum[key] -= prev_state_dict[key]

        new_h = copy.deepcopy(self.h)
        for key in self.h.keys():
            new_h[key] = self.h[key] - (self.alpha * (1/self.num_clients)) * client_sum[key]
        self.h = new_h


    def update_coord_model(self, vendor_weights):
        client_sum = copy.deepcopy(vendor_weights[0])
        coord_dict = self.model.state_dict()
        for key in client_sum.keys():
            for i in range(1, len(vendor_weights)):
                client_sum[key] += vendor_weights[i][key]
        for key in coord_dict.keys():
            coord_dict[key] = (1 / self.num_selected) * client_sum[key] - (1 / self.alpha)
        self.model.load_state_dict(coord_dict)


    def average_weights(self, vendor_weights):
        w_avg = copy.deepcopy(vendor_weights[0])

        for key in w_avg.keys():
            for i in range(1, len(vendor_weights)):
                w_avg[key] += vendor_weights[i][key]
            w_avg[key] = w_avg[key] * (1/len(vendor_weights))

        return w_avg


    def run(self):
        client_sockets = []
        zmq_context = zmq.Context()
        for client_port in self.client_ports:
            client_socket = zmq_context.socket(zmq.REQ)
            client_sockets.append(client_socket)
            client_socket.connect(f"tcp://localhost:{client_port}")

        for r in range(self.num_rounds):
            print(f"Round {r + 1}")

            selected_clients = np.random.permutation(self.num_clients)[:self.num_selected]

            # Request encrypted weights
            for client in selected_clients:
                client_sockets[client].send_pyobj((0, None))

            loss = 0
            vendor_weights = []
            # Receive encrypted weights
            for client in selected_clients:
                data = client_sockets[client].recv_pyobj()
                loss += data[0]
                vendor_weights.append(data[1])
            
            avg_loss = loss / self.num_selected
            #if (avg_loss < 0.01):
            #    print(f"We're done with avg loss {avg_loss}")
            #    break
            
            # Average and distribute weights
            new_weights = self.average_weights(vendor_weights)
            for client in client_sockets:
                client.send_pyobj((1, new_weights))
            for client in client_sockets:
                client.recv_string()

            # Receive test accuracy
            client_sockets[0].send_pyobj((2, None))
            test_loss, test_accuracy = client_sockets[0].recv_pyobj()

            print(f'average train loss {avg_loss:.3g} | test loss  {test_loss:.3g} | test acc: {test_accuracy:.3f}')
        
        for client_socket in client_sockets: client_socket.send_pyobj((3, None))
        print(f"Exiting {self.name}")