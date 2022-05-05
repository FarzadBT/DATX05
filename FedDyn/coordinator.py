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
        for key in self.h:
            self.h[key] = torch.zeros(self.h[key].shape)
        self.global_model = SimpleModel()

    
    def compute_new_h(self, vendor_weights):
        client_sum = copy.deepcopy(vendor_weights[0])
        prev_state_dict = self.global_model.state_dict()
        for key in client_sum.keys():
            for i in range(1, len(vendor_weights)):
                client_sum[key] += (vendor_weights[i][key] - prev_state_dict[key])
            
        #delta_sum = {}
        #for key in client_sum.keys():
            #delta_sum[key] = client_sum[key] - prev_state_dict[key]

        for key in self.h.keys():
            self.h[key] -= (self.alpha / self.num_clients) * client_sum[key]


    def update_coord_model(self, vendor_weights):
        client_sum = copy.deepcopy(vendor_weights[0])
        coord_dict = self.global_model.state_dict()
        for key in client_sum.keys():
            for i in range(1, len(vendor_weights)):
                client_sum[key] += vendor_weights[i][key]

        for key in coord_dict.keys():
            coord_dict[key] = ((1.0 / self.num_selected) * client_sum[key]) - ((1.0 / self.alpha) * self.h[key])
        self.global_model.load_state_dict(coord_dict)


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
                client_sockets[client].send_pyobj((0, self.global_model.state_dict()))

            loss = 0
            vendor_weights = []
            # Receive encrypted weights
            for client in selected_clients:
                data = client_sockets[client].recv_pyobj()
                loss += data[0]
                vendor_weights.append(data[1])
            
            self.compute_new_h(vendor_weights)
            self.update_coord_model(vendor_weights)

            # Receive test accuracy
            client_sockets[selected_clients[0]].send_pyobj((2, None))
            test_loss, test_accuracy = client_sockets[selected_clients[0]].recv_pyobj()

            print(f'test loss  {test_loss:.3g} | test acc: {test_accuracy:.3f}')
        
        for client_socket in client_sockets: client_socket.send_pyobj((3, None))
        print(f"Exiting {self.name}")