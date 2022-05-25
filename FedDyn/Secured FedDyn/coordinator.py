from select import select
import threading
import numpy as np
import torch
import copy
import zmq
import tenseal as ts

from simpleModel import SimpleModel

class Coordinator (threading.Thread):
    def __init__(self, threadID, name, num_clients, num_selected, num_rounds, context, port, client_ports, alpha):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.num_clients = num_clients
        self.num_selected = num_selected
        self.num_rounds = num_rounds
        self.context = context
        self.port = port
        self.client_ports = client_ports
        self.alpha = alpha

        self.h = SimpleModel().state_dict()
        self.global_dict = SimpleModel().state_dict()
        # Initial encryption
        for key in self.global_dict:
            self.global_dict[key] = ts.ckks_tensor(self.context, self.global_dict[key])
        self.context.make_context_public()

    
    def compute_new_h(self, h_weights):
        """
        client_sum = copy.deepcopy(vendor_weights[0])
        prev_state_dict = self.global_dict
        for key in client_sum.keys():
            for i in range(1, len(vendor_weights)):
                client_sum[key] += (vendor_weights[i][key] - prev_state_dict[key])
        
        for key in self.h.keys():
            #print("attempt 1")
            temp_h = client_sum[key] * (self.alpha / self.num_clients)
            #print("1 success")
            #print("attempt 2")
            self.h[key] -= temp_h
            #print("2 success")
            #self.h[key] -= (self.alpha / self.num_clients) * client_sum[key]
            # Dirty hack to get around scaling error
            # temp = self.h[key].decrypt()
            # self.h[key] = ts.ckks_tensor(self.context, temp)
        """
        sum = copy.deepcopy(h_weights[0])
        for key in sum.keys():
            for i in range(1, len(h_weights)):
                sum[key] += h_weights[i][key]
        
        for key in self.h.keys():
            self.h[key] -= sum[key]


    def update_coord_model(self, vendor_weights):
        client_sum = copy.deepcopy(vendor_weights[0])
        coord_dict = self.global_dict
        for key in client_sum.keys():
            for i in range(1, len(vendor_weights)):
                client_sum[key] += vendor_weights[i][key]

        for key in coord_dict.keys():
            coord_dict[key] = ((1.0 / self.num_selected) * client_sum[key]) - ((1.0 / self.alpha) * self.h[key])
        self.global_dict = coord_dict


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
            temp_dict = copy.deepcopy(self.global_dict)
            for key in temp_dict:
                temp_dict[key] = temp_dict[key].serialize()
            for client in selected_clients:
                client_sockets[client].send_pyobj((0, temp_dict))

            loss = 0
            vendor_weights = []
            h_weights = []
            # Receive encrypted weights
            for client in selected_clients:
                enc_data = client_sockets[client].recv_pyobj()
                loss += enc_data[0]
                vendor_weights.append(enc_data[1])
                h_weights.append(enc_data[2])

            # Deserialise the encrypted weights
            for vendor_weight in vendor_weights:
                for key in vendor_weight:
                    temp = ts.ckks_tensor_from(self.context, vendor_weight[key])
                    vendor_weight[key] = temp
            
            for h_weight in h_weights:
                for key in h_weight:
                    temp = ts.ckks_tensor_from(self.context, h_weight[key])
                    h_weight[key] = temp

            self.compute_new_h(h_weights)
            self.update_coord_model(vendor_weights)

            # Receive test accuracy
            client_sockets[selected_clients[0]].send_pyobj((1, None))
            test_loss, test_accuracy = client_sockets[selected_clients[0]].recv_pyobj()

            print(f'test loss  {test_loss:.3g} | test acc: {test_accuracy:.3f}')
        
        for client_socket in client_sockets: client_socket.send_pyobj((2, None))
        print(f"Exiting {self.name}")