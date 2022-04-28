from select import select
import threading
import numpy as np
import torch
import copy
import zmq
import tenseal as ts

from simpleModel import SimpleModel

class Coordinator (threading.Thread):
    def __init__(self, threadID, name, num_clients, num_selected, num_rounds, port, client_ports):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.num_clients = num_clients
        self.num_selected = num_selected
        self.num_rounds = num_rounds
        self.port = port
        self.client_ports = client_ports


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
                enc_tuple = client_sockets[client].recv_pyobj()
                loss += enc_tuple[0]
                vendor_weights.append(enc_tuple[1])
            
            avg_loss = loss / self.num_selected
            if (avg_loss < 0.01):
                print(f"We're done with avg loss {avg_loss}")
                break
            
            # Average and distribute weights
            new_enc_weights = self.socketed_average_weights(vendor_weights)
            for client in client_sockets:
                client.send_pyobj((1, new_enc_weights))
            for client in client_sockets:
                client.recv_string()

            # Receive test accuracy
            client_sockets[0].send_pyobj((2, None))
            test_loss, test_accuracy = client_sockets[0].recv_pyobj()

            print(f'average train loss {avg_loss:.3g} | test loss  {test_loss:.3g} | test acc: {test_accuracy:.3f}')
        
        for client_socket in client_sockets: client_socket.send_pyobj((3, None))
        print(f"Exiting {self.name}")