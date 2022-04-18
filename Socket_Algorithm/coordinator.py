from select import select
import threading
import numpy as np
import torch
import copy
import zmq
import tenseal as ts

from simpleModel import SimpleModel

class Coordinator (threading.Thread):
    def __init__(self, threadID, name, num_clients, num_selected, num_rounds, send_queues, receive_queue, port, client_ports, ts_context):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.num_clients = num_clients
        self.num_selected = num_selected
        self.num_rounds = num_rounds
        self.send_queues = send_queues
        self.receive_queue = receive_queue
        self.port = port
        self.client_ports = client_ports
        self.ts_context = ts_context


    # Given a list of (encrypted) weight dictionaries, averages the weights and returns the average
    def average_weights(self, vendor_weights):
        w_avg = copy.deepcopy(vendor_weights[0])

        for key in w_avg.keys():
            for i in range(1, len(vendor_weights)):
                w_avg[key] += vendor_weights[i][key]
            w_avg[key] = w_avg[key] * (1/len(vendor_weights))

        return w_avg


    # Given a list of encrypted weights, averages the weights and returns the average
    def socketed_average_weights(self, vendor_weights):
        temp_dict = copy.deepcopy(vendor_weights[0])
        for key in temp_dict.keys():
            temp = ts.ckks_tensor_from(self.ts_context, temp_dict[key])
            for i in range(1, len(vendor_weights)):
                temp += ts.ckks_tensor_from(self.ts_context, vendor_weights[i][key])
            temp *= (1 / len(vendor_weights))
            temp_dict[key] = temp.serialize()          

        return temp_dict


    def run(self):
        client_sockets = []
        zmq_context = zmq.Context()
        for client_port in self.client_ports:
            client_socket = zmq_context.socket(zmq.REQ)
            #client_socket.bind(f"tcp://localhost:{client_port}")
            client_sockets.append(client_socket)
            client_socket.connect(f"tcp://localhost:{client_port}")

        for r in range(self.num_rounds):
            print(f"Round {r + 1}")

            selected_clients = np.random.permutation(self.num_clients)[:self.num_selected]

            # Request encrypted weights
            for client in selected_clients:
                client_sockets[client].send_string("0")

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

            new_enc_weights = self.socketed_average_weights(vendor_weights)

            # Distribute weighted average weights
            for client in selected_clients:
                client_sockets[client].send_pyobj(new_enc_weights)

            test_losses = []
            test_accuracies = []
            # Receive test accuracy
            for client in selected_clients:
               test_loss, test_accuracy = client_sockets[client].recv_pyobj()
               test_losses.append(test_loss)
               test_accuracies.append(test_accuracy)
            
            avg_test_loss = sum(test_losses) / len(selected_clients)
            avg_acc = sum(test_accuracies) / len(selected_clients)

            print(f'average train loss {avg_loss:.3g} | test loss  {avg_test_loss:.3g} | test acc: {avg_acc:.3f}')
        
        for client_socket in client_sockets: client_socket.send_string("1")
        print(f"Exiting {self.name}")
         

    def normal_run(self):
        print(f"Starting {self.name}")

        for r in range(self.num_rounds):
            print(f"Round {r+1}")

            selected_clients = np.random.permutation(self.num_clients)[:self.num_selected]

            loss = 0
            vendor_weights = []
            # Sends signal to client to perform training
            for client in selected_clients: self.send_queues[client].put((1, None))

            # Retrieve the encrypted client models and training losses
            for client in selected_clients:
                (temp_loss, enc_weight) = self.receive_queue.get()
                loss += temp_loss
                vendor_weights.append(enc_weight)

            avg_loss = loss / self.num_selected
            if (avg_loss < 0.01):
                print(f"We're done with avg loss {avg_loss}")
                break
            
            # Aggregate and compute the weighted average model
            new_enc_weights = self.average_weights(vendor_weights)
            
            # Send the aggregated model back to the clients
            for queue in self.send_queues: queue.put((2, copy.deepcopy(new_enc_weights)))
            
            # Retrieve test results of a vendor model and print stats
            self.send_queues[0].put((3, None))
            (test_loss, acc) = self.receive_queue.get()
            print(f'average train loss {avg_loss:.3g} | test loss  {test_loss:.3g} | test acc: {acc:.3f}')

        for queue in self.send_queues: queue.put((4, None))
        print(f"Exiting {self.name}")
    