from http import server
import zmq
import tenseal as ts
import threading
import copy


class Server (threading.Thread):

    def __init__(self, threadID, client_ports, ts_context):
        threading.Thread.__init__(self)
        self.threadID = threadID,
        self.client_ports = client_ports,
        self.ts_context = ts_context
        self.zmq_context = zmq.Context()


    def add_data(self, data):
        temp = copy.deepcopy(data[0])
        for i in range(1, len(data)):
            temp += data[i]
        
        return temp


    def run(self):
        client_sockets = []
        for client_port in self.client_ports[0]:
            client_socket = self.zmq_context.socket(zmq.REP)
            client_socket.bind(f"tcp://*:{client_port}")
            client_sockets.append(client_socket)

        enc_data = []
        for client_socket in client_sockets:
            enc_serial = client_socket.recv()
            enc_d = ts.ckks_tensor_from(self.ts_context, enc_serial)
            enc_data.append(enc_d)

        enc_result = self.add_data(enc_data)
        for client_socket in client_sockets:
            client_socket.send(enc_result.serialize())
    