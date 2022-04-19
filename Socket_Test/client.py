import threading
import zmq
import tenseal as ts


class Client (threading.Thread):

    def __init__(self, threadID, server_port, ts_context, data):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.server_port = server_port
        self.ts_context = ts_context
        self.data = data
        self.zmq_context = zmq.Context()

    def run(self):
        enc_a = ts.ckks_tensor(self.ts_context, self.data)
        
        print("Connecting to server")
        socket = self.zmq_context.socket(zmq.REQ)
        socket.connect(f"tcp://localhost:{self.server_port}")

        print("Sending encrypted data")
        socket.send(enc_a.serialize())

        results_reply = socket.recv()
        enc_result = ts.ckks_tensor_from(self.ts_context, results_reply)
        print(f"Received result {enc_result.decrypt().tolist()}")
        