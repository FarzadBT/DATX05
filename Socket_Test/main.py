import tenseal as ts

from server import Server
from client import Client


bits_scale = 26

ts_context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)
ts_context.global_scale = 2 ** bits_scale
ts_context.generate_galois_keys()

num_clients = 2

ports = [10000, 10001]
data = [[1,2,3,4,5,6,7,8,9], [9,8,7,6,5,4,3,2,1]]

server = Server(0, ports, ts_context)
clients = [Client(i + 1, ports[i], ts_context, data[i]) for i in range(num_clients)]

server.start()
for client in clients: client.start()
