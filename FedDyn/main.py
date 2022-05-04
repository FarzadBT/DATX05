import queue
import numpy as np
import pandas as pd
import tenseal as ts
from sklearn.model_selection import train_test_split
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset

from vendor import Vendor
from coordinator import Coordinator

"""
# controls precision of the fractional part
bits_scale = 26

# Create TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)

# set the scale
context.global_scale = pow(2, bits_scale)

# galois keys are required to do ciphertext rotations
context.generate_galois_keys()

# public context for coordinator, lacks the ability to decrypt
pub_context = context.copy()
pub_context.make_context_public()
"""

# Port number
port = 10000

# Parameters
num_clients = 3
num_selected = 2
num_rounds = 10
epochs = 10
alpha = 0.01

# Preprocessing
def encode_species(species):
    if species == 'setosa':
        return 0
    if species == 'versicolor':
        return 1
    if species == 'virginica':
        return 2
    else:
        raise ValueError('Species \'{}\' is not recognized.'.format(species))

dataset = pd.read_csv("iris.csv")

x = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = np.array(list(map(encode_species, dataset['species'].values)))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

torch_x = torch.tensor(x_train, dtype=torch.float32)
torch_y = torch.tensor(y_train, dtype=torch.int64)
train_dataset = TensorDataset(torch_x, torch_y)
train_data_split = random_split(train_dataset, [int(len(train_dataset) / num_clients) for _ in range(num_clients)])
train_data_loaders = [DataLoader(x, batch_size=32, shuffle=True) for x in train_data_split]

torch_x = torch.tensor(x_test, dtype=torch.float32)
torch_y = torch.tensor(y_test, dtype=torch.int64)
test_dataset = TensorDataset(torch_x, torch_y)
test_loader = DataLoader(test_dataset, batch_size=32)

# Prepare and create processes
client_ports = [(port + i) for i in range(num_clients)]

coordinator = Coordinator(0, "Coordinator", num_clients, num_selected, num_rounds, port, client_ports, alpha)

vendors = [Vendor(i+1, f"Vendor-{i+1}", train_data_loaders[i], epochs, test_loader, client_ports[i], alpha) for i in range(num_clients)]

# Start processes
for vendor in vendors: vendor.start()
coordinator.start()