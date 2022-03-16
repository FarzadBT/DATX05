import tenseal as ts
import copy
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split

from simpleModel import SimpleModel

## Encryption Parameters

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


##### Hyperparameters for federated learning #########
num_clients = 3
num_selected = 2
num_rounds = 200
epochs = 10
batch_size = 1024


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

dataset = pd.read_csv("Algorithm/iris.csv")

x = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = np.array(list(map(encode_species, dataset['species'].values)))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

torch_x = torch.tensor(x_train, dtype=torch.float32)
torch_y = torch.tensor(y_train, dtype=torch.int64)
t_dataset = TensorDataset(torch_x, torch_y)
t_data_split = random_split(t_dataset, [int(len(t_dataset) / num_clients) for _ in range(num_clients)])
t_data_loaders = [DataLoader(x, batch_size=32, shuffle=True) for x in t_data_split]

torch_x = torch.tensor(x_test, dtype=torch.float32)
torch_y = torch.tensor(y_test, dtype=torch.int64)
test_dataset = TensorDataset(torch_x, torch_y)
test_loader = DataLoader(test_dataset, batch_size=32)


# Helper functions
def update_vendor(client_model, optimizer, train_loader, loss_fn, epoch=5):
    client_model.train()
    for e in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = client_model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

    client_dict = copy.deepcopy(client_model.state_dict())
    for key in client_dict:
        client_dict[key] = ts.ckks_tensor(context, client_dict[key])

    return (loss.item(), client_dict)

def average_weights(global_model, vendor_models):
    w_avg = copy.deepcopy(vendor_models[0].state_dict())

    for key in w_avg.keys():
        for i in range(1, len(vendor_models)):
            w_avg[key] += vendor_models[i].state_dict()[key]
        w_avg[key] = torch.div(w_avg[key], len(vendor_models))

    global_model.load_state_dict(w_avg)
    for model in vendor_models:
        model.load_state_dict(global_model.state_dict())

def average_enc_weights(vendor_weights):
    w_avg = copy.deepcopy(vendor_weights[0])

    for key in w_avg.keys():
        for i in range(1, len(vendor_weights)):
            w_avg[key] += vendor_weights[i][key]
        w_avg[key] = w_avg[key] * (1/len(vendor_weights))

    return w_avg

def test(global_model, test_loader):
    #This function test the global model on test data and returns test loss and test accuracy
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = global_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc



global_model = SimpleModel()

vendor_models = [SimpleModel() for _ in range(num_clients)]
for model in vendor_models:
    model.load_state_dict(global_model.state_dict())

vendor_optimizers = [torch.optim.Adam(model.parameters(), lr=0.01) for _ in range(num_clients)]

loss_fns = [nn.CrossEntropyLoss() for _ in range(num_clients)]

for r in range(num_rounds):
    print(f"Round {r+1}")

    selected_clients = np.random.permutation(num_clients)[:num_selected]

    loss = 0
    enc_vendor_weights = []
    for i in range(num_selected):
        #print(f"Vendor {i+1} update")
        client = selected_clients[i]
        (tempLoss, tempEnc) = update_vendor(vendor_models[client], vendor_optimizers[client], t_data_loaders[client], loss_fns[client], epochs)
        loss += tempLoss
        enc_vendor_weights.append(tempEnc)

    w_avg = average_enc_weights(enc_vendor_weights)

    for key in w_avg:
        w_avg[key] = torch.Tensor(w_avg[key].decrypt().tolist())
    for model in vendor_models:
        model.load_state_dict(w_avg)

    test_loss, acc = test(vendor_models[0], test_loader)
    print(f'average train loss {(loss / num_selected):.3g} | test loss  {test_loss:.3g} | test acc: {acc:.3f}')