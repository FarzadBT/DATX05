import tenseal as ts
import copy
from tqdm import tqdm
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.nn.functional as F
import numpy as np

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
num_clients = 20
num_selected = 6
num_rounds = 150
epochs = 10
batch_size = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = datasets.MNIST("./Algorithm/", train=True, transform=transforms.ToTensor())

train_data.data = train_data.data.float() / 255
N, H, W = train_data.data.shape
train_data.data = train_data.data.reshape((N, 1, H, W))

#train_set = TensorDataset(train_data.data.to(device), train_data.targets.to(device))
train_set = TensorDataset(train_data.data, train_data.targets)
train_data_split = random_split(train_set, [int(len(train_set) / num_clients) for _ in range(num_clients)])
train_loader = [DataLoader(x, batch_size=batch_size, shuffle=True) for x in train_data_split]

test_data = datasets.MNIST("./Algorithm/", train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=2048, shuffle=True)



def update_vendor(client_model, optimizer, train_loader, loss_fn, epoch=5):
    client_model.train()
    for e in tqdm(range(epochs)):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = client_model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

    client_dict = client_model.state_dict()
    for key in client_dict:
        print(client_dict[key][0])
        client_dict[key] = ts.ckks_tensor(context, client_dict[key])

    return (loss.item(), client_dict)

def average_enc_weights(vendor_weights):
    w_avg = copy.deepcopy(vendor_weights[0])

    for key in w_avg.keys():
        for i in range(1, len(vendor_weights)):
            w_avg[key] += vendor_weights[i].state_dict()[key]
        w_avg[key] = torch.div(w_avg[key], len(vendor_weights))

    return w_avg

def test(global_model, test_loader):
    """This function test the global model on test data and returns test loss and test accuracy """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc



#global_model = SimpleModel().to(device)
global_model = SimpleModel()

#vendor_models = [SimpleModel().to(device) for _ in range(num_clients)]
vendor_models = [SimpleModel() for _ in range(num_clients)]
for model in vendor_models:
    model.load_state_dict(global_model.state_dict())

vendor_optimizers = [torch.optim.Adam(model.parameters(), lr=0.01) for _ in range(num_clients)]

loss_fns = [nn.NLLLoss() for _ in range(num_clients)]

for r in range(num_rounds):
    print(f"Round {r+1}")

    selected_clients = np.random.permutation(num_clients)[:num_selected]

    loss = 0
    enc_vendor_weights = []
    for i in range(num_selected):
        print(f"Vendor {i+1} update")
        client = selected_clients[i]
        (tempLoss, tempEnc) = update_vendor(vendor_models[client], vendor_optimizers[client], train_loader[client], loss_fns[client], epochs)
        loss += tempLoss
        enc_vendor_weights.append(tempEnc)

    w_avg = average_enc_weights(enc_vendor_weights)
    for key in w_avg:
        w_avg[key] = w_avg[key].decrypt()
    
    for model in vendor_models:
        model.load_state_dict(w_avg)

    test_loss, acc = test(vendor_models[0], test_loader)
    print(f'average train loss {(loss / num_selected):.3g} | test loss  {test_loss:.3g} | test acc: {acc:.3f}')