from re import M
import tenseal as ts
import queue

from vendor import Vendor
from coordinator import Coordinator
from simpleModel import SimpleModel

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm


context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40

num_clients = 20
num_selected = 6
num_rounds = 150
epochs = 5
batch_size = 1024


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
train_data = datasets.MNIST("./Algorithm/", train=True, download=False, transform=transform_train)
train_data_split = random_split(train_data, [int(train_data.data.shape[0] / num_clients) for _ in range(num_clients)])
train_loader = [DataLoader(x, batch_size=batch_size, shuffle=True) for x in train_data_split]

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
test_loader = DataLoader(datasets.MNIST("./Algorithm/", train=False, download=False, transform=transform_test), batch_size=batch_size, shuffle=True)


def client_update(model, optimizer, train_loader, loss_fn, epoch=5):
    """
    This function updates/trains client model on client data
    """
    model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()

def server_aggregate(global_model, client_models):
    """
    This function has aggregation method 'mean'
    """
    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


def test(model, test_loader):
    """This function test the global model on test data and returns test loss and test accuracy """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc


coordinator_model = SimpleModel().cuda()

client_models = [SimpleModel().cuda() for _ in range(num_selected)]
for model in client_models:
    model.load_state_dict(coordinator_model.state_dict())
optimizer = [optim.Adam(coordinator_model.parameters(), lr=0.01) for _ in client_models]
loss_fn = nn.NLLLoss()

###### List containing info about learning #########
losses_train = []
losses_test = []
acc_train = []
acc_test = []
# Runnining FL

for r in range(num_rounds):
    # select random clients
    client_idx = np.random.permutation(num_clients)[:num_selected]
    # client update
    loss = 0
    for i in tqdm(range(num_selected)):
        loss += client_update(client_models[i], optimizer[i], train_loader[client_idx[i]], loss_fn, epoch=epochs)
    
    losses_train.append(loss)
    # server aggregate
    server_aggregate(coordinator_model, client_models)
    
    test_loss, acc = test(coordinator_model, test_loader)
    losses_test.append(test_loss)
    acc_test.append(acc)
    print('%d-th round' % r)
    print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_selected, test_loss, acc))


"""


test_data = datasets.MNIST("./Algorithm/", train=False, download=False, transform=transforms.ToTensor())
x_test = test_data.data
y_test = test_data.targets


x_train_norm = x_train.float() / 255
N, H, W = x_train_norm.shape
x_train_norm = x_train_norm.reshape((N, 1, H, W))
print(f"Dataset shape: {x_train_norm.shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = TensorDataset(x_train_norm.to(device), y_train.to(device))

n_samples = len(dataset)

# We split the training data into 70% for training, 30% for validation.
n_train_samples = int(n_samples*0.7)
n_val_samples = n_samples - n_train_samples
# We again use the random_split function to get to random subsets of the data.
train_dataset, val_dataset = random_split(dataset, [n_train_samples, n_val_samples])

train_data_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=n_val_samples)

model = SimpleModel()
# We moved the dataset to `device`, then we must move model as well.
# Don't worry, if you forget it pytorch will throw an error and your code will not run before you fix it.
model.to(device)
loss_fn = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def evaluate_model(val_data_loader, model, loss_fn):
    losses = []
    n_correct = 0
    with torch.no_grad():
        for b_x, b_y in val_data_loader:
            pred = model(b_x)
            loss = loss_fn(pred, b_y)
            losses.append(loss.item())
            
            hard_preds = pred.argmax(dim=1)
            n_correct += torch.sum(pred.argmax(dim=1) == b_y).item()
        val_accuracy = n_correct/len(val_dataset)
        val_avg_loss = sum(losses)/len(losses)    
    
    return val_accuracy, val_avg_loss

for epoch in range(10):
    print('------ Epoch {} ------'.format(epoch))
    for i, (b_x, b_y) in enumerate(train_data_loader):
        
        # Compute predictions and losses
        pred = model(b_x)
        loss = loss_fn(pred, b_y)
        
        # Count number of correct predictions
        hard_preds = pred.argmax(dim=1)
        n_correct = torch.sum(pred.argmax(dim=1) == b_y).item()

        # Backpropagate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Every 10 batches, display progress
        if i % 10 == 0:
            display_str = 'Batch {} '
            display_str += '\tLoss: {:.3f} '
            display_str += '\tLoss (val): {:.3f}'
            display_str += '\tAccuracy: {:.2f} '
            display_str += '\tAccuracy (val): {:.2f}'
            
            val_accuracy, val_avg_loss = evaluate_model(val_data_loader, model, loss_fn)
            print(display_str.format(i, loss, val_avg_loss, n_correct/len(b_x), val_accuracy)) 
"""
"""
bin_edges = [x for x in range(121)] # [0,1,2,...,118,119,120]


histQ = queue.Queue(100) # Queue for encrypted histograms going from vendors to coordinator
mergedQ = queue.Queue() # Queue for encrypted merged histograms going from coordinator to vendors
lossQ = queue.Queue() # Queue for (unencrypted) loss-values going from vendors to coordinator

vendor1 = Vendor(1, "Vendor-1", context, bin_edges, histQ, mergedQ, lossQ)
vendor2 = Vendor(2, "Vendor-2", context, bin_edges, histQ, mergedQ, lossQ)
vendor3 = Vendor(3, "Vendor-3", context, bin_edges, histQ, mergedQ, lossQ)

coordinator = Coordinator(4, "Coordinator", context, histQ, mergedQ, lossQ, 3)

vendor1.start()
vendor2.start()
vendor3.start()
coordinator.start()
"""