import torch
from simpleModel import SimpleModel

def zero_model_dict(model):
    state_dict = model.state_dict()
    for key in state_dict: 
        state_dict[key] = torch.zeros(state_dict[key].shape)
    model.load_state_dict(state_dict)

model = SimpleModel()
state_dict = model.state_dict()
for key in state_dict:
    zeros = torch.zeros(state_dict[key].shape)
    state_dict[key] = zeros
model.load_state_dict(state_dict)
print(state_dict)
print(model.state_dict())