from torch import nn

# Our model inherits from `nn.Module`
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)

    def forward(self, x):
        return self.lin(x)