import torch
from torch.nn.parameter import Parameter

## deprecated

class ParaLayer(torch.nn.Module):
    def __init__(self, indim: int, outdim: int, bias: bool=True):
        super(ParaLayer, self).__init__()

        self.indim = indim
        self.outdim = outdim
        self.b = bias

        if bias:
            self.weights = Parameter(torch.rand(indim + 1, outdim))
            self.bias = Parameter(self.weights[-1, :])
        else:
            self.weights = Parameter(torch.rand(indim, outdim))
    
    def check_input(self, input):
        if self.b:
            add_ones = torch.ones(input.shape[0], 1)
            return torch.cat((input, add_ones), 1)
        else:
            return input
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        checked_input = self.check_input(input)
        return checked_input @ self.weights
