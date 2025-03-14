import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.kaiming_uniform_(self.weight)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            torch.nn.init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, x):
        output = torch.matmul(x, self.weight.T)
        if self.bias is not None:
            output += self.bias
        return output