import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

class MLP(nn.Module):
    
    def __init__(self,
            dimensions,
            activation='relu',
            dropout=0.,
            batchnorm=False,
            ):
        super(MLP, self).__init__()
        self.all_dimensions = dimensions
        if len(self.all_dimensions) < 2:
            return
        self.input_dim = dimensions[0]
        self.other_dimensions = dimensions[1:]
        self.activation = activation
        self.dropout = dropout
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.batchnorm_modules = nn.ModuleList(
                [nn.BatchNorm1d(d) for d in self.other_dimensions[:-1]]
            )

        # Modules
        self.linears = nn.ModuleList([nn.Linear(self.input_dim, self.other_dimensions[0])])
        for din, dout in zip(self.other_dimensions[:-1], self.other_dimensions[1:]):
            self.linears.append(nn.Linear(din, dout))
    
    def forward(self, x):
        if len(self.all_dimensions) < 2:
            # identity
            return x
        for i,lin in enumerate(self.linears):
            x = lin(x)
            if (i < len(self.linears)-1):
                x = F.__dict__[self.activation](x)
                if self.batchnorm:
                    x = self.batchnorm_modules[i](x)
                if self.dropout > 0:
                    x = F.dropout(x, self.dropout, training=self.training)
        return x
