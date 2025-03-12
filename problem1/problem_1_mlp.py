import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int,
        bias: bool = True,
        activation: str = "ReLU",  # ReLU, Tanh, GELU, etc
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.bias = bias
        self.activation = activation

        self.net = self.initialize_net()

    def initialize_net(self):
        """Build the network according to the provided hyperparameters."""
        raise NotImplementedError("Not implemented!")
    
    def forward(self, x):
        """
        Implement a forward pass where the output AND the input require
        gradients so as to be differentiable.
        """
        raise NotImplementedError("Not implemented!")
