from torch import nn


class MLP(nn.Module):
    def __init__(
        self, dims: list[int], activation: type[nn.Module] = nn.ReLU, dropout: float = 0.0, activate_last: bool = False
    ):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or activate_last:
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
