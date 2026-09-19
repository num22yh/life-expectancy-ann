"""기대수명 예측 ANN"""

from torch import nn


class LifeExpectancyANN(nn.Module):
    def __init__(self, n_features=14, hidden_nodes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_nodes),
            nn.Sigmoid(),
            nn.Linear(hidden_nodes, 1),
        )

    def forward(self, x):
        return self.network(x)
