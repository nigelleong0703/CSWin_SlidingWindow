from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.activation = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)

        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x
