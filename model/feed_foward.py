from torch import nn
import torch.nn.functional as F
import torch


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model:int, ff_ratio: int) -> None:
        super(FeedForwardNetwork, self).__init__()

        # SETUP
        self.d_model = d_model
        self.d_ff = d_model * ff_ratio

        # LAYERS
        self.in_layer = nn.Linear(d_model, self.d_ff, bias=True)
        self.out_layer = nn.Linear(self.d_ff, d_model, bias=True)

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.in_layer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.in_layer.bias)

        nn.init.normal_(self.out_layer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, x):
        x = F.silu(self.in_layer(x))
        x = self.out_layer(x)
        return x