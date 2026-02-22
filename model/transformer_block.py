# from torch import nn
# from .attention import Attention
# from .feed_foward import FeedForwardNetwork
# import torch


# class TransformerBlock(nn.Module):
#     def __init__(self, d_model: int, head_dim: int, n_heads: int, ff_ratio: int, layer_id: int, mask: bool = False):
#         super(TransformerBlock, self).__init__()

#         # SETUP
#         self.d_model = d_model
#         self.head_dim = head_dim
#         self.n_heads = n_heads
#         self.ff_ratio = ff_ratio
#         self.mask = mask
#         self.layer_id = layer_id

#         # LAYERS
#         self.attention = Attention(d_model=d_model, head_dim=head_dim, n_heads=n_heads, mask=mask)
#         self.norm_1 = nn.RMSNorm(d_model)

#         self.ffn = FeedForwardNetwork(d_model=d_model, ff_ratio=ff_ratio)
#         self.norm_2 = nn.RMSNorm(d_model)

#     @torch.no_grad()
#     def reset_parameters(self):
#         # Submódulos custom
#         if hasattr(self.attention, "reset_parameters"):
#             self.attention.reset_parameters()
#         if hasattr(self.ffn, "reset_parameters"):
#             self.ffn.reset_parameters()

#         # RMSNorm: scale=1 (e bias=0 se existir)
#         if hasattr(self.norm_1, "weight") and self.norm_1.weight is not None:
#             nn.init.ones_(self.norm_1.weight)
#         if hasattr(self.norm_1, "bias") and self.norm_1.bias is not None:
#             nn.init.zeros_(self.norm_1.bias)

#         if hasattr(self.norm_2, "weight") and self.norm_2.weight is not None:
#             nn.init.ones_(self.norm_2.weight)
#         if hasattr(self.norm_2, "bias") and self.norm_2.bias is not None:
#             nn.init.zeros_(self.norm_2.bias)

#     def forward(self, x):
#         y, scores = self.attention(x)
#         x = self.norm_1(x + y)
#         x = self.norm_2(x + self.ffn(x))
#         return x, scores

from torch import nn
from .attention import Attention
from .feed_foward import FeedForwardNetwork
import torch


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_dim: int,
        n_heads: int,
        ff_ratio: int,
        layer_id: int,
        mask: bool = False,
        dropout: float = 0.1,
    ):
        super(TransformerBlock, self).__init__()

        # SETUP
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.ff_ratio = ff_ratio
        self.mask = mask
        self.layer_id = layer_id

        # LAYERS
        self.attention = Attention(d_model=d_model, head_dim=head_dim, n_heads=n_heads, mask=mask)
        self.norm_1 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNetwork(d_model=d_model, ff_ratio=ff_ratio)
        self.norm_2 = nn.LayerNorm(d_model)

        # DROPOUT
        self.dropout_attn = nn.Dropout(p=dropout)
        self.dropout_ffn = nn.Dropout(p=dropout)

    @torch.no_grad()
    def reset_parameters(self):
        # Submódulos custom
        if hasattr(self.attention, "reset_parameters"):
            self.attention.reset_parameters()
        if hasattr(self.ffn, "reset_parameters"):
            self.ffn.reset_parameters()

        # RMSNorm: scale=1 (e bias=0 se existir)
        if hasattr(self.norm_1, "weight") and self.norm_1.weight is not None:
            nn.init.ones_(self.norm_1.weight)
        if hasattr(self.norm_1, "bias") and self.norm_1.bias is not None:
            nn.init.zeros_(self.norm_1.bias)

        if hasattr(self.norm_2, "weight") and self.norm_2.weight is not None:
            nn.init.ones_(self.norm_2.weight)
        if hasattr(self.norm_2, "bias") and self.norm_2.bias is not None:
            nn.init.zeros_(self.norm_2.bias)

    def forward(self, x):
        x_norm = self.norm_1(x)
        y, scores = self.attention(x_norm)
        x = x + self.dropout_attn(y)

        x_norm = self.norm_2(x)
        ff = self.ffn(x_norm)
        x = x + self.dropout_ffn(ff)

        return x, scores
