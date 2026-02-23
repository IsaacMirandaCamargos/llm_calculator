from torch import nn
import torch


class Attention(nn.Module):
    def __init__(self, d_model, head_dim, n_heads, mask=False, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.mask = mask
        self.divisor = head_dim ** 0.5

        self.query  = nn.Linear(d_model, head_dim * n_heads, bias=False)
        self.keys   = nn.Linear(d_model, head_dim * n_heads, bias=False)
        self.values = nn.Linear(d_model, head_dim * n_heads, bias=False)
        self.output_projection = nn.Linear(head_dim * n_heads, d_model)

        # Dropouts
        self.attn_dropout = nn.Dropout(dropout)   # aplicado nas probs de atenção
        self.proj_dropout = nn.Dropout(dropout)   # aplicado após out_proj

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.keys.weight)
        nn.init.xavier_uniform_(self.values.weight)

        if self.output_projection.bias is not None:
            nn.init.normal_(self.output_projection.bias, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor):
        batch, seq_len, _ = x.shape

        q = self.query(x).reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.keys(x).reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.values(x).reshape(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        score = (q @ k.transpose(-2, -1)) / self.divisor  # (B, H, T, T)

        if self.mask:
            T = score.size(-1)
            causal_mask = torch.triu(torch.ones(T, T, device=score.device), diagonal=1).bool()
            score = score.masked_fill(causal_mask, float("-inf"))

        attn_probs = torch.softmax(score, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)  # dropout nas atenções

        attn = attn_probs @ v  # (B, H, T, D)

        out = attn.transpose(1, 2).reshape(batch, seq_len, self.n_heads * self.head_dim)
        out = self.output_projection(out)
        out = self.proj_dropout(out)  # dropout na saída do bloco de atenção

        return out, attn_probs
