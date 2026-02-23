from torch import nn
from torch.nn import functional as F
import torch
from model.embedding import Embedding
from model.transformer_block import TransformerBlock


class Transformer(nn.Module):
    def __init__(self, max_len: int, vocab_size: int, d_model: int, head_dim: int, n_heads: int, ff_ratio: int, n_layers: int, mask: bool = False):
        super(Transformer, self).__init__()

        # SETUP
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.ff_ratio = ff_ratio
        self.mask = mask
        self.n_layers = n_layers

        # LAYERS
        self.emb = Embedding(vocab_size, d_model, max_len)
        self.layers = nn.ModuleList([TransformerBlock(d_model, head_dim, n_heads, ff_ratio, layer_id, mask) for layer_id in range(n_layers)])
        self.out_proj = nn.Linear(d_model, vocab_size)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # 1) deixa cada submódulo custom se inicializar do jeito dele
        if hasattr(self.emb, "reset_parameters"):
            self.emb.reset_parameters()
        else:
            raise ValueError("Embedding deveria ter método reset_parameters, mas não tem.")

        for block in self.layers:
            if hasattr(block, "reset_parameters"):
                block.reset_parameters()
            else:
                raise ValueError(f"TransformerBlock deveria ter método reset_parameters, mas não tem. block={block}")

        # 2) inicializa módulos padrão (e quaisquer outros que existirem)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        scores = {}
        for layer in self.layers:
            x, score = layer(x)
            scores[layer.layer_id] = score

        x = self.out_proj(x) # F.softmax(self.out_proj(x), dim=-1)
        return x, scores


def plot_attention_batch_layers_heads_rich(
    scores: dict,
    tokens_batch: list[list[str]],
    out_dir: str = "assets/attn",
    cmap: str = "cividis",        # perceptualmente uniforme. [web:192]
    dpi: int = 250,
    max_tokens: int = 40,         # evita plot ilegível
    tick_stride: int | None = None,  # se None, escolhe automaticamente
    annotate: bool = False,       # liga anotação numérica (bom só p/ T pequeno). [web:178]
    annotate_max_T: int = 14,     # acima disso vira poluição visual
    fmt: str = "{:.2f}",
):
        
    from pathlib import Path
    """
    Gera 1 PNG por item do batch, subplots (n_layers x n_heads), com:
    - mesma escala de cor (vmin/vmax) no grid todo
    - gridlines por célula via minor ticks
    - ticks com tokens (com stride) e rotação
    - anotação opcional por célula
    scores: {layer_id: Tensor(B,H,Q,K)}
    tokens_batch: list[list[str]] len==B
    """
    import numpy as np
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_ids = sorted(scores.keys())
    if not layer_ids:
        raise ValueError("scores está vazio.")

    any_score = scores[layer_ids[0]]
    B, H, Q, K = any_score.shape

    if len(tokens_batch) != B:
        raise ValueError(f"tokens_batch deve ter len==B. Recebi {len(tokens_batch)} e B={B}.")

    # Heurística para stride de ticks
    def choose_stride(T):
        if tick_stride is not None:
            return max(1, int(tick_stride))
        if T <= 12:
            return 1
        if T <= 24:
            return 2
        if T <= 40:
            return 4
        return 6

    for b in range(B):
        tokens = tokens_batch[b]
        T = min(len(tokens), Q, K, max_tokens)

        tokens = tokens[:T]
        stride = choose_stride(T)
        tick_pos = np.arange(0, T, stride)
        tick_lab = [tokens[i] for i in tick_pos]

        n_layers = len(layer_ids)
        n_heads = H

        fig, axes = plt.subplots(
            nrows=n_layers,
            ncols=n_heads,
            figsize=(3.2 * n_heads, 3.6 * n_layers),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        # normaliza axes para 2D
        if n_layers == 1 and n_heads == 1:
            axes = np.array([[axes]])
        elif n_layers == 1:
            axes = np.array([axes])
        elif n_heads == 1:
            axes = np.array([[ax] for ax in axes])

        # coleta matrizes + escala global
        vmin, vmax = float("inf"), float("-inf")
        mats = {}
        for lid in layer_ids:
            s = scores[lid][b].detach().cpu().numpy()  # (H,Q,K)
            s = s[:, :T, :T]
            mats[lid] = s
            vmin = min(vmin, float(s.min()))
            vmax = max(vmax, float(s.max()))

        # evita colorbar “flat” se tudo igual (caso patológico)
        if vmin == vmax:
            vmax = vmin + 1e-9

        last_im = None
        for li, lid in enumerate(layer_ids):
            for h in range(n_heads):
                ax = axes[li, h]
                mat = mats[lid][h]  # (T,T)

                last_im = ax.imshow(
                    mat,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect="equal",      # células quadradas = mais legível
                    interpolation="nearest",
                )

                # títulos/labels
                if li == 0:
                    ax.set_title(f"Head {h}", fontsize=11)
                if h == 0:
                    ax.set_ylabel(f"Layer {lid}\nQuery", fontsize=10)

                # ticks com palavras (stride) + rotação. [web:163]
                ax.set_xticks(tick_pos)
                ax.set_yticks(tick_pos)
                ax.set_xticklabels(tick_lab, rotation=45, ha="right", rotation_mode="anchor")
                ax.set_yticklabels(tick_lab)

                # gridlines por célula com minor ticks. [web:185]
                ax.set_xticks(np.arange(-0.5, T, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, T, 1), minor=True)
                ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6, alpha=0.6)
                ax.tick_params(which="minor", bottom=False, left=False)
                ax.tick_params(axis="both", which="major", labelsize=8)

                # anotação numérica (apenas se T pequeno). [web:178]
                if annotate and T <= annotate_max_T:
                    # escolhe cor do texto pelo valor (contraste simples)
                    mid = (vmin + vmax) / 2
                    for i in range(T):
                        for j in range(T):
                            val = mat[i, j]
                            txt_color = "black" if val > mid else "white"
                            ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=6, color=txt_color)

        for ax in axes[-1, :]:
            ax.set_xlabel("Key", fontsize=10)

        fig.suptitle(
            f"Attention scores | batch={b} | layers={n_layers}, heads={n_heads} | T={T} (stride={stride})",
            fontsize=14,
            y=1.02,
        )

        # colorbar única (comparável entre subplots)
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
        cbar.set_label("score", rotation=90)

        out_path = out_dir / f"attention_b{b}_L{n_layers}_H{n_heads}_T{T}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    texto_fake = [
        ['Eu', 'gosto', 'de', 'estudar', 'llms', 'porque', 'admiro', 'como', 'funcionam', 'bem', '.'],
        ['Transformers', 'são', 'incríveis', 'para', 'processar', 'linguagem', 'natural', ',', 'eles', 'arrasam.']
    ]
    x = torch.randint(0, 100, (2, 10))
    max_len = 20
    vocab_size = 100
    d_model = 64
    head_dim = 16
    n_heads = 4
    ff_ratio = 4
    n_layers = 2
    mask = True

    model = Transformer(max_len, vocab_size, d_model, head_dim, n_heads, ff_ratio, n_layers, mask)
    out, scores = model(x)

    tokens_batch = [
    texto_fake[0][:10],  # corta para bater com x.shape[1]==10
    texto_fake[1][:10],
]

    plot_attention_batch_layers_heads_rich(
        scores,
        tokens_batch=tokens_batch,
        out_dir="assets/attn",
        max_tokens=10,
    )