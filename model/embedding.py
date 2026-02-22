from torch import nn
import torch

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int):
        super().__init__()

        # SETUP
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        
        # LAYERS
        self.emb = nn.Embedding(vocab_size, d_model)
        pe = self._build_positional_encoding(max_len, d_model)
        self.register_buffer("pe", pe)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # 1) token embedding: GPT-like (Normal(0, 0.02))
        nn.init.normal_(self.emb.weight, mean=0, std=0.1)  # nn.init funciona direto em tensors/weights [web:121]

        # 2) se tiver padding_idx, zera o vetor de PAD (prática padrão do Embedding)
        if self.emb.padding_idx is not None:
            self.emb.weight[self.emb.padding_idx].zero_()  # Embedding costuma manter padding em zero [web:171]    

    # Positional Encoding
    def _build_positional_encoding(self, max_len, d_model):
        base = torch.tensor(10_000)
        arange = torch.arange(0, max_len).unsqueeze(1)  # shape (max_len, 1)
        i_model = torch.repeat_interleave(torch.arange(0, d_model // 2), repeats=2)
        div_term = torch.exp((i_model / d_model) * torch.log(base))
        pe = arange / div_term
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe  

    def forward(self, x):
        batch, seq_len = x.shape
        x = self.emb(x)
        x = x + self.pe[:seq_len, :]
        return x


def plot_pe_pairs(pe, title=None, savepath=None):
    import matplotlib.pyplot as plt
    import numpy as np
    """
    Plota cada par (sin, cos) em um subplot (um embaixo do outro).
    pe: np.ndarray shape (seq_len, d_model), onde d_model é par.
    """
    seq_len, d_model = pe.shape
    assert d_model % 2 == 0, "d_model precisa ser par para formar pares (sin, cos)."

    pos = np.arange(seq_len)
    n_pairs = d_model // 2

    # Estilo "bonito" usando style sheet do Matplotlib. [web:146]
    plt.style.use("seaborn-v0_8-whitegrid")  # se não existir, comente esta linha

    fig, axes = plt.subplots(
        nrows=n_pairs,
        ncols=1,
        figsize=(11, 2.6 * n_pairs),
        sharex=True,
        constrained_layout=True,  # ajuda a evitar sobreposição de elementos. [web:135][web:143]
    )
    if n_pairs == 1:
        axes = [axes]

    for i in range(n_pairs):
        ax = axes[i]
        sin_dim = 2 * i
        cos_dim = 2 * i + 1

        ax.plot(pos, pe[:, sin_dim], label=f"dim {sin_dim} (sin)", linewidth=2.2)
        ax.plot(pos, pe[:, cos_dim], label=f"dim {cos_dim} (cos)", linewidth=2.2)

        ax.set_ylabel("valor")
        ax.set_title(f"Par i={i}: dims ({sin_dim}, {cos_dim})", loc="left", fontsize=11)
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.25)

        # Linha de referência em y=0 pra leitura rápida
        ax.axhline(0, color="black", linewidth=1, alpha=0.15)

    axes[-1].set_xlabel("pos")

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig, axes


if __name__ == "__main__":
    seq_len = 160
    d_model = 8

    base = torch.tensor(10_000)
    arange = torch.arange(0, seq_len).unsqueeze(1)  # shape (seq_len, 1)
    i_model = torch.repeat_interleave(torch.arange(0, d_model // 2), repeats=2)
    div_term = torch.exp((i_model / d_model) * torch.log(base))
    pe = arange / div_term
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])

    plot_pe_pairs(
        pe,
        title=f"Sinusoidal Positional Encoding (seq_len={seq_len}, d_model={d_model})",
        savepath="assets/positional_encoding_pairs.png",
    )
