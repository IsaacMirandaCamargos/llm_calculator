import torch

def _l2(x):
    return torch.linalg.norm(x.detach().float()).item()  # [web:9]

def _is_leaf(m):
    return len(list(m.children())) == 0

class WeightNormRecorder:
    def __init__(self, model):
        self.model = model
        self.names = None
        self.norms = {}
        self.steps = [0]

    def calibrate(self):
        self.names = []
        for name, m in self.model.named_modules():  # [web:16]
            if name and _is_leaf(m) and hasattr(m, "weight"):
                self.names.append(name)

        name_to_module = dict(self.model.named_modules())  # [web:16]
        self.norms = {n: [_l2(name_to_module[n].weight)] for n in self.names}

    @torch.no_grad()
    def log(self, step):
        if self.names is None:
            self.calibrate()

        self.steps.append(step)
        name_to_module = dict(self.model.named_modules())  # [web:16]

        for n in self.names:
            self.norms[n].append(_l2(name_to_module[n].weight))

        return self.norms

    def plot(
        self,
        savepath,
        ncols=3,
        suptitle="Weight L2 norms (per module)",
        dpi=200,
        yscale="log",                 # "linear" ou "log"
        sort_by="last",               # "name" ou "last"
        top_k=None,                   # ex: 24 (plotar só as top-k); None = todas
        max_plots_per_fig=30,         # paginação: quantos painéis por figura
        w_per_ax=5.4,                 # polegadas por subplot (largura)
        h_per_ax=2.6,                 # polegadas por subplot (altura)
        min_figsize=(12, 7),
        max_figsize=(30, 20),
        wspace=0.35,                  # espaço horizontal entre subplots [web:46]
        hspace=0.55,                  # espaço vertical entre subplots [web:46]
        title_fontsize=9,
        line_width=1.6,
    ):
        import math
        import matplotlib.pyplot as plt
        from pathlib import Path

        if not self.norms:
            raise RuntimeError("Nada para plotar: rode calibrate() e/ou log(step) antes.")

        # --- escolher e ordenar módulos ---
        keys = list(self.norms.keys())

        if sort_by == "last":
            keys.sort(key=lambda k: (self.norms[k][-1] if self.norms[k] else float("-inf")), reverse=True)
        else:
            keys.sort()

        if top_k is not None:
            keys = keys[:int(top_k)]

        nplots_total = len(keys)
        if nplots_total == 0:
            raise RuntimeError("Nada para plotar: lista de módulos vazia.")

        # paginação
        max_plots_per_fig = max(1, int(max_plots_per_fig))
        pages = int(math.ceil(nplots_total / max_plots_per_fig))

        savepath = Path(str(savepath))
        savepath.parent.mkdir(parents=True, exist_ok=True)

        # eixo x
        xs_full = list(self.steps) if self.steps else None

        out_paths = []

        for p in range(pages):
            chunk = keys[p * max_plots_per_fig : (p + 1) * max_plots_per_fig]
            nplots = len(chunk)

            ncols_eff = max(1, int(ncols))
            nrows_eff = int(math.ceil(nplots / ncols_eff))

            # --- figsize dinâmico por página ---
            fig_w = ncols_eff * w_per_ax
            fig_h = nrows_eff * h_per_ax
            fig_w = max(min_figsize[0], min(fig_w, max_figsize[0]))
            fig_h = max(min_figsize[1], min(fig_h, max_figsize[1]))
            figsize = (fig_w, fig_h)

            fig, axes = plt.subplots(
                nrows=nrows_eff, ncols=ncols_eff, figsize=figsize, sharex=True, squeeze=False
            )
            fig.subplots_adjust(wspace=wspace, hspace=hspace)  # controla wspace/hspace [web:46]
            axes_flat = axes.ravel()

            for i, name in enumerate(chunk):
                ax = axes_flat[i]
                ys = self.norms[name]

                if xs_full is None:
                    xs = list(range(len(ys)))
                else:
                    xs = xs_full[:len(ys)]

                ax.plot(xs, ys, linewidth=line_width)

                if yscale:
                    ax.set_yscale(yscale)

                ax.set_title(name, loc="left", fontsize=title_fontsize)
                ax.grid(True, alpha=0.25)

                # menos poluição: só última linha mostra ticks do x
                if not ax.get_subplotspec().is_last_row():
                    ax.tick_params(labelbottom=False)

            # desligar eixos vazios
            for j in range(nplots, len(axes_flat)):
                axes_flat[j].axis("off")

            page_title = suptitle if pages == 1 else f"{suptitle} (page {p+1}/{pages})"
            fig.suptitle(page_title)

            # reservar espaço para o suptitle ao usar tight_layout (rect) [web:6]
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # rect = (left,bottom,right,top) [web:9]

            # salvar
            if pages == 1:
                out_file = savepath
            else:
                out_file = savepath.with_name(f"{savepath.stem}_page{p+1:02d}{savepath.suffix}")

            fig.savefig(str(out_file), dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            out_paths.append(str(out_file))

        return out_paths[0] if len(out_paths) == 1 else out_paths
