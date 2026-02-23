import torch


def _l2(x):
    return torch.linalg.norm(x.detach().float()).item()


def _l2_grad(param):
    # param é um Parameter (ex.: module.weight)
    if (param is None) or (param.grad is None):
        return None
    return torch.linalg.norm(param.grad.detach().float()).item()


def _is_leaf(m):
    return len(list(m.children())) == 0


class WeightNormRecorder:
    def __init__(self, model):
        self.model = model
        self.names = None
        self.norms = {}
        self.grad_norms = {}
        self.steps = [0]

    def calibrate(self):
        self.names = []
        for name, m in self.model.named_modules():
            if name and _is_leaf(m) and hasattr(m, "weight"):
                self.names.append(name)

        name_to_module = dict(self.model.named_modules())
        self.norms = {n: [_l2(name_to_module[n].weight)] for n in self.names}
        self.grad_norms = {n: [_l2_grad(name_to_module[n].weight)] for n in self.names}

    @torch.no_grad()
    def log(self, step):
        if self.names is None:
            self.calibrate()

        self.steps.append(step)
        name_to_module = dict(self.model.named_modules())

        for n in self.names:
            w = name_to_module[n].weight
            self.norms[n].append(_l2(w))
            self.grad_norms[n].append(_l2_grad(w))

        # Retorna ambos (você pode mudar para retornar só um se preferir)
        return {"weight_norms": self.norms, "grad_norms": self.grad_norms, "steps": self.steps}


    def plot(self, savepath, ncols=3, yscale="log", dpi=200,
            suptitle="Weight/Grad L2 norms (per module)"):
        import math
        import matplotlib.pyplot as plt
        from pathlib import Path

        if not self.norms:
            raise RuntimeError("Nada para plotar: rode calibrate() e/ou log(step) antes.")
        if not getattr(self, "grad_norms", None):
            raise RuntimeError("grad_norms não encontrado: use o recorder que registra grad_norms.")

        keys = sorted(self.norms.keys())
        n = len(keys)
        ncols = max(1, int(ncols))
        nrows = math.ceil(n / ncols)

        xs_full = list(self.steps) if self.steps else None

        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 2.6 * nrows), sharex=True, squeeze=False)
        axes = axes.ravel()

        for i, name in enumerate(keys):
            ax = axes[i]

            ys_w = self.norms[name]
            xs = (xs_full[:len(ys_w)] if xs_full is not None else list(range(len(ys_w))))
            ax.plot(xs, ys_w, lw=1.6, c="C0")

            axg = ax.twinx()  # segundo eixo Y no mesmo subplot [web:16]
            ys_g = self.grad_norms.get(name, [])[:len(xs)]
            ys_g = [float("nan") if v is None else float(v) for v in ys_g]
            axg.plot(xs, ys_g, lw=1.3, c="C1", alpha=0.9)

            if yscale:
                ax.set_yscale(yscale)
                axg.set_yscale(yscale)

            ax.set_title(name, loc="left", fontsize=9)
            ax.grid(True, alpha=0.25)

            if not ax.get_subplotspec().is_last_row():
                ax.tick_params(labelbottom=False)
                axg.tick_params(labelbottom=False)

        for j in range(n, len(axes)):
            axes[j].set_axis_off()  # desliga subplots vazios [web:43]

        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return str(savepath)

