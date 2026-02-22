# lr_schedulers_plot_2epochs.py
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR, OneCycleLR, LinearLR, LambdaLR
)

def simulate(factory, total_steps=2000, base_lr=1e-3):
    p = nn.Parameter(torch.zeros(1))
    opt = SGD([p], lr=base_lr)
    sched = factory(opt)

    lrs = []
    for _ in range(total_steps):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return lrs

def main():
    epochs = 2
    steps_per_epoch = 1000
    total_steps = epochs * steps_per_epoch
    base_lr = 1e-3

    factories = {
        "Constant": lambda opt: LambdaLR(opt, lr_lambda=lambda step: 1.0),

        # Atenção: StepLR “por definição” é em épocas, mas como aqui chamamos .step() a cada batch,
        # step_size=200 significa “a cada 200 steps” nesta simulação. [web:119]
        "StepLR": lambda opt: StepLR(opt, step_size=200, gamma=0.5),

        "Exponential": lambda opt: ExponentialLR(opt, gamma=0.999),
        "CosineAnneal": lambda opt: CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-5),  # T_max = iterações [web:97]
        "LinearLR": lambda opt: LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=total_steps),

        # OneCycleLR: total_steps é inferido por epochs * steps_per_epoch [web:96]
        "OneCycle": lambda opt: OneCycleLR(
            opt,
            max_lr=3e-3,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        ),
    }

    plt.figure(figsize=(12, 6))
    xs = list(range(total_steps))

    for name, fac in factories.items():
        lrs = simulate(fac, total_steps=total_steps, base_lr=base_lr)
        plt.plot(xs, lrs, label=name, linewidth=2)

    # linhas verticais marcando fronteira entre épocas
    plt.axvline(steps_per_epoch, color="k", linestyle="--", linewidth=1, alpha=0.5)
    plt.text(steps_per_epoch + 10, max(base_lr, 3e-3), "epoch 2", fontsize=10)

    plt.title("Learning rate vs step (2 epochs, 1,000 steps each)")
    plt.xlabel("Step")
    plt.ylabel("LR")
    plt.legend(ncol=3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("assets/lr_schedulers_2epochs.png")

if __name__ == "__main__":
    main()
