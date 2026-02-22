# CODIGO DE TREINO TRANSFORMER
from numpy import rec
import torch
from torch import nn
from dataloader import TextDatalador
from tokenizer import Tokenizer
from record import WeightNormRecorder as Recorder
from model.transformer import Transformer
import torch.nn.functional as F
from tqdm import tqdm

if __name__ == "__main__":
    config = {
        "data": {
            "path": "data/",
            "batch_size": 128
        },
        "tokenizer": {
            "vocab": "tokenizer/vocab.json",
            "merges": "tokenizer/merges.txt",
        },
        "model": {
            "max_len": 1024,
            "vocab_size": 1000,
            "d_model": 256,
            "head_dim": 64,
            "n_heads": 4,
            "ff_ratio": 16,
            "n_layers": 1,
            "mask": True
        },
        "training": {
            "epochs": 1,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "lr": 1e-3,
            "scheduler_step": 10,
            "scheduler_gamma": 0.1,
            "eval_interval": 100,
            "record_interval": 5,
            "eval_max_gen_len": 32,
            "eval_questions": [
                "<start> 2 + 2 <answer> = 4 <end>",
                "<start> 3 * 5 <answer> = 15 <end>",
                "<start> 10 / 2 <answer> = 5 <end>",
                "<start> 2 * 3 <answer> = 6 <end>",
                "<start> 15 - 7 <answer> = 8 <end>"
            ]
        }
    }

    # Init
    device = config["training"]["device"]
    dataset = TextDatalador(**config["data"]) # datalaoder
    tokenizer = Tokenizer(**config["tokenizer"]) # tokenizer
    model = Transformer(**config["model"]) # modelo
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"]) # otimizador
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step"], gamma=config["training"]["scheduler_gamma"]) # scheduler
    recorder = Recorder(model)
    recorder.calibrate()

    # Training loop
    eval_interval = config["training"]["eval_interval"]
    eval_max_gen_len = config["training"]["eval_max_gen_len"]
    record_interval = config["training"]["record_interval"]
    model.to(device)
    for epoch in range(config["training"]["epochs"]):
        model.train()
        
        last_five_loss = []
        train_dataloader = iter(dataset)
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}", total=len(dataset) // config["data"]["batch_size"])

        for b in pbar:
            
            optimizer.zero_grad() # limpar gradientes antigos
            
            # Tokenize batch
            batch = tokenizer.encode_with_tokens(b)
            inputs = torch.tensor([b['tokens'] for b in batch], dtype=torch.long).to(device)

            # Shift inputs for teacher forcing
            x = inputs[:, :-1]
            y = inputs[:, 1:]

            # Forward pass
            x, scores = model(x)

            # Compute loss
            loss = F.cross_entropy(x.reshape(-1, x.size(-1)), y.reshape(-1), ignore_index=tokenizer.pad_token_id)

            # Backward pass and optimization
            loss.backward() # calcular novos gradientes
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # opcional: clipping de gradiente para evitar explosão
            optimizer.step() # atualizar pesos
            last_five_loss.append(loss.item())
            if len(last_five_loss) > 5:
                last_five_loss.pop(0)

            # Atualizar barra de progresso com perda média
            lr = optimizer.param_groups[0]["lr"]
            valid = (y != tokenizer.pad_token_id).float().mean().item()
            pbar.set_postfix({
                "grad_norm": f"{grad_norm:.4f}",
                "loss": sum(last_five_loss) / len(last_five_loss) if last_five_loss else 0,
                "lr": f"{lr:.2e}",
                "valid%": f"{valid*100:.2f}%"
            })

            # RECORDING
            if (pbar.n + 1) % record_interval == 0:
                recorder.log(pbar.n + 1)
                recorder.plot(f"assets/weight_norms/epoch{epoch+1}.png")

            # EVALUATION
            if (pbar.n + 1) % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    for q in config["training"]["eval_questions"]:
                        batch = tokenizer.encode_with_tokens([q.split(" = ")[0]])  # codifica só a parte da pergunta, sem a resposta
                        inputs = torch.tensor([b['tokens'] for b in batch], dtype=torch.long).to(device)
                        while True:
                            outputs, _ = model(inputs)
                            pred = outputs.argmax(dim=-1)
                            inputs = torch.cat([inputs, pred[:, -1:]], dim=1) # adiciona a última predição como próximo input
                            text = tokenizer.tokenizer.decode(inputs[0].detach().cpu().numpy() , skip_special_tokens=False)
                            if text.endswith("<end>") or len(inputs[0]) > eval_max_gen_len:
                                print(f"Q: {q} -> A: {text}")
                                break
                        print()  # nova linha após cada pergunta
                model.train()

        scheduler.step()
        