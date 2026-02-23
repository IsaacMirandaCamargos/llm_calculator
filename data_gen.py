import polars as pl
from joblib import Parallel, delayed
import random as r
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import cpu_count
import shutil

class ParallelTqdm(Parallel):
    def __init__(self, *args, total=None, desc=None, tqdm_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._total = total
        self._desc = desc
        self._tqdm_kwargs = {} if tqdm_kwargs is None else dict(tqdm_kwargs)
        self._pbar = None

    def __call__(self, *args, **kwargs):
        with tqdm(total=self._total, desc=self._desc, **self._tqdm_kwargs) as pbar:
            self._pbar = pbar
            return super().__call__(*args, **kwargs)

    def print_progress(self):
        # Se total não foi passado, dá pra inferir depois que começou a despachar tasks
        if self._pbar is None:
            return
        if self._pbar.total is None and self.n_dispatched_tasks:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def worker_create_text(n_batch, max, min, save_path, indice):
    operations = np.array(["+", "-", "*", "/"])
    data = []
    for _ in range(n_batch):
        num_numbers = 2 # r.randint(2, 5)
        num_expressions = 1 # num_numbers - 1
        numbers = np.random.randint(min, max, size=num_numbers)
        expressions = operations[np.random.randint(0, len(operations), size=num_expressions)]
        result = numbers[0]
        for i, exp in enumerate(expressions):
            if exp == "+":
                result += numbers[i + 1]
            elif exp == "-":
                result -= numbers[i + 1]
            elif exp == "*":
                result *= numbers[i + 1]
            elif exp == "/":
                if numbers[i + 1] != 0:
                    result /= numbers[i + 1]
                else:
                    result = "undefined"

        text = "<start> "
        for i in range(num_numbers):
            text += f"{numbers[i]}"
            if i < num_expressions:
                text += f" {expressions[i]} "
        text += " <answer>"
        text += f" = {result:.2f}" if isinstance(result, (int, float)) else f" = {result}"
        text += " <end>"
        data.append(text)

    with open(Path(save_path) / f"data_{indice}.txt", "w") as f:
        for line in data:
            f.write(line + "\n")

def create_data(n_total, n_batch, max, min, save_path, num_workers):
    save_path = Path(save_path)
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    num_batches = n_total // n_batch
    pbar = ParallelTqdm(n_jobs=num_workers, total=num_batches, desc="Generating data")
    
    pbar(delayed(worker_create_text)(n_batch, max, min, save_path, i) for i in range(num_batches))

if __name__ == "__main__":
    n_total = 1_000_000
    n_batch = 100_000
    min_value = -10
    max_value = 10
    save_path = "data"
    num_workers = max(1, cpu_count() - 1)
    create_data(n_total, n_batch, max_value, min_value, save_path, num_workers)
    