from pathlib import Path
from torch.utils.data import IterableDataset, get_worker_info

class TextDatalador(IterableDataset):
    def __init__(self, path: str | Path, batch_size: int = 32):
        super().__init__()
        self.path = Path(path)
        self.files = sorted(self.path.rglob("*.txt"))
        self.batch_size = batch_size

        # rows counts
        self.lines_per_file = {fp: self.count_lines(fp) for fp in self.files}
        self.total_lines = sum(self.lines_per_file.values())

    def count_lines(self, path: Path) -> int:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)  # stream, sem carregar tudo

    def __len__(self):
        return self.total_lines

    def __iter__(self):
        batch = []
        info = get_worker_info()
        if info is None:
            files = self.files
        else:
            # sharding por arquivos: worker i pega i, i+num_workers, i+2*num_workers...
            files = self.files[info.id :: info.num_workers]

        for fp in files:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line:
                        continue
                    batch.append(line.strip())
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
        if batch:
            yield batch


if __name__ == "__main__":
    dataset = TextDatalador("data/", batch_size=32)
    train_dataloader = iter(dataset)