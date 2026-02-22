from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

class Tokenizer:
    def __init__(self, vocab: str, merges: str, special_tokens: list[str] = ["<start>", "<answer>", "<end>", "<unk>", "<pad>"]):
        self.tokenizer = ByteLevelBPETokenizer(vocab, merges)
        self.tokenizer.add_special_tokens(special_tokens)
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")

    def encode_with_tokens(self, batch: list[str]):
        encs = self.tokenizer.encode_batch(batch)  # list[Encoding] [web:9]
        max_len = max(len(e.ids) for e in encs)
        out = []
        for e in encs:
            delta_size = max_len - len(e.ids)
            pieces = [self.tokenizer.decode([i], skip_special_tokens=False) for i in e.ids]  # [web:9]
            out.append({"tokens": e.ids + [self.pad_token_id] * delta_size, "pieces": pieces + ["<pad>"] * delta_size})
        return out

def batch_iterator(batch_size=256):
    batch = []
    for file in ds:
        with open(file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                batch.append(line)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []

    if batch:
        yield batch

if __name__ == "__main__":
    data_path = Path("data")

    ds = list(data_path.rglob("*.txt"))

    # tokenizer = ByteLevelBPETokenizer()
    # tokenizer.train_from_iterator(
    #     batch_iterator(batch_size=2048),
    #     vocab_size=100,
    #     min_frequency=10,
    #     special_tokens=["<start>", "<answer>", "<end>", "<unk>"],
    # )

    # tokenizer.save_model("tokenizer")  # cria vocab.json e merges.txt

    tokenizer = Tokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
    text = ["<start> 323230 * 2324 <answer> = 751186520 <end>", "<start> 1 + 2 <answer> = 3 <end><pad>"]
    print("Input text:", text)
    enc = tokenizer.encode_with_tokens(text)
    print("Encoded:", enc)



