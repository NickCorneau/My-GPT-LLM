import torch
from torch.utils.data import Dataset, Dataloader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        # Use torches for more efficient target-pair encodings
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __row__(self, idx):
        return self.inputs_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = Dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers
            )
    return dataloader

def read_file(relative_file_path) -> str:
    print("\nThe file we are reading in is:", relative_file_path)
    with open(relative_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return raw_text

raw_text = read_file("data/the-verdict.txt")
dataloader = create_dataloader_v1(
            raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
        )
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

