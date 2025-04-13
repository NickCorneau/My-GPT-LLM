import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

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

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
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

# testing our encoder on a random file
raw_text = read_file("data/the-verdict.txt")
dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=6, stride=4, shuffle=False
        )
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)


# testing embedding vectors
input_ids = torch.tensor([2,3,5,1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print("\nRandomly generated embedding matrix: ")
print(embedding_layer.weight)

print("\nManually selected input_ids for the example: ")
print(input_ids)

print("\nEmbedding vectors (matrix) associated to the input_ids: ")
print(embedding_layer(input_ids))
