import tiktoken

def read_file(relative_file_path) -> str:
    print("\nThe file we are reading in is:", relative_file_path)
    with open(relative_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return raw_text

tokenizer = tiktoken.get_encoding("gpt2")

# Encode using byte-pair encoding
encoded_text = tokenizer.encode(read_file("data/the-verdict.txt"), allowed_special={"<|endoftext|>"})

# Test with a small sample size
sample_size = 16
encoded_sample = encoded_text[:sample_size]
print(f"The first {sample_size} tokens are: {encoded_sample}")

context_size = 4
for i in range(1, context_size+1):
    context = encoded_sample[:i]
    desired = encoded_sample[i]
    print(tokenizer.decode(context), "--->", tokenizer.decode([desired]))

