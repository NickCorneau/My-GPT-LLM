import tiktoken

def read_file(relative_file_path) -> str:
    print("\nThe file we are reading in is:", relative_file_path)
    with open(relative_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return raw_text

tokenizer = tiktoken.get_encoding("gpt2")

tokenized_text = tokenizer.encode(read_file("the-verdict.txt"), allowed_special={"<|endoftext|>"})
print(tokenized_text[:50])

strings = tokenizer.decode(tokenized_text)
print(strings[:50])
