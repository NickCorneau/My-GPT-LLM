import re

def read_file(relative_file_path) -> str:
    
    print("The file we are reading in is:", relative_file_path)

    with open(relative_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    return raw_text

class SimpleVocabBuilderV1:
    def __init__(self):
        pass

    def build_vocab(self, raw_text) -> dict:

        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [x.strip() for x in preprocessed if x.strip()]
        print("\nThe first 50 tokens are:", preprocessed[0:50])
        
        all_words = sorted(set(preprocessed))
        print()
        print("There is a vocabulary size of:", len(all_words))


        vocab = {token:integer for integer,token in enumerate(all_words)}
        print("\nThe first 25 words in the vocabulary are:")
        for i, item, in enumerate(vocab.items()):
            print(item)
            if i >= 25:
                break

        return vocab

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text) -> list[int]:
        print("\nEncoding text to a vector...")
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        print("The first 50 ids are:", ids[:50])

        return ids

    def decode(self, ids) -> str:
        print("\nDecoding a vector back to text...")
        # converts ids back into text, since we are indexing into the int_to_str map using our IDs
        text = " ".join([self.int_to_str[i] for i in ids])

        # remove whitespace before this punctuation
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        print("The first 50 chars are:", text[:50])
        return text

def main():
    file_path = "the-verdict.txt"
    raw_text = read_file(file_path)

    vocab_builder = SimpleVocabBuilderV1()
    vocab = vocab_builder.build_vocab(raw_text)
    
    tokenizer = SimpleTokenizerV1(vocab)
    ids = tokenizer.encode(raw_text)

    text = tokenizer.decode(ids)
    return vocab

main()


