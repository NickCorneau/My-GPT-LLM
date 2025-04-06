# My GPT LLM

A repository exploring GPT and Large Language Models (LLMs), covering fundamental concepts, training methods, and practical applications.

This repository documents an educational journey through understanding and implementing GPT and related language models.

---

## Contents

### Chapter 1: Understanding LLMs

#### Steps for Training

- **Pre-training**: Training on large-scale datasets to learn general language patterns.
- **Fine-tuning**: Further training on specific datasets tailored to certain tasks or domains.

#### Transformer Architecture

- Introduced in ["Attention is All You Need"](https://arxiv.org/html/1706.03762v7).
- Components:
  - **Encoder**: Encodes input text into contextual vectors.
  - **Decoder**: Generates output text from encoded vectors.
  - **Self-attention**: Captures dependencies between words in text.
- Extensions:
  - **BERT**: Encoder-based, optimal for classification tasks.
  - **GPT**: Decoder-based, ideal for generating text predictions.

#### GPT Architecture

- Initially introduced in ["Improving Language Understanding by Generative Pre-Training"](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).
- Expanded with GPT-3, significantly increasing layers (96) and parameters (175 billion).
- Demonstrated *emergent behaviors* like effective translation without explicit training.

#### Datasets

- GPT-3 trained using various large datasets, costing approximately $4.6 million.
- Relevant datasets:
  - [EleutherAI](https://www.eleuther.ai/language-modeling)
  - [AllenAI Dolma](https://huggingface.co/datasets/allenai/dolma)

#### Building an LLM

**Blueprint:**

- **Stage 1: Data & Architecture**
  1. Data Preparation
  2. Attention Mechanism
  3. Architecture Selection

- **Stage 2: Pre-training**
  1. Training Loop
  2. Evaluation
  3. Load Pretrained Weights

- **Stage 3: Fine-tuning**
  - Classifier or Personal Assistant Training

---

### Chapter 2: Working with Text Data

#### IDE Tools

- **nvim**:
  ```powershell
  wsl
  nvim <filename>
  ```
  - Needs PyTorch/autocompletion setup.
- **TorchStudio**: User-friendly, may require further tutorials.

#### Embeddings

- Convert unstructured data into continuous vectors.
- Types: Word embeddings (Word2Vec), paragraph embeddings.
- **Dimensionality**: GPT-3 uses 12,288-dimensional embeddings.

#### Tokenization

- **Simple Regex Tokenizer**:
  ```python
  preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
  ```

- **Simple Vocabulary Builder**:
  - Maps words to unique integers for token encoding.

- **Contextual Tokens**:
  - Special tokens (`<|endoftext|>`, `<unk>`, `[PAD]`) enhance model capability.

- **Byte-Pair Encoding (BPE)**:
  - Used by GPT models for efficient tokenization.
  - Python implementation:
    ```python
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt-2")
    tokenizer.encode("Example text")
    ```

#### Data Sampling

- Create input-target pairs:
  ```python
  tokenizer = tiktoken.get_encoding("gpt2")
  encoded_text = tokenizer.encode(read_file("the-verdict.txt"))
  ```

- PyTorch efficient data handling:
  ```python
  class GPTDataset(Dataset):
      def __init__(self, txt, tokenizer, max_length, stride):
          # Initialization logic
  ```

---

## Next Steps

- Explore BPE in depth ([Hugging Face Course](https://huggingface.co/learn/llm-course/en/chapter6/5)).
- Experiment with data sampling parameters for optimal performance.

---

## References

- [GPT-3 Training](https://arxiv.org/abs/2203.02155)
- [Attention Mechanism](https://arxiv.org/html/1706.03762v7)

---


