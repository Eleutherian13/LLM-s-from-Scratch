# 🧠 GPT-2 From Scratch — Pretraining, Fine-Tuning & Evaluation

Building a GPT-2-style (124M parameter) Large Language Model from the ground up — no Hugging Face `transformers`, no shortcuts. Every component (tokenization, attention, transformer blocks, training loop, fine-tuning, evaluation) is implemented in raw PyTorch to understand *exactly* what happens inside an LLM.

> 📖 Based on **"Build a Large Language Model (From Scratch)"** by Sebastian Raschka, and the excellent **"Building LLMs from Scratch"** YouTube series by **Dr. Raj Dandekar (Vizuara AI Labs)**. This repo is my personal implementation, notes, and experiments while working through both resources.

---

## 📌 What's in this repo

This project walks through the complete lifecycle of an LLM:

**Tokenization & Data → Attention → Architecture → Pretraining → Loading Pretrained Weights → Fine-Tuning → Evaluation**

| Stage | What it does |
|---|---|
| Data & Tokenization | Custom byte-pair tokenizer, input-target pair generation, sliding-window dataloaders, learned position embeddings |
| Attention Mechanisms | Self-attention from first principles → trainable self-attention → causal (masked) attention → multi-head attention |
| Architecture | Full GPT-2 style decoder-only transformer: layer norm, GELU, feed-forward blocks, residual connections, 124M parameter config |
| Pretraining | Cross-entropy loss function, training/validation loop, loss curve tracking, temperature/top-k decoding strategies |
| Pretrained Weights | Downloading and loading OpenAI's official GPT-2 (124M) weights into the custom architecture |
| Fine-Tuning | Classification fine-tuning (SMS spam detection) and instruction fine-tuning (Alpaca-style instruction-response data) |
| Evaluation | Benchmarking the fine-tuned model's outputs |

---

## 📂 Repository Structure

```
├── PreRequisites/
│
├── 01_data_and_tokenization/
│   ├── SimpleTextTokenizer.ipynb
│   ├── LLM_Data_Preprocessing.ipynb
│   ├── inputTargetPair.ipynb
│   └── Position Embeddings.ipynb
│
├── 02_attention_mechanisms/
│   ├── Attention mechanism.ipynb
│   ├── Trainable Self Attention.ipynb
│   └── Causal Attention.ipynb
│
├── 03_architecture/
│   └── LLM Architecture.ipynb
│
├── 04_pretraining/
│   ├── LLM Loss function.ipynb
│   ├── LLM Pretraining.ipynb
│   ├── LLM Training Validation Loss.ipynb
│   ├── LLM Decoding Strategies.ipynb
│   ├── setiingupGPU.ipynb
│   └── the-verdict.txt              # small demo pretraining corpus (short story)
│
├── 05_pretrained_weights/
│   ├── Model_Weights_Loaded.ipynb
│   └── Loading pretrained weights instruction finetuning.ipynb
│
├── 06_finetuning_classification/
│   ├── Classification finetuning data loading.ipynb
│   ├── Architecture Classification finetuning.ipynb
│   └── sms+spam+collection.zip
│
├── 07_finetuning_instruction/
│   ├── Instruction fine-tuning dataset prep loading.ipynb
│   ├── DataLoaders Instruction finetuning.ipynb
│   ├── instruction-data.json
│   └── LLM finetuning training.ipynb
│
├── 08_evaluation/
│   └── Fine-tuned LLM Evaluation.ipynb
│
├── requirements.txt
├── LICENSE
└── README.md
```

*(Folder numbering shown here is a suggested reorganization for readability — rename/move the notebooks on GitHub to match if you'd like the repo to read top-to-bottom as a course.)*

---

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/gpt2-from-scratch.git
cd gpt2-from-scratch

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Launch any notebook with `jupyter lab` or `jupyter notebook`. Notebooks are numbered/ordered to be run sequentially within each stage.

---

## 🖥️ Training Details & Compute — the honest version

This is the section most people gloss over, so here it is in full.

**Model config used:** GPT-2 small — 124M parameters, 12 transformer layers, 12 attention heads, 768-dim embeddings, context length as configured in each notebook (256 for the pretraining demo, extendable to 1024 to match original GPT-2).

**Hardware:** Laptop with an Intel Core i7-13700HX and an NVIDIA RTX 4050 (6 GB VRAM).

There are two *very different* things happening in this repo, and it's important not to conflate them:

**1. Pretraining from scratch (`LLM Pretraining.ipynb`) — done locally, and genuinely laptop-feasible.**
The model is initialized with **random weights** and trained on a small text corpus (`the-verdict.txt`, a short story — a few tens of thousands of tokens). At 124M parameters, the raw memory footprint for weights + gradients + Adam optimizer states in fp32 is roughly ~2 GB, and with a small batch size and context length of 256 the activation memory comfortably fits inside 6 GB VRAM — this is exactly why Raschka designed the book's pretraining demo to run on an ordinary laptop. This step proves the training loop, loss function, and optimizer all work correctly, and demonstrates the model learning to predict text in the style of the training corpus. **It does not, and is not intended to, reproduce GPT-2's actual language ability** — that would require training on a web-scale corpus (the original GPT-2 was trained on ~40 GB of internet text using data-parallel training across multiple high-end GPUs for an extended period). A single 6 GB laptop GPU cannot realistically pretrain a general-purpose 124M model to GPT-2-level quality in a practical amount of time.

**2. Loading pretrained weights + fine-tuning (`Model_Weights_Loaded.ipynb`, classification & instruction notebooks) — also done locally, and this is where the real capability comes from.**
OpenAI's official GPT-2 (124M) checkpoint is loaded directly into the custom architecture, then fine-tuned for:
- **Classification** — SMS spam vs. ham detection on the SMS Spam Collection dataset
- **Instruction following** — supervised fine-tuning on an Alpaca-style instruction-response dataset (`instruction-data.json`)

Fine-tuning a 124M model that already starts from good pretrained weights, on a small dataset, with a small batch size and mixed precision, is entirely practical on a 6 GB GPU — this is standard practice even for much larger models via techniques like partial-layer fine-tuning.

**If/when I scale to actual large-corpus pretraining:** that's a different compute regime and I'd rent GPU hours from a cloud provider rather than pretend a laptop GPU did it — Indian options like **E2E Networks** or **Jarvislabs.ai** offer pay-as-you-go A100/A6000-class GPUs at a fraction of AWS/GCP pricing, which is the realistic path for anyone doing this from India without enterprise infra.

**Training run specifics (fill in your actual numbers before publishing):**

| Metric | Pretraining (the-verdict.txt) | Classification fine-tune | Instruction fine-tune |
|---|---|---|---|
| Epochs | `<fill in>` | `<fill in>` | `<fill in>` |
| Batch size | `<fill in>` | `<fill in>` | `<fill in>` |
| Context length | `<fill in>` | `<fill in>` | `<fill in>` |
| Final train loss | `<fill in>` | `<fill in>` | `<fill in>` |
| Final val loss | `<fill in>` | `<fill in>` | `<fill in>` |
| Training time | `<fill in>` | `<fill in>` | `<fill in>` |
| Test accuracy / metric | — | `<fill in>` | `<fill in>` |

---

## 📊 Results

Add your loss curves, sample generations (before vs. after fine-tuning), and classification accuracy here. Screenshots or the matplotlib plots from the notebooks work well.

---

## 🛣️ Roadmap

This repo is the **foundation**, not the finish line. Next up:
- [ ] Wrap this fine-tuned GPT-2 base with a **production-grade RAG (Retrieval-Augmented Generation) pipeline** — vector store, chunking strategy, retrieval + generation orchestration
- [ ] Serve the model behind a FastAPI inference endpoint
- [ ] Explore LoRA/parameter-efficient fine-tuning to push further on limited VRAM
- [ ] Benchmark against Hugging Face's GPT-2 checkpoint on held-out tasks

---

## 🙏 Acknowledgements

- Sebastian Raschka — [*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- Dr. Raj Dandekar & [Vizuara AI Labs](https://www.vizuara.ai) — *Building LLMs from Scratch* YouTube series

This repo is a personal learning implementation built while studying both resources — full credit for the pedagogy and structure goes to them.

## 📄 License

See [LICENSE](./LICENSE).
