# Train your first GPT

This is a small, readable entry point alongside [Karpathy's build-nanogpt](https://github.com/karpathy/build-nanogpt). It trains randomly initialized weights using the GPT-2 architecture: causal attention, learned positional embeddings, pre-layer normalization, GELU MLPs, and tied input/output embeddings. The original upstream scripts are unchanged.

Start with the included Tiny Shakespeare text and a character tokenizer. This lets you learn the entire training loop on CPU. It is an educational experiment; a small run will produce mostly gibberish, and this dataset cannot reproduce GPT-2's capabilities.

## Prepare your environment and data

```bash
cd gpt-from-scratch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r starter/requirements.txt
```

Use Python 3.11 or newer. These installation commands target CPU; see the GPU setup instructions below for other hardware. If you have already created the environment, just activate it.

Prepare the character dataset from the included text:

```bash
python -m starter.data
```

Preparation splits the raw text 90/10 before tokenization. The character vocabulary is fitted on training text only. Existing prepared files are protected against overwriting. For your own UTF-8 corpus, use `--input path/to/text.txt --output data/my-text.pt`, then pass `--data data/my-text.pt` to training.

## Train and generate

Run a small experiment (choose a new output directory for each fresh run):

```bash
python -m starter.train --out-dir runs/first --steps 500 \
  --batch-size 8 --block-size 128 --layers 2 --heads 4 --embed 128
python -m starter.sample --checkpoint runs/first/best.pt \
  --prompt "ROMEO:" --tokens 300
```

Defaults limit CPU use to four threads. Use `--threads 2` for a lighter run. `--device auto` selects CUDA, then Apple MPS, then CPU; you can explicitly select a device. CUDA uses BF16 autocast where supported, otherwise FP32. GPU paths have not been verified on this CPU-only host.

Each run writes:

- `latest.pt`: most recently evaluated checkpoint, including optimizer and random state.
- `best.pt`: checkpoint with the lowest sampled validation loss in this output directory.
- `config.json`: architecture, dataset fingerprint, and training settings.
- `metrics.jsonl`: training loss, validation loss, gradient norm, and throughput.

Resume with 500 **additional** optimizer updates:

```bash
python -m starter.train --resume runs/first/latest.pt --steps 500 --batch-size 8
```

Model dimensions and optimizer state are restored; the dataset must match. Keep the same batch size, gradient accumulation, and device for a reproducible continuation. These CLI settings are not automatically restored. Learning rate is restored unless overridden with `--lr`. Evaluation repeats at the resume step. To branch from an older checkpoint, supply a new `--out-dir`; `best.pt` in that directory is only written when the checkpoint's previous best validation loss is beaten. `latest.pt` is always written.

Validation samples fixed windows from held-out text, not the whole validation set. Its random generator is separate from training. Training uses AdamW, a constant learning rate (default `3e-4`), gradient clipping, and optional accumulation. Effective tokens per optimizer update = batch size × context length × `--grad-accum`.

## Read and experiment

1. `data.py`: token IDs, train/validation separation, and shifted input/target windows.
2. `model.py`: follow `(batch, time, channels)` through embeddings, attention, MLPs, and the vocabulary projection. Cross-entropy trains every position to predict its next token.
3. `train.py`: forward pass → loss → backward → clipping → optimizer step; then validation and checkpoints.
4. `sample.py`: use the last position's logits to sample a token, append it, and repeat.

Start by varying context length or depth while keeping the other settings fixed. Compare held-out loss and generated samples. Then try a longer run and the GPT-2 tokenizer. Changing tokenizers changes the prediction units, so their raw loss values are not directly comparable.

Run the focused checks:

```bash
python -m unittest starter.test_model -v
```

These check causal masking, next-token alignment, held-out vocabulary handling, invalid dimensions, and whether a tiny model can learn a predictable sequence.

## Verified locally

On CPU with PyTorch 2.10.0, all five focused checks passed. A 106,368-parameter character model completed 100 updates; fixed-window validation loss fell from 4.1957 to 3.0908. Checkpoints and generated samples are local artifacts and are not included in the repository. The output is still mostly gibberish at this stage. To repeat the experiment in a new directory:

```bash
python -m starter.train --out-dir runs/smoke-repeat --steps 100 \
  --batch-size 4 --block-size 32 --layers 2 --heads 2 --embed 64 \
  --eval-every 25 --eval-batches 5 --threads 2
```

A separate CPU check compared five continuous updates against three updates followed by two resumed updates, using gradient accumulation of two. All model tensors and optimizer-state tensors matched exactly. GPT-2 BPE preparation, two tiny-model training updates, and checkpoint-based generation were also exercised; this only checks that the BPE path runs.

## Move toward the lecture's scale

Prepare the GPT-2 BPE tokenizer (50,257 vocabulary entries; first use downloads tokenizer assets into `.cache/tiktoken`):

```bash
python -m starter.data --tokenizer gpt2
python -m starter.train --data data/shakespeare-gpt2.pt \
  --out-dir runs/bpe-small --steps 500 --block-size 128 \
  --layers 2 --heads 4 --embed 128 --batch-size 4
```

For a GPU with sufficient memory, this is an example of the 124M-parameter architecture; it has **not** been launched here:

```bash
python -m starter.train --data data/shakespeare-gpt2.pt \
  --out-dir runs/gpt2-experiment --device cuda \
  --layers 12 --heads 12 --embed 768 --block-size 1024 \
  --batch-size 1 --grad-accum 8 --steps 1000
```

Memory requirements depend on hardware and precision. This is still a Shakespeare experiment, not a GPT-2 reproduction. The starter holds its corpus in RAM and uses a single device. For large-scale pretraining, study upstream `fineweb.py` and `train_gpt2.py`: those add sharded data loading, a learning-rate schedule, distributed training, compilation, and HellaSwag evaluation. FineWeb preprocessing targets 10 billion tokens (about 20 GB of uint16 shards plus download/cache storage). It is not part of the quick start and does not feed directly into this starter's dataset format.

## Set up on another machine

Use Python 3.11 or newer. For an independent CPU environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r starter/requirements.txt
```

For NVIDIA hardware, install the appropriate PyTorch wheel using the [PyTorch installation selector](https://pytorch.org/get-started/locally/) before installing the remaining requirements. This starter is tested with PyTorch 2.10.0 and tiktoken 0.12.0.

The model implementation is adapted from upstream commit `6104ab1b53920f6e2159749676073ff7d815c1fa`. Upstream's README declares the MIT license; that checkout does not include a separate LICENSE file. All datasets, checkpoints, tokenizer caches, and the environment remain under this repository and are git-ignored.
