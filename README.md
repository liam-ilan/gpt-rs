# `gpt-rs`
An implementation of a causal decoder transformer and BPE tokeinzer.

*Built with Rust 🦀 and [`tch-rs`](https://github.com/LaurentMazare/tch-rs) 🔥.*

## Quickstart
This project relies on `just` for script management. Install it with cargo,

```sh
cargo install just
```

Then install `libtorch` locally and setup the correct configuration with:
```sh
just setup-macos-arm64
```

> Currently, there's only a setup script for arm64 macos. If you are not on arm64 macos, Take a look at the [`tch-rs` docs](https://github.com/LaurentMazare/tch-rs) for more information on how to setup libtorch for your machine.

To generate text from a model trained on [`tinystories-10k`](https://huggingface.co/datasets/flpelerin/tinystories-10k), run

```sh
cargo run --release -- generate \
--dataset-file example/dataset_2048_stories.parquet.ron \
--transformer-config-file example/transformer_config.ron \
--token-count 256 \
--input "Once" \
--transformer-safetensors example/trained_transformer.safetensors
```

> Once upon a time, there was a little boy named Timmy. Timmy liked to play outside and play with his toys when he saw a bird in the sky. He was very excited to see what it was where it was, so he asked the bird if it could be like to fly to the clouds. The bird said yes! 
>
> Timmy and his owner went out of the forest and went to the park. They were so happy to see the bird in the park. But then, the bird saw a big bird who was trying to feet it. Timmy and his mom were very scared and wanted to help. They decided to say hello and stay in the woods. They said he could not find the bird's family. They said they could grow. They said they could make the bird happy.
> 
> Later that day, Timmy and his family went to the park to play. They got a new bird in the sky. They were happy and said, "Thank you, mom! You are so kind." They went back to the park. They played with the ball every day.
> 
> After that, Timmy and his family went to the park to play. They saw something very big and wide and round. They decided to pretend it was a big, red car to play. They played with the car and had lots of fun.

## Train a Model from Scratch
### Tokenization
The [`tinystories-10k`](https://huggingface.co/datasets/flpelerin/tinystories-10k) dataset is included in this repository under `./stories.parquet`. Any `.parquet` file, who's first column contains the text entries to train on, can be used.

Start by tokenizing the `stories.parquet` dataset. 
```sh
cargo run --release -- --seed 1234 tokenize \
--data-parquet stories.parquet \
--transformer-config-file example/transformer_config.ron
```
This will generate a file in `datasets/dataset_2048_stories.parquet.ron`, containing the tokenized dataset.

### Training
Next, train a model on the tokenized dataset.
```sh
cargo run --release -- --seed 1234 train \
--dataset-file datasets/dataset_2048_stories.parquet.ron \
--train-config-file example/train_config.ron \
--transformer-config-file example/transformer_config.ron
```
This will generate `.safetensors` checkpoints in a directory under `./training`. The fully trained model's `.safetensors` will be named `final.safetensors`.

> The example model was trained with MPS on an M3 Macbook Air, it took around ~2.5 hours.
> If you are on an MPS-supporting Mac, you can enable it explicitly with `--mps`
>
> ```sh
> cargo run --release -- --seed 1234 --mps train \
> --dataset-file datasets/dataset_2048_stories.parquet.ron \
> --train-config-file example/train_config.ron \
> --transformer-config-file example/transformer_config.ron
> ```

### Generating Text
You can generate text from your trained model with
```sh
cargo run --release -- generate \
--dataset-file datasets/dataset_2048_stories.parquet.ron \
--transformer-config-file example/transformer_config.ron \
--token-count <TOKEN_COUNT> \
--input <INPUT> \
--transformer-safetensors <SAFETENSORS>
```

## Navigating the Codebase
This implementation was built to be as verbose as possible. The core of the model can be found under [`./src/model.rs`](./src/model.rs), heavily inspired by Karpathy's [minGPT](https://github.com/karpathy/minGPT).

Some details:
- 4 dropout layers per block, one post feed forward, one post attention head, and one on each residual [Residual Dropout: A Simple Approach to Improve Transformer’s Data Efficiency](https://aclanthology.org/2024.sigul-1.35.pdf). There is also a dropout applied post embedding.
- Layer norm is applied prior to the multi-attention head/feed forward in each block (Pre-LN), see [On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745).

The BPE tokenizer can be found under [`./src/tokenizer.rs`](./src/tokeinzer.rs). Tokenized datasets are serialized into `.ron` (Rusty-Object-Notation) files for later use. The tokenizer is modified from a classing BPE to include space-prefixing, no merges across words, and some do-not-merge tokens.

The training loop can be found under [`./src/train.rs`](./src/train.rs). It utilizes an AdamW optimizer. There is no learning rate scheduling in this branch, take a look at the [`linear_warmup-cosine-decay` branch](https://github.com/liam-ilan/gpt-rs/tree/linear-warmup-cosine-decay) for an implementation of a more refined scheduler (though it may be out of date with `main`).

## Advanced Usage
Custom models and training strategies can be configured with `.ron` files. See [`./example/train_config.ron`](./example/train_config.ron) and [`./example/transformer_config.ron`](./example/transformer_config.ron) for example configuration.

- Run `cargo run --release -- -h` to get information on what commands are available, and upper level arguments (ie. seed, MPS). 
- Run `cargo run --release -- <COMMAND> -h` to get information on what arguments specific commands take.

## Author
Built by [Liam Ilan](https://www.liamilan.com/), made in Canada 🇨🇦 ❤️.