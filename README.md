## How To Use
### Setup
This project relies on `just` for script management. Install it with cargo,

```sh
cargo install just
```

Then install libtorch locally and setup the correct `.cargo/config.toml` with the following,
```sh
just setup-macos-arm64 
```

> Currently, there's only a setup script for arm64 macos. If you are not on arm64 macos, Take a look at the [`tch-rs` docs](https://github.com/LaurentMazare/tch-rs) for more information on how to setup libtorch for your machine.

### TinyStories Example
Let's train a model to mimic the [tinystories-10k](https://huggingface.co/datasets/flpelerin/tinystories-10k) dataset. The dataset is included in this repo as `stories.parquet`.

Tokenize the `stories.parquet` dataset. 
```sh
cargo run -- tokenize \
--data-parquet stories.parquet \
--transformer-config-file example/transformer_config.ron
```
This will generate a file in `datasets/dataset_512_stories.parquet.ron`, containing the tokenized dataset.

Next, train a model on the tokenized dataset.
```sh
cargo run -- train \
--dataset-file datasets/dataset_512_stories.parquet.ron \
--train-config-file example/train_config.ron \
--transformer-config-file example/transformer_config.ron
```
This will generate `.safetensors` checkpoints in a directory under `./training`. The fully trained model's `.safetensors` will be named `final.safetensors`.

We can now generate text with the tokenized dataset.
```sh
cargo run \
--dataset-file datasets/dataset_512_stories.parquet.ron \
--transformer-config-file example/transformer_config.ron \
--token-count <TOKEN_COUNT> \
--input <INPUT> \
--transformer-safetensors <SAFETENSORS>
```