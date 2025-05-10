# `gpt-rs`
An implementation of a casual decoder transformer and BPE tokeinzer.

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

> Once upon a time, there was a little boy named Timmy. Timmy loved to play with his toys to drink, especially his favorite toy truck. One day, Timmy's mommy said, "Timmy, I'm going to the park with locks." Timmy was very excited and started to play with his toys all day.
> 
> As they were playing, Timmy saw a shovel near the corner of the grass. He asked his mommy if he could buy it on the floor. Billy said yes, but Timmy said no.
> 
> After dinner, Timmy went outside and found a nice jeep. She asked Timmy to take his toy car and they fold it together. Timmy was so happy that he had found the toy car to sit in the mailbox.
> 
> After a while, Timmy went back home and found it under the cereal bed. He started to cry and his mom said, "Wow, mommy! Give it back to your room!" Timmy smiled and said, "Thank you, Timmy! You can make it go too fast." From that day on, Timmy always made sure to always ask his mom if he could have a new toy for his sister." 
> 
> Timmy listened to his mom and said, "I will go to the store to play with my
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