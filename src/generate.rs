//! Generate text.

use std::cmp::min;

use tch::{nn::ModuleT, Device, Kind, Tensor};

use crate::{model, tokenizer};

/// Generate `tokens_to_generate` tokens, add them to `input`, and return the result.
pub fn generate(
    input: &str,
    tokens_to_generate: u32,
    temperature: f64,
    transformer_config: &model::TransformerConfig,
    dataset: &tokenizer::Dataset,
    device: Device,
    transformer: &impl ModuleT,
) -> anyhow::Result<String> {
    let transformer_context_length = transformer_config.context_length as i64;

    // Encode input.
    println!("Encoding input.");
    let mut result = dataset.encode(input, device);

    for i in 0..tokens_to_generate {
        println!("Generating token {i}.");

        // Select last `context_length` tokens.
        // If less than `context_length` tokens are not available,
        // just use the whole input.
        let result_length = result.size()[0];
        let context_start = (result_length - transformer_context_length).max(0);
        let context_length = min(result_length, transformer_context_length);
        let context = result.narrow(-1, context_start, context_length);

        // Pad context with an arbritrary char so that transformer can accept it.
        let context = context.pad(
            [0, transformer_context_length - context_length],
            "constant",
            0.,
        );

        // Forward through model.
        let logits = transformer.forward_t(&context, false) / temperature;
        let probabilities = logits.softmax(-1, Kind::Float);
        let sample = probabilities
            .multinomial(1, false)
            .view([transformer_config.context_length as i64]);

        // Grab last token from input context.
        let next_token = sample.get(context_length - 1).view([1]);
        result = Tensor::cat(&[&result, &next_token], -1);
    }

    // Decode.
    println!("Decoding result.");
    dataset.decode(&result)
}
