//! Generate text.

use tch::{nn::ModuleT, Device, Kind};

use crate::{model, tokenizer};

/// Mutate `input`, adding `tokens_to_generate` tokens.
pub fn generate(
    input: &mut String,
    tokens_to_generate: u32,
    transformer_config: &model::TransformerConfig,
    dataset: &tokenizer::Dataset,
    device: Device,
    transformer: &impl ModuleT,
) -> anyhow::Result<()> {
    for _ in 0..tokens_to_generate {
        // Select last `context_length` chars.
        let context = &input[input
            .len()
            .saturating_sub(transformer_config.context_length as usize)..];

        // Encode.
        let context = dataset.encode(context, device);
        let real_context_length = context.size()[0];

        // Pad context with arbritrary char so that transformer can accept it.
        let context = context.pad(
            [
                0,
                transformer_config.context_length as i64 - real_context_length,
            ],
            "constant",
            0.,
        );

        // Forward through model.
        let logits = transformer.forward_t(&context, false);
        let probabilities = logits.softmax(-1, Kind::Float);
        let sample = probabilities
            .multinomial(1, false)
            .view([transformer_config.context_length as i64]);

        // Grab last token from input context.
        let next_token = sample.get(real_context_length - 1).view([1]);

        // Decode and mutate result.
        let next_token = dataset.decode(&next_token)?;
        input.push_str(next_token.as_str());
    }

    Ok(())
}
