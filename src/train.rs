//! Train a [`model::transformer`].
//! 
//! Uses an AdamW optimizer,
//! implements a linear warmup and cosine decay.

use std::{f64::consts::PI, fs, path::Path, time::Instant};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tch::{
    nn::{self, ModuleT, OptimizerConfig},
    Tensor,
};

use crate::model;

/// Configuration for [`train`].
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Minimum learning rate for linear warmup with cosine decay.
    pub min_learning_rate: f64,

    /// Maximum learning rate for linear warmup with cosine decay.
    pub max_learning_rate: f64,

    /// Number of iterations to warmup training with.
    pub warmup_iterations: usize,

    /// Number of batches to train on before stopping.
    pub training_iterations: usize,

    /// Number of training/testing batches to use when evaluating the performance of the model.
    pub evaluation_iterations: usize,

    /// Number of iterations to train between each performance evaluation.
    pub evaluation_interval: usize,

    /// Number of contexts computed at the same time in a single batch.
    pub contexts_per_batch: u32,
}

/// Compute a transformer's loss on a given batch using cross-entropy.
///
/// The input batch should be of shape `(contexts_per_batch, context_length + 1)`.
///
/// The first `context_length` tokens are used as the input,
/// the last `context_length` tokens are used as the expected output.
fn compute_loss(
    transformer: &impl ModuleT,
    transformer_config: &model::TransformerConfig,
    train_config: &TrainConfig,
    batch: &Tensor,
    train: bool,
) -> Tensor {
    let context_length = transformer_config.context_length as i64;
    let vocab_size = transformer_config.vocab_size as i64;
    let contexts_per_batch = train_config.contexts_per_batch as i64;

    // Batch is of shape `(batch_size, context_length + 1)`.
    // We extract the first `context_length` chars to be the input,
    // and the last `context_length` chars to be the expected output.
    let input = batch.narrow(-1, 0, context_length);
    let expected_output = batch.narrow(-1, 1, context_length);

    // Compute logits
    let logits = transformer.forward_t(&input, train);

    // Flatten logits and expected output across batches, then compute loss with cross entropy.
    let flattened_logits = logits.view([contexts_per_batch * context_length, vocab_size]);
    let flattened_expected_output = expected_output.reshape([contexts_per_batch * context_length]);

    flattened_logits.cross_entropy_for_logits(&flattened_expected_output)
}

/// Evaluate the performance of the transformer.
///
/// Returns a tuple, of training and testing loss in that order,
/// averaged over a number of batches.
fn evaluate(
    transformer: &impl ModuleT,
    transformer_config: &model::TransformerConfig,
    train_config: &TrainConfig,
    get_training_batch: &mut impl FnMut() -> Tensor,
    get_testing_batch: &mut impl FnMut() -> Tensor,
) -> anyhow::Result<(f64, f64)> {
    // Get training an testing batches.
    let training_evaluation_batches =
        (0..train_config.evaluation_iterations).map(|_| get_training_batch());
    let testing_evaluation_batches =
        (0..train_config.evaluation_iterations).map(|_| get_testing_batch());

    // Compute losses.
    let training_losses = training_evaluation_batches
        .map(|batch| compute_loss(transformer, transformer_config, train_config, &batch, false));
    let testing_losses = testing_evaluation_batches
        .map(|batch| compute_loss(transformer, transformer_config, train_config, &batch, false));

    // Average out losses.
    let training_loss =
        f64::try_from(training_losses.sum::<Tensor>() / train_config.evaluation_iterations as f64)?;
    let testing_loss =
        f64::try_from(testing_losses.sum::<Tensor>() / train_config.evaluation_iterations as f64)?;

    Ok((training_loss, testing_loss))
}

/// Train a [`model::transformer`].
///
/// Saves the intermediate and final transformers as safetensors in the file system.
pub fn train(
    var_store: &nn::VarStore,
    transformer: &impl ModuleT,
    transformer_config: &model::TransformerConfig,
    train_config: &TrainConfig,
    mut get_training_batch: impl FnMut() -> Tensor,
    mut get_testing_batch: impl FnMut() -> Tensor,
) -> anyhow::Result<()> {
    // Create directory to save results to.
    let output_directory = format!(
        "training/run_{}",
        Utc::now().to_string().replace(" ", "_").to_lowercase()
    );
    let output_directory = Path::new(output_directory.as_str());
    fs::create_dir_all(output_directory)?;

    // Initialize optimizer.
    let mut optimizer = nn::AdamW::default().build(var_store, train_config.min_learning_rate)?;
    optimizer.set_weight_decay_group(model::NO_WEIGHT_DECAY_GROUP, 0.);
    optimizer.set_weight_decay_group(model::WEIGHT_DECAY_GROUP, 0.1);

    // Training loop.
    for index in 0..train_config.training_iterations {
        let start = Instant::now();

        // Compute learning rate using cosine-decay with warmup.
        let normailzed_learning_rate = if index < train_config.warmup_iterations {
            index as f64 / train_config.warmup_iterations as f64
        } else {
            let cosine_decay_index = index - train_config.warmup_iterations;
            let cosine_decay_steps = train_config.training_iterations - train_config.warmup_iterations;
            ((PI * cosine_decay_index as f64 / cosine_decay_steps as f64).cos() + 1.) / 2.
        };
        let learning_rate = normailzed_learning_rate * 
                (train_config.max_learning_rate - train_config.min_learning_rate) + 
                train_config.min_learning_rate;

        optimizer.set_lr(learning_rate);

        // Evaluate and save the model every evaluation interval.
        if index % train_config.evaluation_interval == 0 {
            var_store.save(output_directory.join(format!("{index}.safetensors")))?;

            let (training_loss, testing_loss) = evaluate(
                transformer,
                transformer_config,
                train_config,
                &mut get_training_batch,
                &mut get_testing_batch,
            )?;

            println!("Iteration: {index}, Training loss: {training_loss:.3}, Testing loss: {testing_loss:.3}");
        }

        // Step.
        let loss = compute_loss(
            transformer,
            transformer_config,
            train_config,
            &get_training_batch(),
            true,
        );
        optimizer.backward_step_clip(&loss, 0.5);

        let delta = start.elapsed();
        let delta_ms = delta.as_millis();

        println!(
            "Iteration {index} took {delta_ms} ms, learning rate: {learning_rate:.8}, loss: {:.3}",
            f64::try_from(loss)?
        )
    }

    // Evaluate final model.
    var_store.save(output_directory.join("final.safetensors"))?;

    let (training_loss, testing_loss) = evaluate(
        transformer,
        transformer_config,
        train_config,
        &mut get_training_batch,
        &mut get_testing_batch,
    )?;

    println!("Final, Training loss: {training_loss:.3}, Testing loss: {testing_loss:.3}");

    Ok(())
}
