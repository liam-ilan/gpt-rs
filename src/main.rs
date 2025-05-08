mod generate;
mod model;
mod tokenizer;
mod train;

use std::{
    cell::RefCell,
    fs::{self, File},
    path::{Path, PathBuf},
};

use anyhow::bail;
use parquet::file::reader::{FileReader, SerializedFileReader};
use tch::{nn, Device};

use clap::{Parser, Subcommand};

/// Clap app entry point.
#[derive(Debug, Parser)]
pub struct App {
    /// Subcommand.
    #[clap(subcommand)]
    command: Command,
}

/// Clap app commands.
#[derive(Debug, Subcommand)]
enum Command {
    /// Tokenize a file using bite-pair-encoding,
    /// and produce a dataset.
    Tokenize {
        /// `.parquet` file to tokenize,
        /// Expects a single column of strings.
        #[arg(long)]
        data_parquet: PathBuf,

        /// Transformer configuration.
        #[arg(long)]
        transformer_config_file: PathBuf,
    },

    /// Train a model on a dataset from scratch or from a checkpoint,
    /// and produce `.safetensors` files for each checkpoint.
    Train {
        /// Optional checkpoint to start from.
        #[arg(long)]
        transformer_safetensors: Option<PathBuf>,

        /// Dataset `.ron` file, generated from the `tokenize` command.
        #[arg(long)]
        dataset_file: PathBuf,

        /// Custom training configuration.
        #[arg(long)]
        train_config_file: PathBuf,

        /// Custom transformer configuration.
        #[arg(long)]
        transformer_config_file: PathBuf,
    },

    /// Use a model to generate text.
    Generate {
        /// Dataset `.ron` file, generated from the `tokenize` command.
        #[arg(long)]
        dataset_file: PathBuf,

        /// Saftensors to use.
        #[arg(long)]
        transformer_safetensors: PathBuf,

        /// Custom transformer configuration.
        #[arg(long)]
        transformer_config_file: PathBuf,

        /// Input to supply context from.
        #[arg(long)]
        input: String,

        /// Number of tokens to generate.
        #[arg(long)]
        token_count: u32,
    },
}

const TRAINING_DATA_PERCENTAGE: f64 = 0.9;

fn main() -> anyhow::Result<()> {
    let command = App::parse().command;
    let device = Device::cuda_if_available();

    match command {
        // Tokenize a file and output it.
        Command::Tokenize {
            data_parquet,
            transformer_config_file,
        } => {
            // Load config.
            let config = fs::read_to_string(transformer_config_file)?;
            let config: model::TransformerConfig = ron::from_str(config.as_str())?;

            // Read .parquet file.
            let file = File::open(&data_parquet)?;
            let reader = SerializedFileReader::new(file)?;

            // Extract fields.
            let fields = reader
                .get_row_iter(None)?
                .map(|record| {
                    let record = record?;
                    let columns = record.into_columns();
                    if columns.len() != 1 {
                        bail!("Expected 1 column in each row of .parquet file.")
                    }

                    let (_, field) = &columns[0];
                    Ok(field.to_string())
                })
                .collect::<Result<Vec<String>, _>>()?;

            // Tokenize and create dataset.
            let dataset = tokenizer::Dataset::new(fields, config.vocab_size as usize, device)?;

            // Setup directory.
            let output_directory = Path::new("datasets");
            fs::create_dir_all(output_directory)?;

            // Write serialzed dataset to file.
            let output_path = output_directory.join(format!(
                "dataset_{}_{}.ron",
                dataset.token_count(),
                data_parquet
                    .file_name()
                    .expect("Parquet path should be a file.")
                    .to_str()
                    .expect("Provided Parquet should be utf-8.")
            ));

            // Write `.ron` file.
            dataset.to_ron(output_path)?;
        }

        // Train a model.
        Command::Train {
            transformer_safetensors,
            dataset_file,
            train_config_file,
            transformer_config_file,
        } => {
            let rng = RefCell::new(rand::rng());

            // Load config.
            let transformer_config = fs::read_to_string(transformer_config_file)?;
            let transformer_config: model::TransformerConfig =
                ron::from_str(transformer_config.as_str())?;

            let train_config = fs::read_to_string(train_config_file)?;
            let train_config: train::TrainConfig = ron::from_str(train_config.as_str())?;

            // Create model and var store, and load safetensors if provided.
            let mut var_store = nn::VarStore::new(device);
            let transformer =
                model::transformer(&(var_store.root() / "transformer"), &transformer_config);

            if let Some(transformer_safetensors) = transformer_safetensors {
                var_store.load(transformer_safetensors)?;
            }

            // Deserialize dataset.
            let dataset = tokenizer::Dataset::from_ron(dataset_file, device)?;

            // Find index to split entries on.
            let training_testing_split =
                (dataset.entry_count() as f64 * TRAINING_DATA_PERCENTAGE) as usize;

            // Create getters for training/testing batches.
            let get_training_batch = || {
                dataset.get_batch(
                    &mut rng.borrow_mut(),
                    0..training_testing_split,
                    train_config.contexts_per_batch as usize,
                    transformer_config.context_length,
                )
            };

            let get_testing_batch = || {
                dataset.get_batch(
                    &mut rng.borrow_mut(),
                    training_testing_split..dataset.entry_count(),
                    train_config.contexts_per_batch as usize,
                    transformer_config.context_length,
                )
            };

            // Train.
            train::train(
                &var_store,
                &transformer,
                &transformer_config,
                &train_config,
                get_training_batch,
                get_testing_batch,
            )?;
        }

        // Generate text.
        Command::Generate {
            dataset_file,
            transformer_safetensors,
            input,
            token_count,
            transformer_config_file,
        } => {
            if input.is_empty() {
                bail!("Input must be non-empty.")
            }

            // Load config.
            let config = fs::read_to_string(transformer_config_file)?;
            let config: model::TransformerConfig = ron::from_str(config.as_str())?;

            // Create model and var store.
            let mut var_store = nn::VarStore::new(device);
            let transformer = model::transformer(&(var_store.root() / "transformer"), &config);
            var_store.load(transformer_safetensors)?;

            // Deserialize dataset.
            let dataset = tokenizer::Dataset::from_ron(dataset_file, device)?;

            // Generate.
            let result =
                generate::generate(&input, token_count, &config, &dataset, device, &transformer)?;
            println!("{result}");
        }
    }

    Ok(())
}
