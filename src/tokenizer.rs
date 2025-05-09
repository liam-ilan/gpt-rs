//! An implementation of the Byte Pair Encoding algorithim.
//!
//! See <https://en.wikipedia.org/wiki/Byte_pair_encoding>.

use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    fs,
    io::Write,
    ops::Range,
};

use anyhow::bail;
use rand::{seq::IndexedRandom, Rng};
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Tensor};

/// Serialize a [`Vec`] of 1d, [`i64`] [`Tensor`]s.
fn serialize_encoded_data<S>(unserialized: &[Tensor], serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let data = unserialized
        .iter()
        .map(|tensor| Vec::<i64>::try_from(tensor).expect("Tensor should be 1d, and of type i64."))
        .collect::<Vec<_>>();

    data.serialize(serializer)
}

/// Deserialize a [`Vec`] of 1d, [`i64`] [`Tensor`]s.
fn deserialize_encoded_data<'de, D>(deserializer: D) -> Result<Vec<Tensor>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let data = Vec::<Vec<i64>>::deserialize(deserializer)?;

    Ok(data
        .iter()
        .map(|entry| Tensor::from_slice(entry.as_slice()))
        .collect::<Vec<_>>())
}

/// Representation of a single BPE merging.
#[derive(Serialize, Deserialize, Debug)]
struct Merging {
    /// Pair to translate from.
    from: [usize; 2],

    /// Resulting ID to encode.
    to: usize,
}

/// A tokenized dataset.
#[derive(Serialize, Deserialize, Debug)]
pub struct Dataset {
    /// Character-by-character encoding, mapping from char to id.
    char_to_id: HashMap<char, usize>,

    /// Character-by-character encoding, mapping from id to char.
    id_to_char: HashMap<usize, char>,

    /// IDs to merge, in order.
    mergings: Vec<Merging>,

    /// Encoded dataset.
    #[serde(
        deserialize_with = "deserialize_encoded_data",
        serialize_with = "serialize_encoded_data"
    )]
    encoded_data: Vec<Tensor>,
}

impl Dataset {
    /// Open a dataset from a `.ron` file.
    pub fn from_ron(path: impl AsRef<std::path::Path>, device: Device) -> anyhow::Result<Self> {
        // Make sure encoded data is moved to correct device.
        let mut res: Self = ron::from_str(fs::read_to_string(path)?.as_str())?;
        res.encoded_data = res
            .encoded_data
            .iter()
            .map(|entry| entry.to_device(device))
            .collect();
        Ok(res)
    }

    // Save to a `.ron` file.
    pub fn to_ron(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let content = ron::to_string(self)?;
        let mut file = fs::File::create_new(path)?;
        let write_len = file.write(content.as_bytes())?;
        if write_len != content.len() {
            bail!("Failed to write full .ron file for dataset.");
        }

        Ok(())
    }

    /// Tokenize a bunch of [`String`]s into `token_count` tokens.
    pub fn new(data: Vec<String>, token_count: usize, device: Device) -> anyhow::Result<Self> {
        // Start by collecting char-by-char encodings.
        println!("Encoding data char-by-char.");
        let id_to_char = data
            .join("")
            .chars()
            .collect::<HashSet<char>>()
            .into_iter()
            .enumerate()
            .collect::<HashMap<usize, char>>();

        let char_to_id = id_to_char
            .iter()
            .map(|(key, item)| (*item, *key))
            .collect::<HashMap<char, usize>>();

        // Keep track of ids that should not merge into new tokens.
        let ids_not_to_merge = ['"', '\n']
            .iter()
            .filter_map(|id| char_to_id.get(id))
            .copied()
            .collect::<Vec<_>>();

        // Keep track of tokens that start with spaces (starting with just ' ').
        // We will not allow tokens to merge with spaces, unless the spaces are at the start.
        let mut space_starting_tokens = [' ']
            .iter()
            .filter_map(|id| char_to_id.get(id))
            .copied()
            .collect::<Vec<_>>();

        if char_to_id.len() > token_count {
            bail!(
                "There are {} unique chars in data, but the requested token count was {}.",
                char_to_id.len(),
                token_count
            )
        }

        // Encode all data.
        let mut encoded_data = data
            .iter()
            .map(|entry| {
                entry
                    .chars()
                    .map(|c| char_to_id[&c])
                    .collect::<Vec<usize>>()
            })
            .collect::<Vec<_>>();

        // While we need more tokens,
        let mut mergings = Vec::<Merging>::new();
        let mut current_token_count = char_to_id.len();
        while current_token_count < token_count {
            println!("Creating token #{}.", current_token_count + 1);

            // Count pairs in data.
            let mut pair_counts: HashMap<[usize; 2], usize> = HashMap::new();
            for entry in encoded_data.as_slice() {
                for pair in entry.windows(2) {
                    let pair = [pair[0], pair[1]];

                    // Skip pairs with non-merge tokens.
                    if pair.iter().any(|id| ids_not_to_merge.contains(id)) {
                        continue;
                    }

                    // Skip pairs who's second item has a space at the start.
                    if space_starting_tokens.contains(&pair[1]) {
                        continue;
                    }

                    match pair_counts.entry(pair) {
                        Entry::Occupied(mut entry) => {
                            *entry.get_mut() += 1;
                        }
                        Entry::Vacant(entry) => {
                            entry.insert(1);
                        }
                    }
                }
            }

            // Find most common pair.
            let Some((most_common_pair, _)) = pair_counts
                .into_iter()
                .max_by(|(_, count_1), (_, count_2)| count_1.cmp(count_2))
            else {
                bail!("Not enough data supplied to find a most common pair.")
            };

            // Create new merging.
            mergings.push(Merging {
                from: most_common_pair,
                to: current_token_count,
            });
            println!(
                "Merging tokens {} and {} to {}.",
                most_common_pair[0], most_common_pair[1], current_token_count
            );

            // Re-encode data.
            for entry in encoded_data.as_mut_slice() {
                // Prepare old entry for analysis.
                let old_entry = entry.clone();
                let mut old_entry = old_entry.iter().enumerate().peekable();

                // Run through each pair.
                let mut lost_tokens_count = 0;
                while let (Some((first_index, &first_id)), Some(&(second_index, &second_id))) =
                    (old_entry.next(), old_entry.peek())
                {
                    // Update indicies to account for tokens lost to prior merges.
                    let (first_index, second_index) = (
                        first_index - lost_tokens_count,
                        second_index - lost_tokens_count,
                    );

                    // Extract pair.
                    let pair = [first_id, second_id];
                    if pair == most_common_pair {
                        // If the pair matches, splice in the new id.
                        entry.splice(first_index..=second_index, [current_token_count]);
                        lost_tokens_count += 1;
                        old_entry.next();
                    }
                }
            }

            // If the most common pair has a space at the start, track it.
            if space_starting_tokens.contains(&most_common_pair[0]) {
                space_starting_tokens.push(current_token_count);
            }

            current_token_count += 1;
        }

        // Convert encoded data to list of tensors.
        let encoded_data = encoded_data
            .into_iter()
            .map(|entry| {
                let entry = entry.iter().map(|&id| id as f64).collect::<Vec<_>>();
                Tensor::from_slice(entry.as_slice())
                    .to_kind(Kind::Int64)
                    .to_device(device)
            })
            .collect::<Vec<_>>();

        Ok(Self {
            char_to_id,
            id_to_char,
            mergings,
            encoded_data,
        })
    }

    /// Encode an entry to a tokenized tensor.
    /// Returns a tensor of shape `(entry.len())`.
    pub fn encode(&self, entry: &str, device: Device) -> Tensor {
        // Encode data char-by-char.
        let mut entry = entry
            .chars()
            .map(|c| self.char_to_id[&c])
            .collect::<Vec<usize>>();

        // Apply mergings.
        for merging in self.mergings.as_slice() {
            // Prepare old entry for analysis.
            let old_entry = entry.clone();
            let mut old_entry = old_entry.iter().enumerate().peekable();

            // Run through each pair.
            let mut lost_tokens_count = 0;
            while let (Some((first_index, &first_id)), Some(&(second_index, &second_id))) =
                (old_entry.next(), old_entry.peek())
            {
                // Update indicies to account for tokens lost to prior merges.
                let (first_index, second_index) = (
                    first_index - lost_tokens_count,
                    second_index - lost_tokens_count,
                );

                // Extract pair.
                let pair = [first_id, second_id];
                if pair == merging.from {
                    // If the pair matches, splice in the new id.
                    entry.splice(first_index..=second_index, [merging.to]);
                    lost_tokens_count += 1;
                    old_entry.next();
                }
            }
        }

        // Convert to tensor.
        let entry = entry.into_iter().map(|id| id as i64).collect::<Vec<_>>();
        Tensor::from_slice(entry.as_slice())
            .to_kind(Kind::Int64)
            .to_device(device)
    }

    /// Decode a tokenized tensor back into a string.
    /// Expects an input of shape `(entry_length)`.
    pub fn decode(&self, encoded_entry: &Tensor) -> anyhow::Result<String> {
        // Convert to vec.
        let mut encoded_entry = Vec::<i64>::try_from(encoded_entry)?
            .iter()
            .map(|id| *id as usize)
            .collect::<Vec<usize>>();

        // Run through mergings in reverse.
        for merging in self.mergings.iter().rev() {
            // Reverse merging.
            encoded_entry = encoded_entry
                .split(|&id| id == merging.to)
                .collect::<Vec<&[usize]>>()
                .join(&merging.from[..]);
        }

        // Decode char-by-char.
        Ok(encoded_entry
            .iter()
            .map(|id| self.id_to_char[id])
            .collect::<String>())
    }

    /// Get random batch from the dataset,
    /// of dimension `(contexts_per_batch, context_length + 1)`,
    /// sampled within the provided range of entries.
    pub fn get_batch(
        &self,
        rng: &mut impl Rng,
        entry_range: Range<usize>,
        contexts_per_batch: usize,
        context_length: u32,
    ) -> Tensor {
        let batch_length = context_length as i64 + 1;

        // Slice data within provided range.
        let encoded_data = &self.encoded_data[entry_range];

        // Filter out entries which are shorter than required.
        let encoded_data = encoded_data
            .iter()
            .filter(|&entry| entry.size()[0] >= batch_length)
            .collect::<Vec<_>>();

        // Select random entries.
        let entries = encoded_data.choose_multiple(rng, contexts_per_batch);

        // Select random `batch_length` sections within those entries.
        let samples = entries
            .map(|&entry| {
                let entry_length = entry.size()[0];
                let start = rand::random_range(0..(entry_length - batch_length));
                entry.narrow(-1, start, batch_length)
            })
            .collect::<Vec<_>>();

        // Convert to tensor.
        Tensor::stack(samples.as_slice(), 0)
    }

    /// Get the number of entries in this dataset.
    pub fn entry_count(&self) -> usize {
        self.encoded_data.len()
    }

    /// Get the number of tokens in this dataset.
    pub fn token_count(&self) -> usize {
        self.char_to_id.len() + self.mergings.len()
    }
}
