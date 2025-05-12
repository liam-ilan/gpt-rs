//! Transformer model.
//!
//! Resources:
//! - Attention is All You Need (<https://arxiv.org/abs/1706.03762>).
//! - Residual Dropout: A Simple Approach to Improve Transformer’s Data Efficiency (<https://aclanthology.org/2024.sigul-1.35.pdf>).
//! - On Layer Normalization in the Transformer Architecture (<https://arxiv.org/pdf/2002.04745>).
//! - 3Blue1Brown's Attention in transformers (<https://youtu.be/eMlx5fFNoYc>).
//! - Andrej Karpathy's Let's build GPT (<https://youtu.be/kCc8FmEb1nY>).
//! - The tch-rs min-gpt example (<https://github.com/LaurentMazare/tch-rs/tree/main/examples/min-gpt>).

use serde::{Deserialize, Serialize};
use tch::{
    nn::{self, ModuleT},
    Kind, Tensor,
};

/// Configuration for a [`transformer`].
#[derive(Debug, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Number of tokens.
    pub vocab_size: u32,

    /// Embedding size.
    pub embedding_size: u32,

    /// Context length.
    pub context_length: u32,

    /// Number of attention heads per block.
    pub head_count: u32,

    /// Number of blocks.
    pub block_count: u32,

    /// Number to multiply the `embedding_size` by,
    /// to obtain the hidden layer dimension size
    /// in the feed forward layer of an attention block.
    pub feed_forward_multiplier: u32,

    // Dropouts - see "Residual Dropout: A Simple Approach to Improve Transformer’s Data Efficiency".
    /// Dropout to apply post-embedding.
    pub embedding_dropout: f64,

    /// Dropout to apply after computing the attention weights.
    pub attention_dropout: f64,

    /// Dropout to apply after the feed forward layer of each block.
    pub feed_forward_dropout: f64,

    /// Dropout to apply on each residual path.
    pub residual_dropout: f64,
}

// Explicitly manage which parameters do apply weight decay to.
// Used to omit layer normalization, embedding, and bias from weight decay.
/// Group ID to not apply weight decay to.
pub const NO_WEIGHT_DECAY_GROUP: usize = 0;

/// Group ID to apply weight decay to.
pub const WEIGHT_DECAY_GROUP: usize = 1;

// Also create our own linear layer instantiators so we can control weight decay.
// See the tch-rs min-gpt example.
/// Linear layer, with no weight decay on bias terms.
fn linear(var_store: &nn::Path, in_dim: i64, out_dim: i64) -> nn::Linear {
    let weight_decay = var_store.set_group(WEIGHT_DECAY_GROUP);
    let no_weight_decay = var_store.set_group(NO_WEIGHT_DECAY_GROUP);

    nn::Linear {
        ws: weight_decay.randn("weight", &[out_dim, in_dim], 0.0, 0.02),
        bs: Some(no_weight_decay.zeros("bias", &[out_dim])),
    }
}

/// Linear layer with no bias.
fn linear_no_bias(var_store: &nn::Path, in_dim: i64, out_dim: i64) -> nn::Linear {
    let weight_decay = var_store.set_group(WEIGHT_DECAY_GROUP);

    nn::Linear {
        ws: weight_decay.randn("weight", &[out_dim, in_dim], 0.0, 0.02),
        bs: None,
    }
}

/// Feed forward layer.
///
/// Composed of:
/// - The input and output sizes are the embedding size.
/// - There is one hidden layer with `embedding_size` * `feed_forward_multiplier`
/// - There is a RELU activation function on the hidden layer.
/// - There is a dropout layer at the end.
///
/// The input and output are of shape `(..., embedding_size)`.
fn feed_forward(var_store: &nn::Path, config: &TransformerConfig) -> impl nn::ModuleT {
    // Extract config.
    let input_output_size = config.embedding_size as i64;
    let hidden_size = input_output_size * config.feed_forward_multiplier as i64;
    let feed_forward_dropout = config.feed_forward_dropout;

    nn::seq_t()
        .add(linear(
            &(var_store / "input"),
            input_output_size,
            hidden_size,
        ))
        .add_fn(|input| input.relu())
        .add(linear(
            &(var_store / "output"),
            hidden_size,
            input_output_size,
        ))
        .add_fn_t(move |input, train| input.dropout(feed_forward_dropout, train))
}

/// Single attention head.
///
/// The input is of shape `(..., context_length, embedding_size)`.
///
/// The output is of shape `(..., context_length, embedding_size / head_count)`.
fn attention_head(var_store: &nn::Path, config: &TransformerConfig) -> impl nn::ModuleT {
    // Extract config.
    let embedding_size = config.embedding_size as i64;
    let head_size = embedding_size / config.head_count as i64;
    let context_length = config.context_length as i64;
    let attention_dropout = config.attention_dropout;

    // Layers for query, key, and value.
    let query = linear(&(var_store / "query"), embedding_size, head_size);

    let key = linear(&(var_store / "key"), embedding_size, head_size);

    let value = linear(&(var_store / "value"), embedding_size, head_size);

    // Mask used to ensure future tokens have no affect on past ones.
    // If omit_mask[token_2_index][token_1_index] is true,
    // the attention head should omit the item.
    // ie.
    //   0     1     2
    // 0 false true true
    // 1 false false true
    // 2 false false false
    // Implies token 1 cannot affect token 0,
    // but token 1 can affect token 2.
    let omit_mask = Tensor::ones(
        [context_length, context_length],
        (Kind::Bool, var_store.device()),
    )
    .triu(1);

    nn::func_t(move |input, train| {
        // Note: dimension -1 is the embedding dimension,
        // dimension -2 is the context.

        // Compute key, query, and value vectors.
        // Input tensor goes from dimensions `(..., context_length, embedding_size)`,
        // to `(..., context_length, head_size)`.
        let key = key.forward_t(input, train);
        let query = query.forward_t(input, train);
        let value = value.forward_t(input, train);

        // Compute dot product between columns of key and columns of query,
        // ie.
        // ---q1---       ---k1---     k1 * q1, k1 * q2, k1 * q3
        // ---q2---  and  ---k2--- ->  k2 * q1, k2 * q2, k2 * q3
        // ---q3---       ---k3---     k3 * q1, k3 * q2, k3 * q3

        // Do this by transposing the key matrix, then doing a matrix multiplication,
        // ie.
        // ---q1---     |  |  |      k1 * q1, k1 * q2, k1 * q3
        // ---q2---  .  k1 k2 k3  =  k2 * q1, k2 * q2, k2 * q3
        // ---q3---     |  |  |      k3 * q1, k3 * q2, k3 * q3

        // At the end of this, "attention" is a `(..., context_length, context_length)` shape.
        // attention[...][token_2_index][token_1_index] tells you how much token_1 attends to token_2,
        // ie. how much token_1 is "relevant" to token_2.
        // Each value is from -inf to inf.
        let key = key.transpose(-2, -1);
        let attention = query.matmul(&key);

        // Scale down every attention value by the square root of the head size.
        let attention = attention / (head_size as f64).powf(0.5);

        // Tokens should not be influenced by tokens ahead of them.
        // To indicate this, if token_1_index > token_2_index,
        // we set attention[...][token_2_index][token_1_index] to -infinity.
        let attention = attention.masked_fill(&omit_mask, f64::NEG_INFINITY);

        // Apply a softmax along each column,
        // so that now attention[...][token_2_index] is a probability distribution.
        let attention = attention.softmax(-1, Kind::Float);

        // Add a dropout layer before the value matrix is introduced.
        let attention = attention.dropout(attention_dropout, train);

        // Return the dot product of the attention weights and value matrix.
        // This projects the attention matrix down to the head size dimension,
        // ie. the matrix is of shape `(..., context_length, head_size)`.
        // `(..., context_length, context_length)` . `(..., context_length, head_size)`
        // -> `(..., context_length, head_size)``
        attention.matmul(&value)
    })
}

/// Multiple attention heads in parallel.
///
/// Mixes information between heads with a final linear layer.
///
/// The input and output are of shape `(..., context_length, embedding_size)`.
fn multiple_attention_head(var_store: &nn::Path, config: &TransformerConfig) -> impl nn::ModuleT {
    // Extract config.
    let embedding_size = config.embedding_size as i64;
    let head_count = config.head_count;

    // Create specified attention heads.
    let heads_var_store = &(var_store / "heads");
    let attention_heads = (0..head_count)
        .map(|index| attention_head(&(heads_var_store / index), config))
        .collect::<Vec<_>>();

    // Final projection linear layer.
    let projection = linear(&(var_store / "projection"), embedding_size, embedding_size);

    nn::func_t(move |input, train| {
        // Forward all attention heads.
        let res = attention_heads
            .iter()
            .map(|attention_head| attention_head.forward_t(input, train))
            .collect::<Vec<_>>();

        // Concatenate all of them along the `head_size` dimension.
        // Since the `head_size` dimension is the `embedding_size` / `head_count`,
        // The concatenation results in a ... `context_length` * `embedding_size` shaped matrix.
        let res = Tensor::cat(res.as_slice(), -1);

        // Finally forward via the projection layer,
        // to mix information between heads.
        projection.forward_t(&res, train)
    })
}

/// A single block of attention.
///
/// Composed of
/// - A multi-attention head.
/// - A feed-forward layer.
/// - Pre-LN Layernorm layers in between.
///     - See "On Layer Normalization in the Transformer Architecture"
///
/// The result of these layers is summed with the input.
///
/// The input and output are of shape `(..., context_length, embedding_size)`.
fn block(var_store: &nn::Path, config: &TransformerConfig) -> impl nn::ModuleT {
    let embedding_size = config.embedding_size as i64;
    let residual_dropout = config.residual_dropout;

    let multi_attention_head_layer_norm = nn::layer_norm(
        var_store / "layer_norm_1",
        vec![embedding_size],
        Default::default(),
    );

    let multi_attention_head =
        multiple_attention_head(&(var_store / "multi_attention_head"), config);

    let feed_forward_layer_norm = nn::layer_norm(
        var_store / "layer_norm_2",
        vec![embedding_size],
        Default::default(),
    );

    let feed_forward = feed_forward(&(var_store / "feed_forward"), config);

    nn::func_t(move |input, train| {
        let result = input;

        // Apply dropout to residual connection.
        let attention_residual = result.dropout(residual_dropout, train);

        // Apply multi-attention head with pre-layer-norm.
        let attention = multi_attention_head.forward_t(
            &multi_attention_head_layer_norm.forward_t(result, train),
            train,
        );

        // Add residual and attention.
        let result = attention_residual + attention;

        // Apply dropout to another residual connection.
        let feed_forward_residual = result.dropout(residual_dropout, train);

        // Apply feed-forward with pre-layer-norm.
        let feed_forward =
            feed_forward.forward_t(&feed_forward_layer_norm.forward_t(&result, train), train);

        // Add residual and feed-forward and return.
        feed_forward_residual + feed_forward
    })
}

/// The final transformer.
///
/// Composed of:
/// - A token and positional embedding, summed together.
/// - A sequential series of blocks.
/// - A final layer normalization and linear layer, to convert from embedding to logits.
///
/// The input is of shape `(..., context_length)`.
///
/// The output is of shape `(..., context_length, vocab_size)`.
pub fn transformer(var_store: &nn::Path, config: &TransformerConfig) -> impl nn::ModuleT {
    // By default, do not apply weight decay.
    // All weight decay will be explicitly allowed.
    let var_store = &var_store.set_group(NO_WEIGHT_DECAY_GROUP);

    // Extract config.
    let block_count = config.block_count;
    let embedding_dropout = config.embedding_dropout;
    let vocab_size = config.vocab_size as i64;
    let context_length = config.context_length as i64;
    let embedding_size = config.embedding_size as i64;

    // Create embedding for tokens.
    let token_embedding = nn::embedding(
        var_store / "token_embedding",
        vocab_size,
        embedding_size,
        Default::default(),
    );

    // Create embedding for position to give positional context to gpt.
    let position_embedding = nn::embedding(
        var_store / "position_embedding",
        context_length,
        embedding_size,
        Default::default(),
    );

    // Create blocks.
    let blocks = {
        let mut blocks = nn::seq_t();
        let blocks_var_store = &(var_store / "blocks");
        for index in 0..block_count {
            blocks = blocks.add(block(&(blocks_var_store / index), config));
        }
        blocks
    };

    // Final normalization.
    let layer_norm = nn::layer_norm(
        var_store / "layer_norm",
        vec![embedding_size],
        Default::default(),
    );

    // Final conversion from embeddings to logits.
    // No bias, since softmax is applied to logits to obtain probabilities,
    // and softmax is invariant to bias shifts.
    let linear_head = linear_no_bias(&(var_store / "linear_head"), embedding_size, vocab_size);

    // Store indicies of input context on device.
    let indicies = Tensor::arange(context_length, (Kind::Int64, var_store.device()));

    nn::func_t(move |input, train| {
        // Get embedding.
        let token_embedding = token_embedding.forward_t(input, train);
        let position_embedding = position_embedding.forward_t(&indicies, train);
        let embedding = position_embedding + token_embedding;

        // Apply dropout.
        let embedding = embedding.dropout(embedding_dropout, train);

        // Run through blocks, then run a layer normilzation.
        let blocks_res = layer_norm.forward_t(&blocks.forward_t(&embedding, train), train);

        // Convert from embedding to logits.
        linear_head.forward_t(&blocks_res, train)
    })
}
