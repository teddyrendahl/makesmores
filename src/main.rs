mod layers;

use crate::layers::{create_wavenet, Layer};
use anyhow::Result;
use itertools::Itertools;
use layers::create_mlp;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use std::collections::{BTreeMap, HashMap};
use tch::{IndexOp, Kind};

const EDGE_TOKEN: char = '.';
const SEED: i64 = 2147483647;
const EMBEDDING_SIZE: i64 = 24;
const CHUNK_SIZE: i64 = 8;
const HIDDEN_LAYER_SIZE: i64 = 128;
const BATCH_SIZE: i64 = 32;
const TRAINING_ITERATIONS: i64 = 200_000;

#[derive(Debug)]
enum ModelType {
    Wave,
    Mlp,
}

fn main() -> Result<()> {
    tch::manual_seed(SEED);
    let device = tch::Device::cuda_if_available();
    let mut names = get_names()?;

    // Generate a mapping of token to integer (and the reverse)
    let (c_to_i, i_to_c) = create_char_maps(&names);
    let vocab_size = (c_to_i.len() + 1) as i64;

    // Split dataset into train, dev, test
    let mut rng = ChaCha12Rng::seed_from_u64(42);
    names.shuffle(&mut rng);
    let test_set_size = (names.len() as f32 * 0.9) as usize;

    let (xtr, ytr) = load_data(&names[0..test_set_size], &c_to_i, CHUNK_SIZE as usize);
    let (xdev, ydev) = load_data(&names[test_set_size..], &c_to_i, CHUNK_SIZE as usize);
    // let mut w = tch::Tensor::randn([27, 27], (tch::Kind::Float, device)).requires_grad_(true);

    // for _ in 0..200 {
    //     let logits = xenc.matmul(&w).exp();
    //     let prob = &logits / logits.sum_dim_intlist([1].as_slice(), true, tch::Kind::Float);
    //     let regularization = w.pow_tensor_scalar(2).mean(tch::Kind::Float);
    //     // Determine the - log likelihood loss
    //     let loss = -prob
    //         .index(&[
    //             Some(tch::Tensor::arange(
    //                 xenc.size()[0],
    //                 (tch::Kind::Int64, device),
    //             )),
    //             Some(yenc.shallow_clone()),
    //         ])
    //         .log()
    //         .mean(tch::Kind::Float)
    //         + regularization * 0.01;
    //     dbg!(&loss);
    //     w.zero_grad();
    //     loss.backward();
    //     tch::no_grad(|| w += w.grad() * -50);
    // }

    let mut model = create_model(ModelType::Mlp, device, vocab_size);

    for i in 0..TRAINING_ITERATIONS {
        let ix = tch::Tensor::randint(xtr.size()[0], [BATCH_SIZE], (Kind::Int, device));
        let samples = xtr.i(&ix);
        let labels = ytr.i(&ix);
        let logits = model.forward(samples);
        let loss = logits.cross_entropy_for_logits(&labels);
        model.zero_grad();
        loss.backward();
        let lr = if i < 100_000 { 0.1 } else { 0.01 };

        tch::no_grad(|| model.backward(lr));

        if i % 10_000 == 0 {
            loss.print()
        }
    }

    tch::manual_seed(SEED + 10);
    model.set_training(false);
    println!("Training loss -> {}", split_loss(xtr, model.as_mut(), &ytr));
    println!(
        "Validation loss -> {}",
        split_loss(xdev, model.as_mut(), &ydev)
    );

    // Generate a name by repeatedly sampling from the probability distribution of bigrams
    // until an end token is found.
    for _ in 0..20 {
        let mut out = Vec::new();
        let mut ctx = tch::Tensor::zeros(CHUNK_SIZE, (Kind::Int, device));
        loop {
            let logits = model.forward(ctx.view([1, CHUNK_SIZE]));
            let prob = logits.softmax(1, Kind::Float);
            let idx = prob.multinomial(1, true).get(0).int64_value(&[0]);
            out.push(i_to_c[&(idx as usize)]);
            if idx == 0 {
                break;
            }
            ctx = tch::Tensor::cat(&[ctx.i(1..), tch::Tensor::from_slice(&[idx])], 0);
        }
        println!("{}", out.into_iter().join(""));
    }
    Ok(())
}

/// Read all of the names from the dataset
fn get_names() -> Result<Vec<String>> {
    Ok(std::fs::read_to_string("names.txt")?
        .lines()
        .map(|s| s.to_string())
        .collect())
}

/// Create a mapping of char to the integer encoding
fn create_char_maps(words: &[String]) -> (BTreeMap<char, usize>, HashMap<usize, char>) {
    let mut c_to_i: BTreeMap<char, usize> = words
        .join("")
        .chars()
        .unique()
        .sorted()
        .enumerate()
        .map(|(i, c)| (c, i + 1))
        .collect();
    c_to_i.insert(EDGE_TOKEN, 0);
    let i_to_c = c_to_i.iter().map(|(c, i)| (*i, *c)).collect();
    (c_to_i, i_to_c)
}

fn load_data(
    words: &[String],
    char_map: &BTreeMap<char, usize>,
    block_size: usize,
) -> (tch::Tensor, tch::Tensor) {
    let mut xs = vec![];
    let mut ys = vec![];

    for name in words.iter() {
        let mut ctx = vec![0_i8; block_size];
        for c in name.chars().chain(std::iter::once(EDGE_TOKEN)) {
            let ci = char_map[&c] as i8;
            xs.push(ctx.clone());
            ys.push(ci);

            ctx.rotate_left(1);
            ctx[block_size - 1] = ci;
        }
    }
    (
        tch::Tensor::from_slice2(&xs).to_kind(Kind::Int),
        tch::Tensor::from_slice(&ys).to_kind(Kind::Int64),
    )
}

fn split_loss(data: tch::Tensor, model: &mut dyn Layer, labels: &tch::Tensor) -> f64 {
    let _g = tch::no_grad_guard();
    let logits = model.forward(data);
    f64::try_from(logits.cross_entropy_for_logits(labels)).unwrap()
}

fn create_model(model_type: ModelType, device: tch::Device, vocab_size: i64) -> Box<dyn Layer> {
    Box::new(match model_type {
        ModelType::Wave => create_wavenet(device, vocab_size, EMBEDDING_SIZE, HIDDEN_LAYER_SIZE),
        ModelType::Mlp => create_mlp(
            device,
            vocab_size,
            EMBEDDING_SIZE,
            HIDDEN_LAYER_SIZE,
            CHUNK_SIZE,
        ),
    })
}
