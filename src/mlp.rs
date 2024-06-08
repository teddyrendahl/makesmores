use std::collections::{BTreeMap, HashMap};

use anyhow::Result;
use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use tch::{IndexOp, Kind};

const EDGE_TOKEN: char = '.';
const SEED: i64 = 2147483647;
const EMBEDDING_SIZE: i64 = 5;
const CHUNK_SIZE: i64 = 3;
const HIDDEN_LAYER_SIZE: i64 = 100;

fn main() -> Result<()> {
    tch::manual_seed(SEED);
    let device = tch::Device::cuda_if_available();
    let mut names = get_names()?;
    // Generate a mapping of token to integer (and the reverse)
    let (c_to_i, i_to_c) = create_char_maps(&names);

    // Split dataset into train, dev, test
    let mut rng = ChaCha12Rng::seed_from_u64(42);
    names.shuffle(&mut rng);
    let n1 = (names.len() as f32 * 0.8) as usize;
    let n2 = (names.len() as f32 * 0.9) as usize;

    let (xtr, ytr) = load_data(&names[0..n1], &c_to_i, CHUNK_SIZE as usize);
    let (xdev, ydev) = load_data(&names[n1..n2], &c_to_i, CHUNK_SIZE as usize);
    let (xte, yte) = load_data(&names[n2..], &c_to_i, CHUNK_SIZE as usize);

    let mut c =
        tch::Tensor::randn([27, EMBEDDING_SIZE], (Kind::Float, device)).requires_grad_(true);
    let mut w1 = tch::Tensor::randn(
        [CHUNK_SIZE * EMBEDDING_SIZE, HIDDEN_LAYER_SIZE],
        (Kind::Float, device),
    )
    .requires_grad_(true);
    let mut b1 =
        tch::Tensor::randn([HIDDEN_LAYER_SIZE], (Kind::Float, device)).requires_grad_(true);
    let mut w2 =
        tch::Tensor::randn([HIDDEN_LAYER_SIZE, 27], (Kind::Float, device)).requires_grad_(true);
    let mut b2 = tch::Tensor::randn([27], (Kind::Float, device)).requires_grad_(true);

    for i in 0..200000 {
        let ix = tch::Tensor::randint(xtr.size()[0], [32], (Kind::Int, device));
        let samples = xtr.i(&ix);
        let labels = ytr.i(&ix);
        let emb = c.index(&[Some(samples)]);
        let h = (emb.view_([-1, CHUNK_SIZE * EMBEDDING_SIZE]).matmul(&w1) + &b1).tanh();
        let logits = h.matmul(&w2) + &b2;
        let loss = logits.cross_entropy_for_logits(&labels);

        for p in [&mut w1, &mut b1, &mut c, &mut w2, &mut b2] {
            p.zero_grad();
        }
        loss.backward();

        let lr = if i < 100000 { -0.1 } else { -0.01 };
        tch::no_grad(|| {
            c += c.grad() * lr;
            w1 += w1.grad() * lr;
            b1 += b1.grad() * lr;
            w2 += w2.grad() * lr;
            b2 += b2.grad() * lr;
        });
    }
    // Final test loss
    println!("Training loss");
    let emb = c.index(&[Some(xtr)]);
    let h = (emb.view_([-1, CHUNK_SIZE * EMBEDDING_SIZE]).matmul(&w1) + &b1).tanh();
    let logits = h.matmul(&w2) + &b2;
    logits.cross_entropy_for_logits(&ytr).print();

    println!("Dev loss");
    let emb = c.index(&[Some(xdev)]);
    let h = (emb.view_([-1, CHUNK_SIZE * EMBEDDING_SIZE]).matmul(&w1) + &b1).tanh();
    let logits = h.matmul(&w2) + &b2;
    logits.cross_entropy_for_logits(&ydev).print();

    println!("Test loss");
    let emb = c.index(&[Some(xte)]);
    let h = (emb.view_([-1, CHUNK_SIZE * EMBEDDING_SIZE]).matmul(&w1) + &b1).tanh();
    let logits = h.matmul(&w2) + &b2;
    logits.cross_entropy_for_logits(&yte).print();

    for _ in 0..20 {
        let mut out = Vec::new();
        let mut ctx = tch::Tensor::zeros([CHUNK_SIZE], (Kind::Int, device));
        loop {
            let emb = c.index(&[Some(&ctx)]);
            let h = (emb.view_([1, -1]).matmul(&w1) + &b1).tanh();
            let logits = h.matmul(&w2) + &b2;
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
    // Generate a name by repeatedly sampling from the probability distribution of bigrams
    // until an end token is found.
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
        tch::Tensor::from_slice(&ys).to_kind(tch::Kind::Int64),
    )
}
