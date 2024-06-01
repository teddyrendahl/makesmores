use std::collections::{BTreeMap, HashMap};

use anyhow::Result;
use itertools::Itertools;
use tch::{IndexOp, Kind};

const EDGE_TOKEN: char = '.';
const SEED: i64 = 2147483647;

fn main() -> Result<()> {
    tch::manual_seed(SEED);
    let device = tch::Device::cuda_if_available();
    let names = get_names()?;

    // Generate a mapping of token to integer (and the reverse)
    let (c_to_i, _i_to_c) = create_char_maps(&names);

    let (xs, ys) = load_data(&names, &c_to_i, 3);

    let mut c = tch::Tensor::randn([27, 2], (Kind::Float, device)).requires_grad_(true);
    let mut w1 = tch::Tensor::randn([6, 100], (Kind::Float, device)).requires_grad_(true);
    let mut b1 = tch::Tensor::randn([100], (Kind::Float, device)).requires_grad_(true);
    let mut w2 = tch::Tensor::randn([100, 27], (Kind::Float, device)).requires_grad_(true);
    let mut b2 = tch::Tensor::randn([27], (Kind::Float, device)).requires_grad_(true);

    for _ in 0..100000 {
        let ix = tch::Tensor::randint(xs.size()[0], [32], (Kind::Int, device));
        let samples = xs.i(&ix);
        let labels = ys.i(&ix);
        let emb = c.index(&[Some(samples)]);
        let h = (emb.view_([-1, 6]).matmul(&w1) + &b1).tanh();
        let logits = h.matmul(&w2) + &b2;
        let loss = logits.cross_entropy_for_logits(&labels);
        // loss.print();

        for p in [&mut w1, &mut b1, &mut c, &mut w2, &mut b2] {
            p.zero_grad();
        }
        loss.backward();

        let lr = -0.1;
        tch::no_grad(|| {
            c += c.grad() * lr;
            w1 += w1.grad() * lr;
            b1 += b1.grad() * lr;
            w2 += w2.grad() * lr;
            b2 += b2.grad() * lr;
        });
    }
    let emb = c.index(&[Some(xs)]);
    let h = (emb.view_([-1, 6]).matmul(&w1) + &b1).tanh();
    let logits = h.matmul(&w2) + &b2;
    let loss = logits.cross_entropy_for_logits(&ys);
    loss.print();
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
        // for (c1, c2) in edged_name.chars().zip(edged_name.chars().skip(1)) {
        //     xs.push(char_map[&c1] as i8);
        //     ys.push(char_map[&c2] as i8);
        // }
    }
    (
        tch::Tensor::from_slice2(&xs).to_kind(Kind::Int),
        tch::Tensor::from_slice(&ys).to_kind(tch::Kind::Int64),
    )
}
