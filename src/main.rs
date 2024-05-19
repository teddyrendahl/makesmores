use std::collections::{BTreeMap, HashMap};

use anyhow::Result;
use itertools::Itertools;

const EDGE_TOKEN: char = '.';
const SEED: i64 = 2147483647;

fn main() -> Result<()> {
    tch::manual_seed(SEED);
    let device = tch::Device::cuda_if_available();
    let names = get_names()?;

    // Generate a mapping of token to integer (and the reverse)
    let (c_to_i, i_to_c) = create_char_maps(&names);

    let (xenc, yenc) = load_bigram_data(&names, &c_to_i);
    let mut w = tch::Tensor::randn([27, 27], (tch::Kind::Float, device)).requires_grad_(true);

    for _ in 0..200 {
        let logits = xenc.matmul(&w).exp();
        let prob = &logits / logits.sum_dim_intlist([1].as_slice(), true, tch::Kind::Float);
        let regularization = w.pow_tensor_scalar(2).mean(tch::Kind::Float);
        // Determine the - log likelihood loss
        let loss = -prob
            .index(&[
                Some(tch::Tensor::arange(
                    xenc.size()[0],
                    (tch::Kind::Int64, device),
                )),
                Some(yenc.shallow_clone()),
            ])
            .log()
            .mean(tch::Kind::Float)
            + regularization * 0.01;
        dbg!(&loss);
        w.zero_grad();
        loss.backward();
        tch::no_grad(|| w += w.grad() * -50);
    }

    tch::manual_seed(SEED);
    // Generate a name by repeatedly sampling from the probability distribution of bigrams
    // until an end token is found.
    let mut idx = 0;
    let mut out = Vec::new();
    loop {
        let xenc = tch::Tensor::from_slice(&[idx])
            .one_hot(27)
            .to_kind(tch::Kind::Float);
        let counts = xenc.matmul(&w).exp();
        let prob = &counts / counts.sum_dim_intlist([1].as_slice(), true, tch::Kind::Float);
        idx = prob.multinomial(1, true).get(0).int64_value(&[0]);
        out.push(i_to_c[&(idx as usize)]);
        if idx == 0 {
            break;
        }
    }
    println!("{}", out.into_iter().join(""));
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

fn load_bigram_data(
    words: &[String],
    char_map: &BTreeMap<char, usize>,
) -> (tch::Tensor, tch::Tensor) {
    let mut xs = vec![];
    let mut ys = vec![];

    for name in words {
        let edged_name = format!("{}{}{}", EDGE_TOKEN, name, EDGE_TOKEN);
        for (c1, c2) in edged_name.chars().zip(edged_name.chars().skip(1)) {
            xs.push(char_map[&c1] as i8);
            ys.push(char_map[&c2] as i8);
        }
    }
    (
        tch::Tensor::from_slice(&xs).onehot(27),
        tch::Tensor::from_slice(&ys).to_kind(tch::Kind::Int64),
    )
}
