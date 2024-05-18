use std::collections::{BTreeMap, HashMap};

use anyhow::Result;
use itertools::Itertools;
use nalgebra::SMatrix;

const EDGE_TOKEN: char = '.';
const SEED: i64 = 2147483647;

fn main() -> Result<()> {
    let names: Vec<_> = std::fs::read_to_string("names.txt")?
        .lines()
        .map(|s| s.to_string())
        .collect();

    // Generate a mapping of token to integer (and the reverse)
    let mut c_to_i: BTreeMap<_, _> = names
        .join("")
        .chars()
        .unique()
        .sorted()
        .enumerate()
        .map(|(i, c)| (c, i + 1))
        .collect();
    c_to_i.insert(EDGE_TOKEN, 0);
    let i_to_c: HashMap<usize, char> = c_to_i.iter().map(|(c, i)| (*i, *c)).collect();

    // Calculate the number of time each bigram occurs in the dataset
    let mut data = SMatrix::<f32, 27, 27>::zeros();
    for name in &names {
        let edged_name = format!("{}{}{}", EDGE_TOKEN, name, EDGE_TOKEN);
        for (c1, c2) in edged_name.chars().zip(edged_name.chars().skip(1)) {
            data[(c_to_i[&c1], c_to_i[&c2])] += 1.0;
        }
    }

    // Determine the - log likelihood loss over our training data
    let mut loss = 0.0;
    let mut n = 0.0;
    for name in &names {
        let edged_name = format!("{}{}{}", EDGE_TOKEN, name, EDGE_TOKEN);
        for (c1, c2) in edged_name.chars().zip(edged_name.chars().skip(1)) {
            let prob = (data[(c_to_i[&c1], c_to_i[&c2])] + 1.0) / data.row(c_to_i[&c1]).sum();
            loss += prob.ln();
            n += 1.0;
        }
    }
    loss /= -n as f32;
    dbg!(loss);

    // Generate a name by repeatedly sampling from the probability distribution of bigrams
    // until an end token is found.
    tch::manual_seed(SEED);
    let mut idx = 0;
    let mut out = Vec::new();
    loop {
        idx = tch::Tensor::from_slice((data.row(idx) / data.row(idx).sum()).as_slice())
            .multinomial(1, true)
            .int64_value(&[0]) as usize;
        out.push(i_to_c[&idx]);
        if idx == 0 {
            break;
        }
    }
    println!("{}", out.into_iter().join(""));
    // }
    Ok(())
}
