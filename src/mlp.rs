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
const BATCH_SIZE: i64 = 32;

trait Layer {
    fn forward(&mut self, x: tch::Tensor) -> tch::Tensor;
    fn zero_grad(&mut self);
    fn backward(&mut self, learning_rate: f64);
}

struct Linear {
    weight: tch::Tensor,
    bias: Option<tch::Tensor>,
}

impl Linear {
    pub fn new(fan_in: i64, fan_out: i64, bias: bool, device: tch::Device) -> Self {
        let weight =
            tch::Tensor::randn([fan_in, fan_out], (Kind::Float, device)).set_requires_grad(true);
        let bias_layer = if bias {
            let b = tch::Tensor::randn([fan_out], (Kind::Float, device)).set_requires_grad(true);
            Some(b)
        } else {
            None
        };
        Linear {
            weight,
            bias: bias_layer,
        }
    }
}

impl Layer for Linear {
    fn forward(&mut self, x: tch::Tensor) -> tch::Tensor {
        let mut out = x.matmul(&self.weight);
        if let Some(b) = &self.bias {
            out += b;
        }
        out
    }

    fn zero_grad(&mut self) {
        self.weight.zero_grad();
        if let Some(b) = self.bias.as_mut() {
            b.zero_grad()
        }
    }

    fn backward(&mut self, learning_rate: f64) {
        self.weight += self.weight.grad() * (-learning_rate);
        let bias = self.bias.take();
        if let Some(mut b) = bias {
            b += b.grad() * (-learning_rate);
            self.bias = Some(b);
        }
    }
}

struct BatchNorm1D {
    eps: f64,
    momentum: f32,
    gamma: tch::Tensor,
    beta: tch::Tensor,
    running_mean: tch::Tensor,
    running_var: tch::Tensor,
    training: bool,
}

impl BatchNorm1D {
    pub fn new(dim: i64, eps: f64, momentum: f32, device: tch::Device) -> Self {
        Self {
            eps,
            momentum,
            gamma: tch::Tensor::ones([dim], (Kind::Float, device)).requires_grad_(true),
            beta: tch::Tensor::zeros([dim], (Kind::Float, device)).requires_grad_(true),
            running_mean: tch::Tensor::zeros([dim], (Kind::Float, device)),
            running_var: tch::Tensor::ones([dim], (Kind::Float, device)),
            training: true,
        }
    }
}

impl Layer for BatchNorm1D {
    fn forward(&mut self, x: tch::Tensor) -> tch::Tensor {
        let (xmean, xvar) = if self.training {
            let xmean = x.mean(Some(Kind::Float));
            let xvar = x.var(true);
            (xmean, xvar)
        } else {
            (
                self.running_mean.shallow_clone(),
                self.running_var.shallow_clone(),
            )
        };
        let xhat = (x - xmean.shallow_clone()) / (&xvar + self.eps).sqrt();
        if self.training {
            let _g = tch::no_grad_guard();
            self.running_mean =
                (1.0 - self.momentum) * self.running_mean.shallow_clone() + self.momentum * xmean;
            self.running_var =
                (1.0 - self.momentum) * self.running_var.shallow_clone() + self.momentum * xvar;
        }
        (&self.gamma * xhat) + &self.beta
    }

    fn zero_grad(&mut self) {
        self.gamma.zero_grad();
        self.beta.zero_grad();
    }

    fn backward(&mut self, learning_rate: f64) {
        self.gamma += self.gamma.grad() * (-learning_rate);
        self.beta += self.beta.grad() * (-learning_rate);
    }
}

struct Tanh {}

impl Tanh {
    pub fn new() -> Self {
        Tanh {}
    }
}

impl Layer for Tanh {
    fn forward(&mut self, x: tch::Tensor) -> tch::Tensor {
        x.tanh()
    }

    fn zero_grad(&mut self) {}
    fn backward(&mut self, _learning_rate: f64) {}
}

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
    let (_xdev, _ydev) = load_data(&names[n1..n2], &c_to_i, CHUNK_SIZE as usize);
    let (_xte, _yte) = load_data(&names[n2..], &c_to_i, CHUNK_SIZE as usize);

    let mut c =
        tch::Tensor::randn([27, EMBEDDING_SIZE], (Kind::Float, device)).requires_grad_(true);

    let mut layers: Vec<Box<dyn Layer>> = [
        (EMBEDDING_SIZE * CHUNK_SIZE, HIDDEN_LAYER_SIZE),
        (HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
        (HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
        (HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
        (HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
    ]
    .into_iter()
    .flat_map(|(f_in, f_out)| {
        let mut linear = Linear::new(f_in, f_out, true, device);
        let _g = tch::no_grad_guard();
        linear.weight *= (5.0 / 3.0) / (EMBEDDING_SIZE as f32 * CHUNK_SIZE as f32).sqrt();
        [
            Box::new(linear) as Box<dyn Layer>,
            Box::new(BatchNorm1D::new(HIDDEN_LAYER_SIZE, 1e-5, 0.1, device)),
            Box::new(Tanh::new()) as _,
        ]
    })
    .collect();

    let mut last_linear = Linear::new(HIDDEN_LAYER_SIZE, 27, true, device);
    {
        let _g = tch::no_grad_guard();
        last_linear.weight *= 0.1;
    }
    layers.push(Box::new(last_linear));
    for i in 0..200_000 {
        let ix = tch::Tensor::randint(xtr.size()[0], [BATCH_SIZE], (Kind::Int, device));
        let samples = xtr.i(&ix);
        let labels = ytr.i(&ix);
        let emb = c.index(&[Some(samples)]);
        let mut x = emb.view_([-1, CHUNK_SIZE * EMBEDDING_SIZE]);

        for layer in &mut layers {
            x = layer.forward(x);
        }
        let loss = x.cross_entropy_for_logits(&labels);
        c.zero_grad();
        for l in &mut layers {
            l.zero_grad()
        }
        loss.backward();
        let lr = if i < 100_000 { 0.1 } else { 0.01 };

        tch::no_grad(|| {
            c += c.grad() * (-lr);
            for l in &mut layers {
                l.backward(lr)
            }
        });

        if i % 10_000 == 0 {
            loss.print()
        }
    }

    tch::manual_seed(SEED + 10);

    // Generate a name by repeatedly sampling from the probability distribution of bigrams
    // until an end token is found.
    for _ in 0..20 {
        let mut out = Vec::new();
        let mut ctx = tch::Tensor::zeros([CHUNK_SIZE], (Kind::Int, device));
        loop {
            let emb = c.index(&[Some(&ctx)]);
            let mut x = emb.view_([1, -1]);
            for l in &mut layers {
                x = l.forward(x);
            }
            let prob = x.softmax(1, Kind::Float);
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
        tch::Tensor::from_slice(&ys).to_kind(tch::Kind::Int64),
    )
}
