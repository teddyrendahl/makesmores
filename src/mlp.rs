use std::{collections::{BTreeMap, HashMap}, vec};

use anyhow::Result;
use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use tch::{IndexOp, Kind};

const EDGE_TOKEN: char = '.';
const SEED: i64 = 2147483647;
const EMBEDDING_SIZE: i64 = 24;
const CHUNK_SIZE: i64 = 8;
const HIDDEN_LAYER_SIZE: i64 = 128;
const BATCH_SIZE: i64 = 32;
const VOCAB_SIZE: i64 = 27;

trait Layer {
    fn forward(&mut self, x: tch::Tensor) -> tch::Tensor;
    fn zero_grad(&mut self) {}
    fn backward(&mut self, _learning_rate: f64) {}
    fn set_training(&mut self, _training: bool) {}
}

struct Linear {
    weight: tch::Tensor,
    bias: Option<tch::Tensor>,
}

impl Linear {
    pub fn new(fan_in: i64, fan_out: i64, bias: bool, device: tch::Device) -> Self {
        let mut weight =
            tch::Tensor::randn([fan_in, fan_out], (Kind::Float, device)).set_requires_grad(true);
        tch::no_grad(|| weight /= (fan_in as f32).sqrt());
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
            let dim = match x.dim() {
                2 => vec![0],
                3 => vec![0, 1],
                d => panic!("Unsupported vector size {}", d)
            };
            let xmean = x.mean_dim(&dim, true,Some(Kind::Float));
            let xvar = x.var_dim(&dim, true, true);
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

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Layer for Sequential {
    fn forward(&mut self, x: tch::Tensor) -> tch::Tensor {
        self.layers.iter_mut().fold(x, |acc, l| l.forward(acc))
    }

    fn zero_grad(&mut self) {
        for l in self.layers.iter_mut() {
            l.zero_grad()
        }
    }

    fn backward(&mut self, learning_rate: f64) {
        for l in self.layers.iter_mut() {
            l.backward(learning_rate)
        }
    }

    fn set_training(&mut self, training: bool) {
        for l in self.layers.iter_mut() {
            l.set_training(training)
        }
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
}

struct Embedding {
    weight: tch::Tensor,
}

impl Embedding {
    pub fn new(num_embeddings: i64, embedding_dim: i64, device: tch::Device) -> Self {
        Embedding {
            weight: tch::Tensor::randn([num_embeddings, embedding_dim], (Kind::Float, device))
                .requires_grad_(true),
        }
    }
}

impl Layer for Embedding {
    fn forward(&mut self, x: tch::Tensor) -> tch::Tensor {
        self.weight.index(&[Some(x)])
    }

    fn zero_grad(&mut self) {
        self.weight.zero_grad()
    }

    fn backward(&mut self, learning_rate: f64) {
        self.weight += self.weight.grad() * -learning_rate
    }
}

struct FlattenConsecutive {
    n: i64,
}

impl Layer for FlattenConsecutive {
    fn forward(&mut self, x: tch::Tensor) -> tch::Tensor {
        let mut out = x.view([x.size()[0], x.size()[1] / self.n, x.size()[2] * self.n]);
        if out.size()[1] == 1 {
            out = out.squeeze_dim(1)
        }
        out
    }
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
    tch::nn::Linear
    let (xtr, ytr) = load_data(&names[0..n1], &c_to_i, CHUNK_SIZE as usize);
    let (xdev, ydev) = load_data(&names[n1..n2], &c_to_i, CHUNK_SIZE as usize);
    let (_xte, _yte) = load_data(&names[n2..], &c_to_i, CHUNK_SIZE as usize);

    let mut layers: Vec<Box<dyn Layer>> = vec![
        Box::new(Embedding::new(VOCAB_SIZE, EMBEDDING_SIZE, device)),
        Box::new(FlattenConsecutive { n: 2 }),
        Box::new(Linear::new(
            EMBEDDING_SIZE * 2,
            HIDDEN_LAYER_SIZE,
            false,
            device,
        )),
        Box::new(BatchNorm1D::new(HIDDEN_LAYER_SIZE, 1e-5, 0.1, device)),
        Box::new(Tanh::new()),
        Box::new(FlattenConsecutive { n: 2 }),
        Box::new(Linear::new(
            HIDDEN_LAYER_SIZE * 2,
            HIDDEN_LAYER_SIZE,
            false,
            device,
        )),
        Box::new(BatchNorm1D::new(HIDDEN_LAYER_SIZE, 1e-5, 0.1, device)),
        Box::new(Tanh::new()),
        Box::new(FlattenConsecutive { n: 2 }),
        Box::new(Linear::new(
            HIDDEN_LAYER_SIZE * 2,
            HIDDEN_LAYER_SIZE,
            false,
            device,
        )),
        Box::new(BatchNorm1D::new(HIDDEN_LAYER_SIZE, 1e-5, 0.1, device)),
        Box::new(Tanh::new()),
    ];

    let mut last_linear = Linear::new(HIDDEN_LAYER_SIZE, 27, false, device);
    tch::no_grad(|| last_linear.weight *= 0.1);
    layers.push(Box::new(last_linear));

    let mut model = Sequential { layers };
    for i in 0..200_000 {
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
    println!("Training loss -> {}", split_loss(xtr, &mut model, &ytr));
    println!("Validation loss -> {}", split_loss(xdev, &mut model, &ydev));

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
        tch::Tensor::from_slice(&ys).to_kind(tch::Kind::Int64),
    )
}

fn split_loss(data: tch::Tensor, model: &mut dyn Layer, labels: &tch::Tensor) -> f64 {
    let _g = tch::no_grad_guard();
    let logits = model.forward(data);
    f64::try_from(logits.cross_entropy_for_logits(labels)).unwrap()
}
