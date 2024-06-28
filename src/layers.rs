use std::vec;

use tch::Kind;

pub(crate) trait Layer {
    fn forward(&mut self, x: tch::Tensor) -> tch::Tensor;
    fn zero_grad(&mut self) {}
    fn backward(&mut self, _learning_rate: f64) {}
    fn set_training(&mut self, _training: bool) {}
}

pub(crate) struct Linear {
    pub weight: tch::Tensor,
    pub bias: Option<tch::Tensor>,
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

pub(crate) struct BatchNorm1D {
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
                d => panic!("Unsupported vector size {}", d),
            };
            let xmean = x.mean_dim(&dim, true, Some(Kind::Float));
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

pub(crate) struct Sequential {
    pub layers: Vec<Box<dyn Layer>>,
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

pub(crate) struct Tanh {}

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

pub(crate) struct Embedding {
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

pub(crate) struct FlattenConsecutive {
    pub n: i64,
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

pub(crate) fn create_wavenet(
    device: tch::Device,
    vocab_size: i64,
    embedding_size: i64,
    hidden_layer_size: i64,
) -> Sequential {
    let mut layers: Vec<Box<dyn Layer>> = vec![
        Box::new(Embedding::new(vocab_size, embedding_size, device)),
        Box::new(FlattenConsecutive { n: 2 }),
        Box::new(Linear::new(
            embedding_size * 2,
            hidden_layer_size,
            false,
            device,
        )),
        Box::new(BatchNorm1D::new(hidden_layer_size, 1e-5, 0.1, device)),
        Box::new(Tanh::new()),
        Box::new(FlattenConsecutive { n: 2 }),
        Box::new(Linear::new(
            hidden_layer_size * 2,
            hidden_layer_size,
            false,
            device,
        )),
        Box::new(BatchNorm1D::new(hidden_layer_size, 1e-5, 0.1, device)),
        Box::new(Tanh::new()),
        Box::new(FlattenConsecutive { n: 2 }),
        Box::new(Linear::new(
            hidden_layer_size * 2,
            hidden_layer_size,
            false,
            device,
        )),
        Box::new(BatchNorm1D::new(hidden_layer_size, 1e-5, 0.1, device)),
        Box::new(Tanh::new()),
    ];
    let mut last_linear = Linear::new(hidden_layer_size, vocab_size, false, device);
    tch::no_grad(|| last_linear.weight *= 0.1);
    layers.push(Box::new(last_linear));
    Sequential { layers }
}

pub(crate) fn create_mlp(
    device: tch::Device,
    vocab_size: i64,
    embedding_size: i64,
    hidden_layer_size: i64,
    chunk_size: i64,
) -> Sequential {
    let mut layers: Vec<Box<dyn Layer>> = vec![
        Box::new(Embedding::new(vocab_size, embedding_size, device)),
        Box::new(FlattenConsecutive { n: chunk_size }),
        Box::new(Linear::new(
            embedding_size * chunk_size,
            hidden_layer_size,
            false,
            device,
        )),
        Box::new(BatchNorm1D::new(hidden_layer_size, 1e-5, 0.1, device)),
        Box::new(Tanh::new()),
    ];
    let mut last_linear = Linear::new(hidden_layer_size, vocab_size, false, device);
    tch::no_grad(|| last_linear.weight *= 0.1);
    layers.push(Box::new(last_linear));
    Sequential { layers }
}
