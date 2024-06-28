# makesmores
Rust implementation of Andrej Karpathy's [makemore](https://youtu.be/PaCmpygFfXo?si=7Zz_Qq9WKmcVCC_c) YouTube series

`makesmores` takes a database of strings (in this case the 32k most popular baby names from [ssa.gov](https://www.ssa.gov/oact/babynames/) for the year 2018) and generates more of them.

Currently allows for the choice between two different model architectures:
* MLP, following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
* CNN, following [DeepMind WaveNet 2016](https://arxiv.org/abs/1609.03499)

This is intended as a low-level exploration of neural networks, so although it depends on `torch`, it only utilizies low level `Tensor` operations. Layers of the network are implemented by hand. Hyper parameters can be adjusted via the constants at the top of the file.

Although this code represents the product at the end of the video series, for those following along, the partial implementations can be found in the commit history.





