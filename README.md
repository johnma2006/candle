Deep learning library, implemented from scratch in pure numpy for educational purposes.

#### Features:
* Tensor-based automatic differentiation
* Object-oriented PyTorch-like API
* Layers: linear, multi-head attention, batch/layer/RMS norm, dropout, convolutional, max/avg pooling
* Optimizers: SGD, AdamW
* LR schedulers: step decay, cosine annealing, warmup
* Image data augmentation: random crop, random horizontal/vertical flips, normalize
* Lightweight Tensorboard-like dashboarding
* Models: ResNet, GPT, MLP


## Experiments

#### Vision
* [Training a ResNet14 on MNIST (notebook)](https://github.com/johnma2006/candle/blob/main/experiments/vision_experiments/2.0%20ResNet14%20on%20MNIST.ipynb)
* [Training an MLP on MNIST (notebook)](https://github.com/johnma2006/candle/blob/main/experiments/vision_experiments/1.0%20MLP%20on%20MNIST%20-%20AdamW.ipynb)


#### Initialization
* Gradient Norm vs. Model {Depth, Norm} under {Xavier, Kaiming} init
  * [Width, Kaiming (notebook)](https://github.com/johnma2006/candle/blob/main/experiments/initialization_experiments/2.0%20Effect%20of%20Model%20Width%20on%20Gradient%20Norm%20-%20MLP%20with%20Kaiming%20Init.ipynb)
  * [Width, Xavier (notebook)](https://github.com/johnma2006/candle/blob/main/experiments/initialization_experiments/2.0%20Effect%20of%20Model%20Width%20on%20Gradient%20Norm%20-%20MLP%20with%20Kaiming%20Init.ipynb)
  * [Depth, Kaiming (notebook)](https://github.com/johnma2006/candle/blob/main/experiments/initialization_experiments/2.0%20Effect%20of%20Model%20Depth%20on%20Gradient%20Norm%20-%20MLP%20with%20Xavier%20Init.ipynb)
  * [Depth, Xavier (notebook)](https://github.com/johnma2006/candle/blob/main/experiments/initialization_experiments/2.0%20Effect%20of%20Model%20Depth%20on%20Gradient%20Norm%20-%20MLP%20with%20Xavier%20Init.ipynb)
* [Activation Distributions vs Init (notebook)](https://github.com/johnma2006/candle/blob/main/experiments/initialization_experiments/1.0%20Activation%20Distribution%20by%20Layer%20w.r.t%20Initialization.ipynb)

## Run Tests

`python -m unittest`
