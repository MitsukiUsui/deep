# What's this?
Various deep learning algorithms implemented in TensorFlow.

## Algorithms
### MNIST
This is a tutorial for image classification using MNIST.
4 different implementation are provided in [MNIST/mymodels](https://github.com/MitsukiUsui/deep/tree/master/mnist/mymodels), together with simple k-NN neighbor.

0. [mlp_notf.py](https://github.com/MitsukiUsui/deep/tree/master/mnist/mymodels/mlp_notf.py)
    * Multiple Layer Perceptron (MLP) without TensorFlow
0. [mlp.py](https://github.com/MitsukiUsui/deep/tree/master/mnist/mymodels/mlp.py)
    * MLP with Tensorflow
0. [???]()
    * Stacked Denoising Autoencoder + MLP
0. [cnn.py](https://github.com/MitsukiUsui/deep/tree/master/mnist/mymodels/cnn.py)
    * Convolutional neural network (CNN)

The accuracy of those 4 (+1) programs can be evaluated on a same interface. Please refer to <https://github.com/MitsukiUsui/deep/blob/master/mnist/score_models.ipynb>.

### Image Classification
Classification of CIFAR-10 by CNN and various preprocessing.

* Data Augmentation
* Global Contrast Normalization (GCN)
* ZCA Whitening

### Sentiment Analysis
Sentiment analysis of IMDb (movie review dataset) using Recurrent Neural Network (RNN).

### Machine Translation
English-Japanese machine translation using Long Short-Term Memory (LSTM), trained by [Tanaka Corpus](https://github.com/odashi/small_parallel_enja).

### Image Caption Generation
Image caption generation of MS COCO (Microsoft Common Objects in Context) using CNN + LSTM + Attention model.

---

## Environment
```
(installation of pyenv)
(installation of anaconda3-??)
pip install upgrade tensorflow
```
