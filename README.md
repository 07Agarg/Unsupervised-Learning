# Unsupervised-Learning

This repository contains upsupervised learning model GANS for MNIST DATASET.

### DATASET:
This network is tested on simplest MNIST DATASET, a large database of handwritten digits that is commonly used for training various image processing systems. It contains 60,000 training images and 10,000 testing images.

The dataset is available in both Tensorflow and PyTorch. Tensorflow loads the dataset in Numpy arrays whereas PyTorch APIs loads the same dataset in Torch Tensors.

### NETWORK DETAILS:
#### Generator Network:
- Fully-connected Layer1: [100(Random noise z), 256] <br />
- Leaky ReLu
- Fully-connected Layer2: [256, 512] 
- Leaky ReLu 
- Fully-connected Layer3: [512, 1024]
- Leaky ReLu
- Fully-connected Layer4: [1024, 784]
- Tanh

#### Discriminator Network: 
- Fully-connected Layer1: [784, 1024] <br />
- Leaky ReLu
- Dropout (keep_probs: 0.3)
- Fully-connected Layer2: [1024, 512] 
- Leaky ReLu 
- Dropout (keep_probs: 0.3)
- Fully-connected Layer3: [512, 256]
- Leaky ReLu
- Dropout (keep_probs: 0.3)
- Fully-connected Layer4: [256, 1]
- Sigmoid

### HYPERPARAMETER DETAILS:
- Batch Size: 100
- Learning Rate: 0.0002
- Number of Epochs: 100
- Adam Optimizer with beta1: 0.5, beta2: default(0.999)
- Dropout: 0.3
- Dataset Normalization: (input_image - 0.5)/0.5
- Weight Initialization: Random normal initialization using mean as 0 and standard deviation(stddev) as 0.02.
- Bias Initialization: Constant value of 0.

### DEVELOPMENT ENVIRONMENT:
The repository contains both PYTORCH and TENSORFLOW models . <br />
- Windows 10
- GeForceRTX 2060
- Install Anaconda  <br />
- Create new environment <br />
- Install python=3.6 <br />
```
conda create -n env_name python=3.6 numpy=1.13.3 scipy
```
For Tensorflow model, install
- Tensorflow 1.9.0
```
conda install tensorflow-gpu
```
For PyTorch model, install
- PyTorch 1.3.1 <br />
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

### RESULT:
#### Tensorflow:

![Tensorflow Loss Curve](https://github.com/07Agarg/Unsupervised-Learning/blob/master/Generative%20Adversarial%20Networks/Tensorflow/RESULT/Best%20Results%20Using%20LeakyRelu%2C%20beta1(0.5)/LossCurve.jpg)
![Tensorflow Output Image](https://github.com/07Agarg/Unsupervised-Learning/blob/master/Generative%20Adversarial%20Networks/Tensorflow/RESULT/Best%20Results%20Using%20LeakyRelu%2C%20beta1(0.5)/Generated_Images_GANS_99.jpg)


#### PyTorch:

![PyTorch Loss Curve](https://github.com/07Agarg/Unsupervised-Learning/blob/master/Generative%20Adversarial%20Networks/PyTorch/RESULT/LossCurve.jpg)
![PyTorch Output Image](https://github.com/07Agarg/Unsupervised-Learning/blob/master/Generative%20Adversarial%20Networks/PyTorch/RESULT/Generated_Images_GANS_99.jpg)
