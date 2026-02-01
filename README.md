# mymltk
## Overview
This is my personal machine learning (ML) toolkit that contains implementations of various ML techniques using PyTorch (or sometimes coded from scratch with NumPy as an exercise). All of the implementations have an interface modeled after Scikit-Learn's estimators. Each model have a `fit`, `predict`, and `score` method. Here are the methods that I've implemented so far:
- K-nearest neighbors clustering algorithm
- Linear regression regressor (via SGD)
- Logistic regression classifier
- Rosenblatt perceptron classifier
- Fully connected neural network classifier
- RNN for sentiment classification using sequences of words
- CNN (a la AlexNet) for image classification
- ViT (visual transformer) for image classification (demo'ed with the CIFAR-10 dataset)

## Simple (binary) classification
Here's are some examples of cross-validated classifier performance on a binary classification task using the 
[breast cancer Wisconsin dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html):

![](/img/classifier_performance.png)

Each confusion matrix shows the fraction of true labels from a test dataset that were correcty or incorrectly labeled by each classifier. Estimates of test performance were bootstrapped using stratified random resampling of the training and test splits (n=100 permutations, training fraction=0.8, test fraction=0.2). The square brackets show the 95% confidence interval for each estimate.
