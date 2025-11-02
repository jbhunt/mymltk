# mymltk
This is my personal machine learning (ML) toolkit that contains implementations of various ML techniques using PyTorch (or sometimes coded from scratch with NumPy just as an exercise). All of the implementations have an interface modeled after Scikit-Learn's estimators. Each model have a `fit`, `predict`, and `score` method. 

Here are the methods that I've implemented:
- K-nearest neighbors clustering algorithm
- Linear regression regressor (via GD/SGD)
- Logistic regression classifier
- Rosenblatt perceptron classifier
- Fully connected neural network classifier

Here's a summary of classifier performance using the Scikit-Learn [breast cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html):
![](/img/classifier_performance.png)
Each confusion matrix shows the fraction of true labels that are correcty or incorrectly labeled by each classifier. The square brackets show the 95% confidence interval for each estimate.
