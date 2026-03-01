# 🧠 Machine Learning & Neural Networks Foundations

A comprehensive collection of fundamental machine learning algorithms, statistical estimation methods, and deep learning architectures. 

Developed for **Assignment 2** of the **"Statistical Modeling and Pattern Recognition" (THL311)** course at the Technical University of Crete (ECE Department).

## 🚀 Project Overview

This repository focuses on transitioning from foundational statistical learning algorithms built from scratch to modern deep learning implementations using popular frameworks. It covers linear classifiers, probabilistic estimation, clustering for image compression, and Convolutional Neural Networks (CNNs).

## 🧮 Core Topics & Implementations

### 1. The Perceptron Algorithm
* **Custom Implementation:** Implemented the batch Perceptron algorithm from scratch in Python.
* **Tasks:** Classified synthetic 2D data across 4 distinct classes ($\omega_1, \omega_2, \omega_3, \omega_4$). Tracked convergence rates (number of epochs) and visualized decision boundaries for various class pairs.

### 2. Logistic Regression
* **Mathematical Derivation:** Analytically derived the gradient for the cross-entropy loss function.
* **Custom Implementation:** Built the `sigmoid` function, `costFunction`, and `gradient` calculations using `numpy` and `scipy.optimize`.
* **Application:** Developed a predictive model to determine university admission probabilities based on past exam scores.

### 3. Maximum Likelihood Estimation (MLE)
* **Gaussian Parameter Estimation:** Calculated the Maximum Likelihood estimates for the mean ($\mu$) and variance/covariance ($\Sigma$).
* **Dimensionality Analysis:** Applied MLE across 1D, 2D, and 3D feature spaces, including scenarios assuming a diagonal covariance matrix, and compared the statistical variations of the estimated parameters.

### 4. Image Compression via K-Means Clustering
* **Custom Algorithm:** Implemented the core components of K-Means (`findClosestCentroids` and `computeCentroids`) without high-level ML clustering libraries.
* **Application:** Applied the algorithm to compress an image (`Fruit.png`)
