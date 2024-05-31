# Handwritten-Digit-Classification-and-Clustering
This repository contains the implementation of various machine learning techniques for the classification and clustering of handwritten digit images. The dataset provided is structured similarly to the one used in practical sessions.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Table of Contents
Overview
Dataset
Dimensionality Reduction
Clustering
Classification Models
Multinomial Logistic Regression
Convolutional Neural Network (CNN)
Support Vector Machine (SVM)
Model Comparison
Results
Installation
Usage
Contributing
Acknowledgements

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Overview
This project focuses on the following tasks:

Dimensionality reduction and visualization of handwritten digit data.
Clustering the original data using a clustering algorithm and visualizing the clusters.
Classification of handwritten digits using various models:
Multinomial Logistic Regression (Softmax)
Convolutional Neural Network (CNN)
Multi-Class Support Vector Machine (SVM)
Comparing the accuracy, confusion matrix, recall, and precision of the classification models.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Dataset
The dataset consists of images of handwritten digits, organized in directories. Each directory corresponds to a different digit (0-9).

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Dimensionality Reduction
We perform dimensionality reduction on the dataset and visualize the data in 2D space to understand the distribution of the different classes.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Clustering
Using a clustering algorithm (K-Means, Gaussian Mixture Model-EM), we cluster the original data and visualize the clusters, marking each cluster distinctly.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Classification Models
Multinomial Logistic Regression
A Multinomial Logistic Regression model (Softmax) is implemented to classify the images. The model is trained on the training dataset and evaluated on the validation dataset.

Convolutional Neural Network (CNN)
A CNN with the following architecture is implemented for classification:

- At least 3 convolutional layers (Convolution + Activation ReLU + Max Pooling)
- 2 fully connected layers
- A final softmax layer for classification
- The model is trained on the training dataset and evaluated on the validation dataset.

Support Vector Machine (SVM)
A Multi-Class Support Vector Machine model is implemented to classify the images. The model is trained on the training dataset and evaluated on the validation dataset.

Contributing
Contributions are welcome! Please read the contributing guidelines first.
