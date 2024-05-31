# Handwritten-Digit-Classification-and-Clustering
This repository contains the implementation of various machine learning techniques for the classification and clustering of handwritten digit images. The dataset provided is structured similarly to the one used in practical sessions.
The theoretical basis can be based in the pdf report file in Vietnamese.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Table of Contents
- Overview
- Dataset
- Dimensionality Reduction
- Clustering
- Classification Models
- Multinomial Logistic Regression
- Convolutional Neural Network (CNN)
- Support Vector Machine (SVM)
- Results
- Contributing
- Acknowledgements

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
![Alt text](https://github.com/LanaDuDay/Handwritten-Digit-Classification-and-Clustering/blob/main/Images/Samples%20of%20MNIST%20data.png)


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
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Result:
```
accuracy training data:  0.8374166666666667
(784, 12000)
accuracy testing data:  0.8285
Confusion Matrix:
 [[1127    0   16    2    3    5   13    8    0    1]
 [   1 1228   56    7    3    6    1   10   10    0]
 [   3    4 1098    6   12    7   16   21    5    2]
 [   4    5  125 1010    2   40    2   14    7   10]
 [   3    2    7    1 1115    2   23   15    1    7]
 [  20    6   31   47   12  944   18    6   18    2]
 [   8    0   38    1    3   15 1107    4    1    0]
 [   5    1   28    3   15    2    0 1234    1   10]
 [  14   19  290   54   14  212   24   17  502   14]
 [  10    1   14   14  262   40    1  271    4  577]]
Precision: 0.8485720395027455
Recall: 0.8261554782756193
```
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Convolutional Neural Network (CNN)
A CNN with the following architecture is implemented for classification:

- At least 3 convolutional layers (Convolution + Activation ReLU + Max Pooling)
- 2 fully connected layers
- A final softmax layer for classification
- The model is trained on the training dataset and evaluated on the validation dataset.
```
Result:
Epoch 1/10
1500/1500 [==============================] - 31s 20ms/step - loss: 0.7425 - accuracy: 0.8130 - val_loss: 0.1382 - val_accuracy: 0.9588
Epoch 2/10
1500/1500 [==============================] - 31s 21ms/step - loss: 0.1913 - accuracy: 0.9410 - val_loss: 0.0955 - val_accuracy: 0.9714
Epoch 3/10
1500/1500 [==============================] - 29s 19ms/step - loss: 0.1456 - accuracy: 0.9550 - val_loss: 0.0882 - val_accuracy: 0.9718
Epoch 4/10
1500/1500 [==============================] - 29s 19ms/step - loss: 0.1226 - accuracy: 0.9626 - val_loss: 0.0810 - val_accuracy: 0.9763
Epoch 5/10
1500/1500 [==============================] - 29s 19ms/step - loss: 0.1114 - accuracy: 0.9669 - val_loss: 0.0699 - val_accuracy: 0.9801
Epoch 6/10
1500/1500 [==============================] - 29s 19ms/step - loss: 0.0975 - accuracy: 0.9700 - val_loss: 0.0659 - val_accuracy: 0.9812
Epoch 7/10
1500/1500 [==============================] - 29s 19ms/step - loss: 0.0907 - accuracy: 0.9719 - val_loss: 0.0686 - val_accuracy: 0.9795
Epoch 8/10
1500/1500 [==============================] - 29s 19ms/step - loss: 0.0848 - accuracy: 0.9734 - val_loss: 0.0641 - val_accuracy: 0.9822
Epoch 9/10
1500/1500 [==============================] - 29s 19ms/step - loss: 0.0780 - accuracy: 0.9761 - val_loss: 0.0539 - val_accuracy: 0.9834
Epoch 10/10
1500/1500 [==============================] - 29s 19ms/step - loss: 0.0777 - accuracy: 0.9765 - val_loss: 0.0650 - val_accuracy: 0.9803
```
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Support Vector Machine (SVM)
A Multi-Class Support Vector Machine model is implemented to classify the images. The model is trained on the training dataset and evaluated on the validation dataset.
Result:
```
Accuracy: 86.12%
Confusion Matrix:
[[1097    1   13    3    5   31   15    2    2    6]
 [   1 1247   10   15    4   21    2   10   10    2]
 [   4   16 1018   26   21   12   21   26   18   12]
 [   7    4   41 1002    6   87    7   21   29   15]
 [   1    1    5    4 1104    6    2    6   14   33]
 [  24   10   16   37   19  899   12   10   55   22]
 [  14    3    9    2   35   32 1074    0    8    0]
 [   7   20   10    7   15    2    0 1189    0   49]
 [  14   76   18   48   14   69   14   17  860   30]
 [   5    2    3   17  208   11    0   93   11  844]]
Precision: 0.8631171961992761
Recall: 0.8611666666666666
```
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Contributing
Contributions are welcome! Please read the contributing guidelines first.
