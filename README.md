# SMAI Assignment 3 README

This README provides an overview of the completed tasks for the Statistical Methods in Artificial Intelligence (SMAI) Assignment 3.

## 1. Multinomial Logistic Regression

### 1.1 Dataset Analysis and Preprocessing [5 marks]
- Described dataset using mean, standard deviation, min, and max values for all attributes
- Graphed distribution of labels across the dataset using Matplotlib
- Partitioned dataset into train, validation, and test sets using sklearn
- Normalized and standardized data, handling missing or inconsistent values

### 1.2 Model Building from Scratch [20 marks]
- Created Multinomial Logistic Regression model from scratch
- Used cross-entropy loss and Gradient Descent optimization
- Trained model, reported metrics on validation set, loss and accuracy on train set

### 1.3 Hyperparameter Tuning and Evaluation [15 marks]
- Used validation set and W&B logging to fine-tune hyperparameters (learning rate, epochs)
- Evaluated model on test dataset and printed sklearn classification report

## 2. Multi Layer Perceptron Classification

### 2.1 Model Building from Scratch [20 marks]
- Built MLP classifier class with modifiable hyperparameters
- Implemented forward propagation, backpropagation, and training methods
- Implemented Sigmoid, Tanh, and ReLU activation functions
- Implemented SGD, Batch Gradient Descent, and Mini-Batch Gradient Descent optimizers

### 2.2 Model Training & Hyperparameter Tuning using W&B [10 marks]
- Logged scores (loss, accuracy) on validation and train sets using W&B
- Reported metrics (accuracy, f-1 score, precision, recall) for all combinations of activation functions and optimizers
- Tuned model on learning rate, epochs, and hidden layer neurons
- Plotted accuracy trends and reported best model parameters

### 2.3 Evaluating Model [10 marks]
- Tested and printed classification report on test set using sklearn
- Compared results with logistic regression model

### 2.4 Multi-Label Classification [20 marks]
- Modified model for multilabel classification on "advertisement.csv" dataset
- Logged scores, reported metrics for all combinations of activation functions and optimizers
- Tuned hyperparameters, plotted accuracy trends, reported best model parameters
- Evaluated model on test set and reported metrics

## 3. Multilayer Perceptron Regression

### 3.1 Data Preprocessing [5 marks]
- Described Boston Housing dataset using statistical measures
- Graphed distribution of labels using Matplotlib
- Partitioned dataset into train, validation, and test sets
- Normalized and standardized data, handling missing or inconsistent values

### 3.2 MLP Regression Implementation from Scratch [20 marks]
- Created MLP regression class with modifiable hyperparameters
- Implemented forward propagation, backpropagation, and training methods
- Implemented Sigmoid, Tanh, and ReLU activation functions
- Implemented SGD, Batch Gradient Descent, and Mini-Batch Gradient Descent optimizers

### 3.3 Model Training & Hyperparameter Tuning using W&B [20 marks]
- Logged MSE loss on validation set using W&B
- Reported MSE, RMSE, R-squared metrics
- Reported scores for all combinations of activation functions and optimizers
- Tuned model on learning rate, epochs, and hidden layer neurons
- Reported best model parameters and scores for all hyperparameter values

### 3.4 Evaluating Model [5 marks]
- Tested model on test set and reported MSE, RMSE, R-squared

## 4. CNN and AutoEncoders

### 4.1 Data Visualization and Preprocessing [10 marks]
- Graphed label distribution across MNIST dataset using Matplotlib
- Visualized samples from each class
- Checked for class imbalance and reported
- Partitioned dataset into train, validation, and test sets
- Wrote function to visualize feature maps of a trained model

### 4.2 Model Building [20 marks]
- Constructed CNN model for image classification using PyTorch
- Included convolutional, pooling, dropout, and fully connected layers
- Constructed and trained baseline CNN with specified architecture
- Displayed feature maps after convolution and pooling layers and provided analysis
- Reported training and validation loss and accuracy at each epoch

### 4.3 Hyperparameter Tuning and Evaluation [20 marks]
- Used W&B for hyperparameter tuning (learning rate, batch size, kernel sizes, strides, epochs, dropout rates)
- Compared effect of using and not using dropout layers
- Logged metrics (loss, accuracy, confusion matrices, class-specific metrics) using W&B

### 4.4 Model Evaluation and Analysis [10 marks]
- Evaluated best model on test set and reported accuracy, per-class accuracy, classification report
- Visualized model's performance using confusion matrix
- Identified and analyzed instances of incorrect predictions

### 4.5 Train on Noisy Dataset [10 marks]
- Trained best model on noisy MNIST dataset (mnist-with-awgn.mat)
- Reported validation losses, scores, training losses, scores
- Evaluated model on test data and printed classification report

### 4.6 AutoEncoders to Save the Day [30 marks]
- Implemented Autoencoder class to de-noise noisy MNIST dataset
- Visualized classes and feature space before and after de-noising
- Trained best model using de-noised dataset
- Reported validation losses, scores, training losses, scores
- Evaluated model on test data and printed classification report
- Analyzed and compared results/accuracy scores from Parts 4.5 and 4.6

## 5. Some Other Variants

### 5.1 Multi-digit Recognition on Multi-MNIST Dataset [25 marks]
- Built and trained models to recognize two digits from a single image in DoubleMNIST dataset
- Filtered out images with the same digit appearing twice
- Split datasets into training and validation sets

#### 5.1.1 MLP on Multi-MNIST
- Implemented and trained MLP model on MultiMNIST dataset
- Performed hyperparameter tuning on hidden layers and neurons
- Reported accuracies on train and validation sets
- Evaluated trained model on test set and reported accuracy

#### 5.1.2 CNN on Multi-MNIST
- Designed and trained CNN model on MultiMNIST dataset
- Performed hyperparameter tuning on learning rates, kernel sizes, dropout rates
- Reported accuracies on train and validation sets
- Evaluated trained model on test set and reported accuracy

#### 5.1.3 Testing on Single digit MNIST
- Evaluated trained models on regular MNIST dataset with single-digit images
- Reported accuracies

### 5.2 Permuted MNIST [15 marks]
- Split Permuted MNIST dataset into training and validation sets

#### 5.2.1 MLP on Permuted-MNIST
- Implemented and trained MLP model on Permuted-MNIST dataset
- Performed hyperparameter tuning on hidden layers and neurons
- Reported accuracies on train and validation sets
- Evaluated trained model on test set and reported accuracy

#### 5.2.2 CNN on Permuted-MNIST
- Designed and trained CNN model on Permuted-MNIST dataset
- Performed hyperparameter tuning on learning rates, kernel sizes, dropout rates
- Reported accuracies on train and validation sets
- Evaluated trained model on test set and reported accuracy

### 5.3 Analysis [10 marks]
- Contrasted performances of MLP vs. CNN for both datasets
- Discussed differences and challenges faced during training and evaluation
- Compared potential for overfitting between CNN and MLP using loss/accuracy plots

## 6. Report
- Submitted separate files for Tasks 1-3 and Tasks 4-5
- Submitted clear and organized W&B reports for all 5 tasks
