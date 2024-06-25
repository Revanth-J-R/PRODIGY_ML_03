# Image Classification Using Support Vector Machine (SVM)
## Overview
    This project focuses on classifying images of cats and dogs using a Support Vector Machine (SVM) model. The dataset comprises 5000 cat images and 5000 dog images. We preprocess the images, train an SVM model, save the trained model for future use, and evaluate its performance.

## Packages Used
    os
    numpy
    pandas
    cv2
    matplotlib.pyplot
    pickle
    random
    sklearn.model_selection.train_test_split
    sklearn.svm.SVC
    tqdm
## Steps
### 1. Preprocess Images
    Scaling Images: 
        Convert images to a 2D format using OpenCV (cv2).
    Storing Preprocessed Data: 
        Save the preprocessed data in a pickle file for efficient storage and retrieval.
### 2. Load and Prepare Data
    Load the data from the pickle file.
    Shuffle the dataset to ensure randomness.
    Split the dataset into training and testing sets using train_test_split.
### 3. Train the SVM Model
    Fit the SVM model to the training data.
    Save the trained model to a .sav file using pickle to avoid retraining.
### 4. Evaluate the Model
    Load the trained model from the .sav file.
    Calculate the accuracy of the model on the test set.
    Predict and display the classification of the first image from the test set, which varies with each run due to dataset shuffling.

# Note:
## Only part of the dataset is given in this repository.
## Get the whole dataset here: https://www.kaggle.com/c/dogs-vs-cats/data
