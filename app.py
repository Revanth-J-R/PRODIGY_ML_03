import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

# --- Scaling the dataset

# data = []

# for img in os.listdir("train"):
#     if "dog" in img:
#         label = 1
#     elif "cat" in img:
#         label = 0
#     else:
#         print("none")
#     img_path = os.path.join("train", img)
#     pet_img = cv2.imread(img_path, 0)
#     try:
#         pet_img = cv2.resize(pet_img, (50, 50))
#         image = np.array(pet_img).flatten()
#         data.append([image, label])
#     except Exception as e:
#         pass

# --- Storing the dataset into a pickle file

# pick_in = open("data.pickle", "wb")
# pickle.dump(data, pick_in)
# pick_in.close()

# --- loading the data into a list using pickle

pick_in = open("data.pickle", "rb")
data = pickle.load(pick_in)
pick_in.close()

# --- Shuffling the data

random.shuffle(data)

features = []
labels = []

# --- Separating the Source(scaled data of images) and Target Variables(label)

for feature, label in data:
    features.append(feature)
    labels.append(label)

# --- Splitting the train and test data

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# --- Creating the model

# model = SVC(C=1, kernel="poly", gamma="auto")

# --- Using TQDM to monitor the progress of processing

# for i in tqdm(range(1)):
#     model.fit(x_train, y_train)

# model.fit(x_train, y_train)

# --- Storing the Model into a sav file using pickle

# pick = open("svm_model.sav", "wb")
# pickle.dump(model, pick)
# pick.close()

# --- Loading the model from the sav file using Pickle

pick1 = open("svm_model.sav", "rb")
model = pickle.load(pick1)
pick1.close()

# --- Predicting for the Test set using the loaded model

prediction = model.predict(x_test)

# --- Checking the accuracy of the model

accuracy = model.score(x_test, y_test)

print("Accuracy: ", accuracy)

# --- Testing for the first image in the test set

categories = ["Cat", "Dog"]

print("Prediction is: ", categories[prediction[0]])


mypet = x_test[0].reshape(50, 50)
plt.imshow(mypet, cmap="gray")

plt.show()
