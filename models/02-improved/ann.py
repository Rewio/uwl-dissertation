# import libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score

# import dataset
dataset = pd.read_csv("dataset.csv")
X = dataset.iloc[:, 3:20].values     # independent variables.
y = dataset.iloc[:, 1].values        # dependent variable.

lb = LabelBinarizer()
y_binarized = lb.fit_transform(y)

# create the test and training sets from our original dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y_binarized, test_size = 0.1, random_state = 80)

# ensure that all values within the dataset are on the same scale.
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# build the ANN
classifier = Sequential()
 
# input layer and first hidden layer.
classifier.add(Dense(units = 144, input_dim = 17, activation = "relu", kernel_initializer = "uniform"))
classifier.add(Dropout(rate = 0.9))

# second hidden layer.
classifier.add(Dense(units = 144, activation = "relu", kernel_initializer = "uniform"))
classifier.add(Dropout(rate = 0.9))

# output layer.
classifier.add(Dense(units = 7, activation = "sigmoid", kernel_initializer = "uniform"))

classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
classifier.fit(X_train, y_train, batch_size = 50, epochs = 200)

# predict the test data using our trained ANN.
y_pred = classifier.predict(X_test)

# grab the highest value for the test and predicted values to be check accuracy.
y_test_values = np.argmax(y_test, 1)
y_pred_values = np.argmax(y_pred, 1)
accuracy = accuracy_score(y_test_values, y_pred_values)

model_location = "saved_model.h5"
classifier.model.save(model_location)
classifier = load_model(model_location)