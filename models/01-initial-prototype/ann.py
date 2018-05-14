# import libraries
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from keras.models import load_model

# import dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, 3:13].values     # independent variables.
y = dataset.iloc[:, 13].values       # dependent variable.

# Encoding categorical data

# encode the country column categorical data.
labelencoder_X_Country = LabelEncoder()
X[:, 1] = labelencoder_X_Country.fit_transform(X[:, 1])

# encode the gender column categorical data.
labelencoder_X_Gender = LabelEncoder()
X[:, 2] = labelencoder_X_Gender.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# split the dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# initialising the ANN
classifier = Sequential()

# add the input and first hidden layers

# activation is activation function, rectifier. input_dim is input nodes. 
# units is output nodes (hidden layer) (input nodes + output nodes / 2), kernal initiliser is weighting the nodes.
classifier.add(Dense(activation = "relu", input_dim = 11, units = 6, kernel_initializer = "uniform"))

# add the second hidden layer
classifier.add(Dense(activation = "relu", units = 6, kernel_initializer = "uniform"))

# add the output layer
classifier.add(Dense(activation = "sigmoid", units = 1, kernel_initializer = "uniform"))

# compile the ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# predict the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
accuracy = accuracy_score(y_test, y_pred)

# save or load the model
model_location = "saved_model.h5"
classifier.model.save(model_location)
classifier = load_model(model_location)