from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import metrics
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit

def build_classifier():

    classifier = Sequential()

    # add the input layer, and the first hidden layer.
    classifier.add(Dense(activation ="relu", input_dim = 77, units = 144, kernel_initializer ="uniform"))
    classifier.add(Dropout(rate = 0.5))
	
    # add the second hidden layer
    classifier.add(Dense(activation = "relu", units = 144, kernel_initializer="uniform"))
    classifier.add(Dropout(rate = 0.5))
	
    # add the output layer
    classifier.add(Dense(activation = "sigmoid", units = 1, kernel_initializer = "uniform"))
	
    # compile the ANN
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy", metrics.mae, metrics.mse])
    return classifier

def train_model(X_train, y_train):
    classifier = KerasClassifier(build_fn = build_classifier, batch_size = 171, epochs = 94)

    cv         = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = cv, verbose = 2)
    
    classifier.fit(X_train, y_train, batch_size = 171, epochs = 94)
    return [classifier, accuracies]