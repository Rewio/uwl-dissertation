from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

def __build_classifier__(X_train, y_train, hidden_nodes = 144, batch_size = 171, epochs = 89, optimizer = "adam", 
                         drop_out_rate = 0.5, num_hidden_layers = 2):

    classifier = Sequential()
	
    # add the input layer, and the first hidden layer.
    classifier.add(Dense(activation = "relu", input_dim = 77, units = hidden_nodes, kernel_initializer = "uniform"))
    classifier.add(Dropout(rate = drop_out_rate))
    
    for x in range(0, num_hidden_layers):
        
        # add a hidden layer with dropout
        classifier.add(Dense(activation = "relu", units = hidden_nodes, kernel_initializer="uniform"))
        classifier.add(Dropout(rate = drop_out_rate))
	
    # add the output layer
    classifier.add(Dense(activation = "sigmoid", units = 1, kernel_initializer = "uniform"))
	
    # compile the ANN
    classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["acc", "mae", "mse"])
	
    # fitting the ANN to the training set
    classifier.fit(X_train, y_train, batch_size = batch_size, epochs = epochs)
    return classifier

def random_search(X_train, y_train):
    
    # train the ANN
    classifier = KerasClassifier(build_fn = __build_classifier__)
    
    # hyperparamater options for tuning
    parameters = {"X_train" : [X_train], 
                  "y_train" : [y_train]}
    
    random_search = RandomizedSearchCV(estimator = classifier, param_distributions = parameters, n_iter = 1, scoring = 'accuracy')
    random_search.fit(X_train, y_train)
    return random_search