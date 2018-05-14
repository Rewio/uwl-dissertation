# import libraries
import numpy as np
import pandas as pd

from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, matthews_corrcoef

import training
import plot

# import dataset
dataset = pd.read_csv("datasets/03-final/train-data.csv")

# import the first half of the test set, and append their dependent variables onto the end.
test_data1 = pd.read_csv("datasets/03-final/test-data1.csv")
test_data1.insert(len(test_data1.columns), "churn_yn", pd.read_csv("datasets/03-final/test-data1-labelled.csv", usecols=[1]))

# import the second half of the test set, and append their dependent variables onto the end.
test_data2 = pd.read_csv("datasets/03-final/test-data2.csv")
test_data2.insert(len(test_data2.columns), "churn_yn", pd.read_csv("datasets/03-final/test-data2-labelled.csv", usecols=[1]))

# merge the two halfs of the test set. !!! delete the variables created previously, for a tidier explorer.
dataset = pd.concat([dataset, test_data1, test_data2])
del test_data1, test_data2

# split our dataset out into independent & dependent variables, then divide that into train and test.
X = dataset.iloc[:, 3:-1].values # independent variables.
y = dataset.iloc[:, -1].values   # dependent variables..
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

# ensure that all values within the dataset are on the same scale.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# load the model that we have already trained.
model = load_model("models/03-final/saved_model.h5")

# cross validate our model, returning a classifier and the predictions from the cross-validation.
model_acc   = training.train_model(X_train, y_train)

# grab the accuracies from the cross-validation, or the saved file, and calculate mean and standard deviation.
accuracies  = model_acc[1]
accuracies  = np.load("models/03-final/saved_accuracies.npy")
mean        = accuracies.mean()
std_dev     = accuracies.std()

# predict the values of the test set, and begin collecting our evaluation metrics.
model_pred        = model.predict(X_test)
model_mse         = mean_squared_error(y_test, model_pred)
model_mae         = mean_absolute_error(y_test, model_pred)

# get the binary value of the predictions, then use this for more metrics.
model_values    = [1 if val > 0.5 else 0 for val in model_pred]
model_acc_score = accuracy_score(y_test, model_values)

# generate a confusion matrix, and generate the metrics that are associated to it.
con_matrix     = confusion_matrix(y_test, model_values)
tn, fp, fn, tp = confusion_matrix(y_test, model_values).ravel()
model_prec     = precision_score(y_test, model_values)
model_recall   = recall_score(y_test, model_values)
model_f1       = f1_score(y_test, model_values)
model_f1_mi    = f1_score(y_test, model_values, average="micro")
model_f1_ma    = f1_score(y_test, model_values, average="macro")
model_corcoef  = matthews_corrcoef(y_test, model_values)

# calculate the error between the predicted value, and the actual value.
model_errors = []
for counter, value in enumerate(model_pred):
    model_errors.append(abs(y_test[counter] - model_pred[counter])[0])
del counter, value

# plot the accuracies on a line graph
plot.plot_line_complete(np.arange(1, len(model_acc[1]) + 1, 1), accuracies)
plot.plot_line_focused (np.arange(1, len(model_acc[1]) + 1, 1), accuracies, mean)

# plot the errors using a scatter graph, and a histogram
plot.plot_scatter(model_errors, model_mse, model_mae)
bins = plot.plot_error_hist(model_errors)

# save the sizes for, and the value of, each bin used in the histogram.
bin_sizes = bins[0]
bin_value = bins[1]

# plot the confusion matrices.
plot.plot_confusion_matrix(con_matrix, [0, 1], normalize=True, title="Confusion Matrix of Predictions Normalized",
                           save_name="confusion_matrix_norm")
plot.plot_confusion_matrix(con_matrix, [0, 1], title="Confusion Matrix of Predictions", save_name="confusion_matrix")