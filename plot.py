import matplotlib.pyplot as plt
import numpy as np
import itertools
    
def plot_scatter(error, mse, mas):
       
    fig, ax = plt.subplots()
    
    ax.plot(range(1, len(error) + 1), [mse] * len(error), linewidth=0.5, c="blue")  
    ax.plot(range(1, len(error) + 1), [mas] * len(error), linewidth=0.5, c="green")  
    plt.scatter(range(0, len(error)), error, s=0.25, c="red") 
    
    major_ticks_x = np.arange(0, 2250, 250)
    minor_ticks_x = np.arange(0, 2050, 50)
    major_ticks_y = np.arange(0, 1.1, 0.1)
    minor_ticks_y = np.arange(0, 1.025, 0.025)
    
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    
    # And a corresponding grid
    ax.grid(which='both')
    
    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.15)
    ax.grid(which='major', alpha=0.3)    
    
    plt.title("Scatter Plot showing Inacccuracy for Each Prediction Made")
    plt.xlabel("Test Set Entry")
    plt.ylabel("Degree of Inaccuracy")
    
    plt.show()
    
    fig.set_size_inches(10, 10)
    fig.savefig("plots/error-scatter.png", dpi=2000, bbox_inches='tight')
    
def plot_error_hist(error):
    
    fig, ax = plt.subplots()
    
    major_ticks_x = np.arange(0, 1.1, 0.1)
    minor_ticks_x = np.arange(0, 1.05, 0.05)
    major_ticks_y = np.arange(0, 110, 10)
    minor_ticks_y = np.arange(0, 110, 1)
    
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    
    # And a corresponding grid
    ax.grid(which='both')
    
    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.15)
    ax.grid(which='major', alpha=0.3)
    
    (n, bins, patches) = plt.hist(error, bins = 500, rwidth = 0.3, align="mid", color="red")

    plt.title("Histogram showing Inaccuracy for Each Prediction Made")
    plt.xlabel("Degree of Inaccuracy")    
    plt.ylabel("Number of Occurences")
    
    plt.show()
    
    fig.set_size_inches(10, 10)
    fig.savefig("plots/error-hist.png", dpi=2000, bbox_inches='tight')
    
    return (n, bins, patches)

def plot_line_complete(x, y):
    fig, ax = plt.subplots()
    
    plt.plot(x, y, "r-o")

    # define what the ticks are along each axis.
    major_ticks_x = np.arange(1, 11, 1)
    major_ticks_y = np.arange(0, 1.1, 0.1)
    minor_ticks_y = np.arange(0, 1.025, 0.025)
    
    # set the ticks on the plot using those defined.
    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    
    # setup the grid and it's opacity.
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.15)
    ax.grid(which='major', alpha=0.3)

    plt.title("Accuracies Achieved using Cross Validation")
    plt.xlabel("Test Number")
    plt.ylabel("Accuracy Achieved")
    
    plt.show()

    fig.set_size_inches(10, 10)
    fig.savefig("plots/accuracies-line-complete.png", dpi=2000, bbox_inches='tight')    

def plot_line_focused(x, y, mean):
    fig, ax = plt.subplots()
    
    ax.plot(range(1, len(x) + 1), [mean] * len(x), linewidth=0.5, c="red")  
    plt.plot(x, y, "r-o")
    
    # define what the ticks are along each axis.
    major_ticks_x = np.arange(1, 11, 1)
    major_ticks_y = np.arange(0.7, 0.79, 0.01)
    minor_ticks_y = np.arange(0.7, 0.7925, 0.0025)
    
    # set the ticks on the plot using those defined.
    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    
    # setup the grid and it's opacity.
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.15)
    ax.grid(which='major', alpha=0.3)
    
    plt.title("Accuracies Achieved using Cross Validation (Focused)")
    plt.xlabel("Test Number")
    plt.ylabel("Accuracy Achieved")
    
    plt.show()

    fig.set_size_inches(10, 10)
    fig.savefig("plots/accuracies-line-focused.png", dpi=2000, bbox_inches='tight')
    
# function courtesy of: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds,
                          save_name="Confusion Matrix"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    fig, ax = plt.subplots()
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    fig.set_size_inches(10, 10)
    fig.savefig("plots/" + save_name + ".png", dpi=2000, bbox_inches='tight')