import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt

import os
import gzip
import numpy as np

mnist_data_path="./data/fashion_mnist"

def load_mnist(path=mnist_data_path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def get_data():
    # load the data
    # return X_train, X_test, y_train, y_test
    X_train, y_train = load_mnist(kind="train")
    X_test, y_test = load_mnist(kind="test")
    return X_train, X_test, y_train, y_test
    

def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    # normalize the data
    # gaussian distribuition with mean as 0
    # Standard scale (Z-score normalization)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8 # add epsilon to avoid div by zero
    
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm


def plot_metrics(metrics: list) -> None:
    # Get unique C values and k values from the metrics list
    C_values = sorted(list(set([m[0] for m in metrics])))
    k_values = sorted(list(set([m[1] for m in metrics])))
    
    # Create a 2x2 grid of subplots for the 4 metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SVM Performance Metrics vs PCA Components (for different C values)', fontsize=16)
    
    # Flatten axes for easy iteration
    ax_acc, ax_prec, ax_rec, ax_f1 = axes.flatten()
    
    # Define distinct colors and markers for the different C lines
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'd', 'x']
    
    # Loop over each C value to draw its line on all 4 subplots
    for i, C_i in enumerate(C_values):
        # Extract the data specifically for this C value
        c_data = [m for m in metrics if m[0] == C_i]
        
        # Sort by k just to be safe before plotting
        c_data.sort(key=lambda x: x[1]) 
        
        ks = [m[1] for m in c_data]
        accs = [m[2] for m in c_data]
        precs = [m[3] for m in c_data]
        recs = [m[4] for m in c_data]
        f1s = [m[5] for m in c_data]
        
        # Plot Accuracy
        ax_acc.plot(ks, accs, color=colors[i], marker=markers[i], label=f'C={C_i}')
        # Plot Precision
        ax_prec.plot(ks, precs, color=colors[i], marker=markers[i], label=f'C={C_i}')
        # Plot Recall
        ax_rec.plot(ks, recs, color=colors[i], marker=markers[i], label=f'C={C_i}')
        # Plot F1 Score
        ax_f1.plot(ks, f1s, color=colors[i], marker=markers[i], label=f'C={C_i}')
        
    # Formatting for all subplots
    plot_configs = [
        (ax_acc, 'Accuracy'),
        (ax_prec, 'Precision'),
        (ax_rec, 'Recall'),
        (ax_f1, 'F1 Score')
    ]
    
    for ax, title in plot_configs:
        ax.set_title(title)
        ax.set_xlabel('Number of Principal Components (k)')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Make room for the main suptitle
    plt.savefig('pca_metrics.png', dpi=300)
    plt.show()