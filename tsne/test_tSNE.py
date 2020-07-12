# testing code from https://github.com/KlugerLab/FIt-SNE/blob/master/examples/test.ipynb


import numpy as np
import pylab as plt
import seaborn as sns; sns.set()


from fast_tsne import fast_tsne



# Load MNIST data
from keras.datasets import mnist


if __name__ == "__main__":
    # (60000, 28, 28), range: 0~255          
    # shape (60000, 784), (60000, ) ; (10000, 784), (10000, )
    (x_train, y_train), (x_test, y_test) = mnist.load_data() 
    x_train = x_train.reshape(60000, 784).astype('float64') / 255
    x_test  =  x_test.reshape(10000, 784).astype('float64') / 255
    X = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    print(X.shape) # (70000, 784)

    # Do PCA and keep 50 dimensions
    X = X - X.mean(axis=0)  # (70000, 784)
    U, s, V = np.linalg.svd(X, full_matrices=False) # 
    X50 = np.dot(U, np.diag(s))[:,:50]

    # We will use PCA initialization later on
    PCAinit = X50[:,:2] / np.std(X50[:,0]) * 0.0001

    # 10 nice colors
    col = np.array(['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99',
                    '#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a'])

    Z = fast_tsne(X50, perplexity=50, seed=42)

    plt.figure(figsize=(5,5))
    plt.scatter(Z[:,0], Z[:,1], c=col[y], s=1)
    plt.tight_layout()
    plt.show()