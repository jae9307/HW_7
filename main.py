import pandas as pd
import numpy as np

def pca():
    data_frame = pd.read_csv("HW_CLUSTERING_SHOPPING_CART_v2245a.csv")

    covariance = pd.DataFrame.cov(data_frame)

    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    print(np.matrix(eigenvectors))
    print(np.matrix(eigenvalues))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pca()
