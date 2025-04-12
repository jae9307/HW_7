import pandas as pd
import numpy as np

def pca():
    data_frame = pd.read_csv("HW_CLUSTERING_SHOPPING_CART_v2245a.csv")
    data_frame = data_frame.drop('ID', axis=1)

    covariance = pd.DataFrame.cov(data_frame)

    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    eigenvectors = np.round(eigenvectors, decimals=2)
    eigenvalues = sorted(np.round(eigenvalues, decimals=3))
    print("Eigenvalues after sort ", eigenvalues)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pca()
