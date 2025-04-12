import pandas as pd
import numpy as np

def pca():
    data_frame = pd.read_csv("HW_CLUSTERING_SHOPPING_CART_v2245a.csv")
    data_frame = data_frame.drop('ID', axis=1)

    covariance = pd.DataFrame.cov(data_frame)
    print(covariance.to_string())

    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    print("Eigenvectors ", np.matrix(eigenvectors))
    print("Eigenvalues ", np.matrix(eigenvalues))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pca()
