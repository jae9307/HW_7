import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def pca():
    data_frame = pd.read_csv("HW_CLUSTERING_SHOPPING_CART_v2245a.csv")
    data_frame = data_frame.drop('ID', axis=1)

    means = np.mean(data_frame)
    matrix_of_means = [means for row in data_frame]
    centered_data = data_frame - matrix_of_means

    covariance = pd.DataFrame.cov(data_frame)

    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    print("Eigenvectors ", np.matrix(eigenvectors))
    print("Eigenvalues ", np.matrix(eigenvalues))

    # round the eigen values and vectors, sort the eigenvalues
    sorted_eigen_values = sorted(np.round(eigenvalues, decimals=3), reverse=True)
    eigenvectors = np.round(eigenvectors, decimals=2)

    total = sum(sorted_eigen_values)
    normalized_eigen_values = [eigen_value/total for eigen_value in sorted_eigen_values]

    # find and plot the cumulative sum of the normalized eigen values
    cumulative = np.cumsum(normalized_eigen_values)
    plt.plot(cumulative)
    plt.xlabel('Number of eigenvectors')
    plt.ylabel('Cumulative Sum')
    plt.show()

    # create a dictionary for each eigenvector, where the key is the attribute name and
    # the value is the value for that attribute in the eigenvector. this makes the eigenvectors
    # easier to read
    vector1 = {}
    vector2 = {}
    for index in range(len(data_frame.columns)):
        vector1[data_frame.columns[index]] = eigenvectors[0][index]
        vector2[data_frame.columns[index]] = eigenvectors[1][index]

    np.set_printoptions(linewidth=np.inf)
    print(list(data_frame.columns))
    print(f"eigenvector 1: {vector1}")
    print(f"eigenvector 2: {vector2}")

    projection1 = np.dot(data_frame, eigenvectors[0])
    projection2 = np.dot(data_frame, eigenvectors[1])

    # clear the figure so another plot can be made
    plt.clf()

    plt.scatter(projection1, projection2)
    plt.xlabel('Projection 1 amount')
    plt.ylabel('Projection 2 amount')
    plt.show()

    print(f"proj1: {projection1}")
    print(f"proj2: {projection2}")

    # reshape the data so that each data point is an array of its projection 1 amount
    # and its projection 2 amount
    projection_matrix = np.stack((projection1, projection2), axis=1)
    print(projection_matrix)

    # cluster the data using k-means
    model = KMeans(n_clusters = 2)
    model.fit(projection_matrix)
    model.predict(projection_matrix)
    cluster_centers = model.cluster_centers_
    print(cluster_centers)

    reprojection1 = (cluster_centers[0][0] * eigenvectors[0]) + (cluster_centers[0][1] * eigenvectors[0]) + means
    reprojection2 = (cluster_centers[1][0] * eigenvectors[1]) + (cluster_centers[1][1] * eigenvectors[1]) + means

    print(f"repro1: {reprojection1}")
    print(f"repro2: {reprojection2}")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pca()
