import numpy as np

def pca_manual(X, n_components):
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    centered_cov_matrix = np.cov(X_centered.T)
    
    eigenvalues, eigenvectors = np.linalg.eig(centered_cov_matrix)
    
    top_n = eigenvalues.argsort()[::-1][:n_components]
    projection = eigenvectors[:, top_n]

    pca_result = X_centered.dot(projection)
    
    return pca_result