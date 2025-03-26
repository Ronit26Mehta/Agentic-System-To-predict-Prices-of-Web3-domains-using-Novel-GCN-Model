import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from pyvis.network import Network

####################################
# 1. Data Loading, Cleaning, and Normalization
####################################

def load_and_clean_data(csv_file):
    """
    Loads the scraped CSV data and cleans it:
      - Converts price strings (e.g. "$20,000,000") into float.
      - Converts watchlists (e.g. "4.4K") into a numeric value.
      - Drops rows where price or watchlists are missing.
    """
    df = pd.read_csv(csv_file)
    
    def parse_price(price_str):
        if isinstance(price_str, str):
            p = price_str.replace("$", "").replace(",", "")
            try:
                return float(p)
            except:
                return None
        return None

    df['price'] = df['price'].apply(parse_price)
    
    def parse_watchlists(watch_str):
        if isinstance(watch_str, str):
            if "K" in watch_str:
                try:
                    return float(watch_str.replace("K", "").strip()) * 1000
                except:
                    return None
            else:
                try:
                    return float(watch_str)
                except:
                    return None
        return None

    df['watchlists'] = df['watchlists'].apply(parse_watchlists)
    df = df.dropna(subset=['price', 'watchlists']).reset_index(drop=True)
    return df

def log_transform(X, Y):
    """
    Applies a log(1+x) transformation to both features and targets.
    """
    return np.log1p(X), np.log1p(Y)

####################################
# 2. Build kNN Graph, Normalize, and Combine with Similarity
####################################

def build_knn_graph(features, k=5):
    """
    Builds an undirected k-nearest neighbors graph from the given features.
    Returns the binary adjacency matrix A.
    """
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(features)
    distances, indices = nbrs.kneighbors(features)
    num_nodes = features.shape[0]
    A = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in indices[i]:
            if i != j:
                A[i, j] = 1
                A[j, i] = 1  # ensure symmetry
    return A

def normalize_adjacency(A):
    """
    Computes the normalized adjacency matrix with self-loops:
      Ĥ = D^(-1/2) (A + I) D^(-1/2)
    """
    A_hat = A + np.eye(A.shape[0])
    D = np.diag(np.sum(A_hat, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

def compute_similarity_matrix(X, tau=None):
    """
    Computes a similarity matrix S from the feature matrix X.
    For each pair (i, j):
         S_{ij} = exp( -|x_i - x_j| / tau )
    If tau is not provided, we use the mean absolute difference.
    """
    n = X.shape[0]
    diff = np.abs(X - X.T)
    if tau is None:
        tau = np.mean(diff)
    S = np.exp(- diff / (tau + 1e-5))
    return S

def combine_adjacency(A_norm, S, beta=0.5):
    """
    Combines the normalized adjacency matrix and similarity matrix using a weighted sum:
       A' = beta * A_norm + (1-beta) * S,
    then row-normalizes the result.
    """
    A_mod = beta * A_norm + (1 - beta) * S
    row_sum = np.sum(A_mod, axis=1, keepdims=True) + 1e-5
    A_combined = A_mod / row_sum
    return A_combined

####################################
# 3. Utility Functions for Activation and Metrics
####################################

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def mape(y_true, y_pred, epsilon=1e-5):
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))

####################################
# 4. Revised Domain-Aware Graph Convolution Model
####################################

class DomainAwareGCN:
    def __init__(self, in_channels, hidden_channels, out_channels):
        # Use a slightly larger initialization to compensate for log-transformed inputs.
        self.W1 = np.random.randn(in_channels, hidden_channels) * 0.1
        self.W2 = np.random.randn(hidden_channels, out_channels) * 0.1

    def forward(self, A_combined, X):
        """
        Forward pass:
          Z = A_combined @ X @ W1
          H = ReLU(Z)
          Y_pred = A_combined @ H @ W2
        """
        self.Z = A_combined @ X @ self.W1   # (n x hidden_channels)
        self.H = relu(self.Z)                # (n x hidden_channels)
        self.Y_pred = A_combined @ self.H @ self.W2  # (n x out_channels)
        return self.Y_pred

    def compute_loss(self, Y_pred, Y_true):
        loss = np.mean((Y_pred - Y_true)**2)
        return loss

    def backward(self, A_combined, X, Y_true):
        n = Y_true.shape[0]
        dY_pred = (self.Y_pred - Y_true) / n  # (n x out_channels)
        dW2 = self.H.T @ (A_combined @ dY_pred)
        dH = (A_combined @ dY_pred) @ self.W2.T
        dZ = dH * relu_deriv(self.Z)
        dW1 = (A_combined @ X).T @ dZ
        return dW1, dW2

    def update_weights(self, dW1, dW2, lr):
        self.W1 -= lr * dW1
        self.W2 -= lr * dW2

####################################
# 5. Training and Plotting Functions
####################################

def train_model(model, A_combined, X, Y, epochs=300, lr=0.005):
    loss_history = []
    mape_history = []
    for epoch in range(epochs):
        Y_pred = model.forward(A_combined, X)
        loss = model.compute_loss(Y_pred, Y)
        loss_history.append(loss)
        metric = mape(Y, Y_pred)
        mape_history.append(metric)
        dW1, dW2 = model.backward(A_combined, X, Y)
        model.update_weights(dW1, dW2, lr)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}, MAPE = {metric:.4f}")
    return loss_history, mape_history, model

def save_model(model, filename_prefix="domain_aware_gcn"):
    np.save(filename_prefix + "_W1.npy", model.W1)
    np.save(filename_prefix + "_W2.npy", model.W2)
    print(f"Model weights saved as {filename_prefix}_W1.npy and {filename_prefix}_W2.npy")

def plot_training(loss_history, mape_history):
    epochs = range(len(loss_history))
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, loss_history, label="MSE Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(epochs, mape_history, label="MAPE", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("MAPE")
    plt.title("Training MAPE")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_predictions(Y_true, Y_pred):
    plt.figure(figsize=(8,6))
    plt.scatter(Y_true, Y_pred, c='blue', label="Predicted")
    plt.plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()], 'r--', label="Ideal")
    plt.xlabel("Actual (log-transformed) Price")
    plt.ylabel("Predicted (log-transformed) Price")
    plt.title("Predicted vs. Actual Price")
    plt.legend()
    plt.show()

####################################
# 6. Interactive Graph Visualization (PyVis)
####################################

def visualize_graph(A, node_labels):
    """
    Converts the binary adjacency matrix A to a NetworkX graph and uses PyVis to create an interactive
    visualization saved as "graph.html".
    """
    G = nx.from_numpy_array(A)
    net = Network(height='750px', width='100%', notebook=False)
    net.from_nx(G)
    for node in net.nodes:
        node['label'] = node_labels[node['id']]
    net.show("graph.html", notebook=False)
    print("Graph visualization saved to graph.html")

####################################
# 7. Main Function
####################################

def main():
    csv_file = "listings_data.csv"
    if not os.path.exists(csv_file):
        print(f"CSV file '{csv_file}' not found. Please run your scraper first.")
        return
    
    # Load and clean data
    df = load_and_clean_data(csv_file)
    print(f"Loaded {len(df)} listings from {csv_file}")
    
    # Extract raw features and targets
    raw_features = df[['watchlists']].values  # (n, 1)
    raw_targets = df['price'].values.reshape(-1, 1)  # (n, 1)
    
    # Apply log transform to normalize scale
    features, targets = log_transform(raw_features, raw_targets)
    
    # Build kNN graph and normalized adjacency matrix (using raw features for graph construction)
    A = build_knn_graph(raw_features, k=5)
    A_norm = normalize_adjacency(A)
    
    # Compute similarity matrix based on raw features
    S = compute_similarity_matrix(raw_features)
    
    # Combine the normalized adjacency and similarity matrices
    # Beta controls the balance between structural and feature similarity information.
    A_combined = combine_adjacency(A_norm, S, beta=0.5)
    
    # Visualize the original binary kNN graph (not the combined one) for reference
    node_labels = df['domain_name'].tolist() if 'domain_name' in df.columns else [f"Node {i}" for i in range(len(df))]
    visualize_graph(A, node_labels)
    
    # Initialize the revised Domain-Aware GCN model
    in_channels = features.shape[1]   # Should be 1 (log-transformed watchlists)
    hidden_channels = 16
    out_channels = 1
    model = DomainAwareGCN(in_channels, hidden_channels, out_channels)
    
    # Train the model on log-transformed data using the combined adjacency
    loss_history, mape_history, trained_model = train_model(model, A_combined, features, targets, epochs=300, lr=0.005)
    
    # Save model weights
    save_model(trained_model, filename_prefix="domain_aware_gcn")
    
    # Plot training curves and predictions (in log-space)
    plot_training(loss_history, mape_history)
    Y_pred = trained_model.forward(A_combined, features)
    plot_predictions(targets, Y_pred)
    
if __name__ == "__main__":
    main()
