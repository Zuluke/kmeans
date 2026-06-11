"""
Machine Learning Concepts Visualizations
Author: ML Specialist
Generates simple, clean visualizations for three fundamental ML concepts:
1. Clustering (K-means)
2. Density Estimation (KDE)
3. Dimensionality Reduction (PCA)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')


def visualization_clustering():
    """
    Create a horizontal visualization with two side-by-side plots showing clustering concept
    Left: Raw unlabeled data, Right: K-means clustering result
    """
    print("\nGenerating Clustering visualization (horizontal layout)...")
    
    # Generate synthetic data with 3 clear clusters
    np.random.seed(42)
    n_samples = 300
    X, y_true = make_blobs(n_samples=n_samples, centers=3, 
                           cluster_std=0.8, random_state=42)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Raw unlabeled data
    ax1.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, s=50, 
                edgecolors='white', linewidth=0.5)
    ax1.set_title('Original Unlabeled Data', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Feature 1', fontsize=11)
    ax1.set_ylabel('Feature 2', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal')
    
    # Right plot: Clustering result
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i in range(3):
        mask = y_kmeans == i
        ax2.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[i], alpha=0.6, s=50, 
                   edgecolors='white', linewidth=0.5,
                   label=f'Cluster {i+1}')
    
    # Plot centroids
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c='red', marker='X', s=200, linewidth=3,
               edgecolors='black', zorder=10, label='Centroids')
    
    # Add decision boundaries
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax2.contour(xx, yy, Z, colors='black', linewidths=1, 
                linestyles='--', alpha=0.5)
    
    ax2.set_title('K-means Clustering Result', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Feature 1', fontsize=11)
    ax2.set_ylabel('Feature 2', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_aspect('equal')
    
    # Add overall title and explanation
    fig.suptitle('Clustering: Finding Natural Groups in Unlabeled Data', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    explanation = "Clustering groups similar points together | Points in same cluster are more similar to each other"
    fig.text(0.5, 0.02, explanation, ha='center', fontsize=10, 
             style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('./clustering_concept.svg', format='svg', dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.show()
    print("  Saved: clustering_concept.svg (horizontal layout with two side-by-side plots)")


def visualization_density_estimation():
    """
    Create a simple, clean visualization of density estimation concept
    Shows how KDE estimates the probability distribution of data
    """
    print("\nGenerating Density Estimation visualization...")
    
    # Generate multimodal data
    np.random.seed(42)
    n_samples = 500
    
    # Create mixture of three distributions
    data1 = np.random.normal(-2, 0.5, n_samples//3)
    data2 = np.random.normal(0, 0.4, n_samples//3)
    data3 = np.random.normal(2.5, 0.6, n_samples//3)
    data = np.concatenate([data1, data2, data3])
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Histogram with KDE overlay
    ax1.hist(data, bins=40, density=True, alpha=0.6, 
             color='skyblue', edgecolor='black', label='Histogram')
    
    # Compute and plot KDE
    kde = KernelDensity(bandwidth=0.3, kernel='gaussian')
    kde.fit(data.reshape(-1, 1))
    x_range = np.linspace(data.min() - 1, data.max() + 1, 1000)
    log_density = kde.score_samples(x_range.reshape(-1, 1))
    density = np.exp(log_density)
    ax1.plot(x_range, density, 'r-', linewidth=2.5, label='KDE Estimate')
    
    ax1.set_title('1D Density Estimation\nKernel Density Estimate (KDE)', 
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('Value', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Right plot: 2D Density visualization with contour
    # Generate 2D data
    X_2d = np.random.randn(500, 2)
    X_2d[:200] = X_2d[:200] + np.array([-2, -2])
    X_2d[200:350] = X_2d[200:350] + np.array([2, 1])
    X_2d[350:] = X_2d[350:] + np.array([0, 3])
    
    # Compute 2D KDE
    kde_2d = gaussian_kde(X_2d.T)
    x_grid = np.linspace(X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1, 100)
    y_grid = np.linspace(X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1, 100)
    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density_2d = kde_2d(positions).reshape(xx.shape)
    
    # Plot contour
    contour = ax2.contourf(xx, yy, density_2d, levels=15, cmap='hot', alpha=0.7)
    ax2.contour(xx, yy, density_2d, levels=8, colors='black', 
                linewidths=0.5, alpha=0.5)
    ax2.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.3, s=20, c='blue')
    
    ax2.set_title('2D Density Estimation\nContour Plot of Density', 
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('Feature 1', fontsize=11)
    ax2.set_ylabel('Feature 2', fontsize=11)
    plt.colorbar(contour, ax=ax2, label='Density')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_aspect('equal')
    
    # Add overall title and explanation (moved lower to avoid occlusion)
    fig.suptitle('Density Estimation: Understanding Data Distribution', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    explanation = "Density estimation reveals where data points are concentrated\n"
    explanation += "KDE provides a smooth, continuous estimate of the probability density"
    fig.text(0.5, 0.01, explanation, ha='center', fontsize=10, 
             style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Add more space at bottom for the explanation box
    plt.savefig('./density_estimation_concept.svg', format='svg', dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.show()
    print("  Saved: density_estimation_concept.svg")


def visualization_dimensionality_reduction():
    """
    Create a simple, clean visualization of dimensionality reduction concept
    Shows how PCA projects high-dimensional data to 2D while preserving structure
    """
    print("\nGenerating Dimensionality Reduction visualization...")
    
    # Generate Swiss roll dataset (3D with non-linear structure)
    np.random.seed(42)
    X_3d, color = make_swiss_roll(n_samples=800, noise=0.1, random_state=42)
    
    # Apply PCA to reduce from 3D to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_3d)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 6))
    
    # Left plot: 3D original data
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    scatter1 = ax1.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], 
                           c=color, cmap='viridis', alpha=0.6, s=20)
    ax1.set_title(f'Original Data: 3 Dimensions\n(3D Swiss Roll)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.set_zlabel('Z', fontsize=10)
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, label='Position')
    
    # Right plot: 2D projected data
    ax2 = fig.add_subplot(1, 2, 2)
    scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=color, 
                           cmap='viridis', alpha=0.6, s=30)
    ax2.set_title(f'After PCA: Reduced to 2 Dimensions\n'
                  f'Preserves {pca.explained_variance_ratio_[0]:.1%} + '
                  f'{pca.explained_variance_ratio_[1]:.1%} = '
                  f'{sum(pca.explained_variance_ratio_):.1%} of variance', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%})', 
                   fontsize=11)
    ax2.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%})', 
                   fontsize=11)
    plt.colorbar(scatter2, ax=ax2, label='Original Position')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_aspect('equal')
    
    # Add overall title and explanation
    fig.suptitle('Dimensionality Reduction: Simplifying Complex Data\n(PCA - Principal Component Analysis)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    explanation = f"Dimensionality reduction transforms data from {X_3d.shape[1]}D to 2D\n"
    explanation += "while preserving the most important structure and relationships"
    fig.text(0.5, 0.02, explanation, ha='center', fontsize=10, 
             style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('./dimensionality_reduction_concept.svg', format='svg', dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.show()
    print("  Saved: dimensionality_reduction_concept.svg")


def main():
    """Generate all three concept visualizations"""
    print("="*60)
    print("MACHINE LEARNING CONCEPTS VISUALIZATIONS")
    print("="*60)
    print("\nGenerating simple, clean visualizations for:")
    print("  1. Clustering (Agrupamento) - Horizontal layout with two side-by-side plots")
    print("  2. Density Estimation (Estimativa de Densidade)")
    print("  3. Dimensionality Reduction (Redução de Dimensionalidade)")
    
    # Generate all visualizations
    visualization_clustering()
    visualization_density_estimation()
    visualization_dimensionality_reduction()
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print("\nGenerated files (SVG format only):")
    print("  1. clustering_concept.svg - Horizontal layout (original data + clustering result)")
    print("  2. density_estimation_concept.svg - 1D and 2D density estimation")
    print("  3. dimensionality_reduction_concept.svg - PCA 3D to 2D")
    print("\nAll images are saved in the current directory (./)")
    print("\nKey concepts illustrated:")
    print("  - Clustering: Groups similar points together, finds natural patterns")
    print("  - Density Estimation: Shows where data is concentrated")
    print("  - Dimensionality Reduction: Simplifies data while preserving structure")
    print("\nThe clustering image now shows a comparison between raw data and clustered result")
    print("All images are saved in SVG format only (no PNG files)")
    print("The density estimation explanation box has been moved lower to avoid occlusion")


if __name__ == "__main__":
    main()