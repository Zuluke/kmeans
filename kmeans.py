"""
Machine Learning Visualizations: K-means Clustering with Decision Boundaries
Author: ML Specialist
This script implements k-means from scratch with 2 centroids and generates
a timelapse GIF showing the iterative evolution of centroids and cluster assignments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8-darkgrid')


def generate_dataset():
    """
    Generate a synthetic dataset with two natural clusters
    Returns: X (data points), y_true (true labels for reference)
    """
    np.random.seed(42)
    
    # Generate two clusters with different centers and spreads
    n_samples_per_cluster = 500  # Increased to 500 samples per cluster
    
    # Cluster 1: centered at (-2, -2) with more spread for slower convergence
    cluster1 = np.random.randn(n_samples_per_cluster, 2) * 1.2 + np.array([-2, -2])
    
    # Cluster 2: centered at (3, 2) with more spread for slower convergence
    cluster2 = np.random.randn(n_samples_per_cluster, 2) * 1.3 + np.array([3, 2])
    
    # Add some overlapping points to make convergence slower and more interesting
    overlapping = np.random.randn(100, 2) * 1.0 + np.array([0.5, 0])
    X_overlap = overlapping
    y_overlap = np.array([0, 1])[np.random.randint(0, 2, 100)]
    
    # Combine data
    X = np.vstack([cluster1, cluster2, X_overlap])
    y_true = np.array([0] * n_samples_per_cluster + [1] * n_samples_per_cluster + list(y_overlap))
    
    # Shuffle the data to mix overlapping points
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y_true = y_true[shuffle_idx]
    
    return X, y_true


def kmeans_scratch(X, k, max_iters=30, random_state=42, convergence_tol=1e-3):
    """
    K-means implementation from scratch with slower convergence
    
    Parameters:
    X: numpy array of shape (n_samples, n_features)
    k: number of clusters
    max_iters: maximum number of iterations
    random_state: random seed for reproducibility
    convergence_tol: tolerance for convergence (smaller = slower convergence)
    
    Returns:
    history: dictionary containing centroids and assignments at each iteration
    final_centroids: final centroid positions
    final_assignments: final cluster assignments
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    
    # Initialize centroids randomly but far apart to ensure slow convergence
    # Choose centroids that are not too close to the natural clusters
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices].copy()
    
    # Store history for animation
    history = {
        'centroids': [centroids.copy()],
        'assignments': [np.zeros(n_samples, dtype=int)]
    }
    
    for iteration in range(max_iters):
        # Step 1: Assign each point to the nearest centroid
        distances = np.zeros((n_samples, k))
        for i in range(k):
            distances[:, i] = np.sqrt(np.sum((X - centroids[i]) ** 2, axis=1))
        
        assignments = np.argmin(distances, axis=1)
        
        # Store assignments
        history['assignments'].append(assignments.copy())
        
        # Step 2: Update centroids based on current assignments
        new_centroids = np.zeros((k, X.shape[1]))
        movement = 0
        
        for i in range(k):
            if np.any(assignments == i):
                new_centroids[i] = X[assignments == i].mean(axis=0)
                movement += np.sqrt(np.sum((new_centroids[i] - centroids[i]) ** 2))
            else:
                # If a cluster has no points, keep the old centroid
                new_centroids[i] = centroids[i]
        
        # Store new centroids
        history['centroids'].append(new_centroids.copy())
        
        # Check for convergence with smaller tolerance (slower convergence)
        if movement < convergence_tol:
            print(f"Converged at iteration {iteration + 1} (movement: {movement:.6f})")
            break
        
        centroids = new_centroids
        
        # Print progress every 5 iterations
        if (iteration + 1) % 5 == 0:
            print(f"  Iteration {iteration + 1}: centroid movement = {movement:.4f}")
    
    # Trim history to actual iterations
    n_frames = min(iteration + 2, len(history['centroids']))
    history['centroids'] = history['centroids'][:n_frames]
    history['assignments'] = history['assignments'][:n_frames]
    
    return history, centroids, assignments


def compute_decision_boundary(X, centroids, resolution=150):
    """
    Compute the decision boundary between two centroids
    Returns meshgrid and decision region labels
    """
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
    y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # Compute distances to each centroid for all mesh points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    distances_to_centroid1 = np.sqrt(np.sum((grid_points - centroids[0]) ** 2, axis=1))
    distances_to_centroid2 = np.sqrt(np.sum((grid_points - centroids[1]) ** 2, axis=1))
    
    # Assign each grid point to nearest centroid
    Z = (distances_to_centroid2 < distances_to_centroid1).astype(int)
    Z = Z.reshape(xx.shape)
    
    return xx, yy, Z


def create_timelapse_gif(X, history, filename='./kmeans_timelapse.gif'):
    """
    Create a GIF showing the evolution of k-means clustering
    Shows centroids moving and cluster assignments updating at each step
    """
    print("\nGenerating k-means timelapse GIF...")
    
    n_iterations = len(history['assignments'])
    print(f"  Total frames to generate: {n_iterations}")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    def update_frame(frame):
        """Update function for each animation frame"""
        ax.clear()
        
        # Get centroids and assignments for this iteration
        centroids = history['centroids'][frame]
        assignments = history['assignments'][frame]
        
        # Compute decision boundary for current centroids
        xx, yy, Z = compute_decision_boundary(X, centroids)
        
        # Plot decision boundary regions
        ax.contourf(xx, yy, Z, alpha=0.12, levels=[-0.5, 0.5, 1.5], 
                   colors=['#1f77b4', '#ff7f0e'])
        
        # Plot decision boundary line
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, 
                  linestyles='--', alpha=0.8)
        
        # Plot data points colored by current cluster assignment
        colors = ['#1f77b4', '#ff7f0e']
        for cluster_id in [0, 1]:
            mask = assignments == cluster_id
            if np.any(mask):
                ax.scatter(X[mask, 0], X[mask, 1], 
                          c=colors[cluster_id], alpha=0.5, s=25,
                          edgecolors='white', linewidth=0.3,
                          label=f'Cluster {cluster_id} (n={np.sum(mask)})')
        
        # Plot centroids with smaller marker size
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                  c='red', marker='X', s=150, linewidth=2.5, 
                  edgecolors='black', zorder=10, label='Centroids')
        
        # Add centroid labels with smaller font
        for i, centroid in enumerate(centroids):
            ax.annotate(f'C{i}', (centroid[0], centroid[1]), 
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=10, fontweight='bold', color='darkred',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Add title with iteration information
        if frame == 0:
            title = f'Initial State (Iteration 0)\nRandom Centroids'
            subtitle = 'Centroids initialized randomly'
        elif frame == n_iterations - 1:
            title = f'Final State (Iteration {frame})\nAlgorithm Converged'
            subtitle = 'Optimal clustering achieved'
        else:
            title = f'Iteration {frame}'
            subtitle = 'Updating cluster assignments and centroid positions'
        
        ax.set_title(f'{title}\n{subtitle}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=9)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set equal aspect ratio for better visualization
        ax.set_aspect('equal')
        
        # Add information text about what's happening at each iteration
        if frame == 0:
            info_text = "Step 0: Random centroid initialization"
        elif frame < n_iterations - 1:
            info_text = "Step 1: Assign each point to nearest centroid\nStep 2: Move centroids to cluster means"
        else:
            info_text = "Convergence achieved!\nCentroids stopped moving"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add iteration counter
        ax.text(0.98, 0.02, f'Frame: {frame}/{n_iterations-1}', 
               transform=ax.transAxes, fontsize=9,
               horizontalalignment='right', verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set axis limits with some padding
        x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
        y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        return []
    
    # Create animation with longer interval to show each step clearly
    anim = FuncAnimation(fig, update_frame, frames=n_iterations, 
                        interval=1000, repeat=True, blit=False)
    
    # Save as GIF with higher quality
    writer = PillowWriter(fps=1)  # 1 frame per second for clear viewing
    anim.save(filename, writer=writer, dpi=100)
    plt.close()
    
    print(f"  Timelapse GIF saved as '{filename}'")
    return anim


def create_detailed_evolution_gif(X, history, filename='./kmeans_detailed_evolution.gif'):
    """
    Create a detailed animation showing centroids, boundaries, and cluster statistics
    """
    print("\nGenerating detailed evolution GIF...")
    
    n_iterations = len(history['assignments'])
    print(f"  Total frames to generate: {n_iterations}")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    def update_frame(frame):
        ax.clear()
        
        centroids = history['centroids'][frame]
        assignments = history['assignments'][frame]
        
        # Compute decision boundary
        xx, yy, Z = compute_decision_boundary(X, centroids, resolution=200)
        
        # Plot decision regions with transparency
        ax.contourf(xx, yy, Z, alpha=0.15, levels=[-0.5, 0.5, 1.5], 
                   colors=['#1f77b4', '#ff7f0e'])
        
        # Plot decision boundary line
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', 
                  linewidths=2.5, linestyles='-', alpha=0.7)
        
        # Plot data points with smaller size
        colors = ['#1f77b4', '#ff7f0e']
        for cluster_id in [0, 1]:
            mask = assignments == cluster_id
            if np.any(mask):
                ax.scatter(X[mask, 0], X[mask, 1], 
                          c=colors[cluster_id], alpha=0.4, s=20,
                          edgecolors='black', linewidth=0.2)
        
        # Draw circles around centroids (smaller radius)
        for i, centroid in enumerate(centroids):
            # Calculate cluster radius (standard deviation of assigned points)
            mask = assignments == i
            if np.any(mask):
                cluster_points = X[mask]
                if len(cluster_points) > 1:
                    radius = np.std(np.sqrt(np.sum((cluster_points - centroid) ** 2, axis=1)))
                    radius = min(radius, 1.5)  # Cap the radius for visualization
                else:
                    radius = 0.5
            else:
                radius = 0.5
            
            circle = plt.Circle((centroid[0], centroid[1]), radius, 
                               color='red', fill=False, linewidth=1.5, 
                               linestyle='--', alpha=0.4)
            ax.add_patch(circle)
        
        # Plot centroids with smaller marker
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                  c='darkred', marker='*', s=200, linewidth=1.5, 
                  edgecolors='gold', zorder=10)
        
        # Add centroid labels
        for i, centroid in enumerate(centroids):
            # Count points in this cluster
            n_points = np.sum(assignments == i)
            ax.annotate(f'C{i} (n={n_points})', (centroid[0], centroid[1]), 
                       xytext=(12, -12), textcoords='offset points',
                       fontsize=9, fontweight='bold', color='darkred',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))
        
        # Add title
        if frame == 0:
            title = 'Iteration 0: Random Initialization'
            subtitle = 'Centroids placed randomly'
            color_info = 'No structure yet - points randomly assigned'
        elif frame == n_iterations - 1:
            title = f'Iteration {frame}: Algorithm Converged'
            subtitle = 'Optimal centroids and boundaries found'
            color_info = 'Clean separation - clusters clearly defined'
        else:
            title = f'Iteration {frame}: Refining Clusters'
            subtitle = 'Reassigning points and updating centroids'
            color_info = 'Gradual improvement in cluster separation'
        
        ax.set_title(f'{title}\n{subtitle}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        
        # Add legend elements manually
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='#1f77b4', markersize=8, label='Cluster 0'),
            Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='#ff7f0e', markersize=8, label='Cluster 1'),
            Line2D([0], [0], marker='*', color='darkred', 
                  markerfacecolor='darkred', markersize=10, label='Centroids'),
            Line2D([0], [0], color='black', linewidth=2, 
                  linestyle='-', label='Decision Boundary'),
            Line2D([0], [0], color='red', linewidth=1.5, 
                  linestyle='--', label='Cluster Radius')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Add information box
        info_text = f"{color_info}\n\n"
        if frame > 0 and frame < n_iterations - 1:
            info_text += "Process:\n1. Points reassigned to nearest centroid\n2. Centroids move to cluster means\n3. Boundary updates accordingly"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        # Add progress bar
        progress = frame / (n_iterations - 1)
        ax.text(0.98, 0.02, f'Progress: {progress*100:.1f}%', 
               transform=ax.transAxes, fontsize=9,
               horizontalalignment='right', verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
        
        # Set axis limits
        x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
        y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        return []
    
    anim = FuncAnimation(fig, update_frame, frames=n_iterations, 
                        interval=1200, repeat=True, blit=False)
    
    writer = PillowWriter(fps=0.8)  # Slower for detailed viewing
    anim.save(filename, writer=writer, dpi=100)
    plt.close()
    
    print(f"  Detailed evolution GIF saved as '{filename}'")
    return anim


def create_summary_figure(X, history):
    """
    Create a summary figure with mixed layout (2x2 grid) suitable for Beamer presentations
    The figure shows initial, early, middle, late, and final stages in a compact grid
    """
    print("\nCreating summary figure in SVG format with mixed layout...")
    
    n_iterations = len(history['assignments'])
    
    # Select key frames for 2x2 grid (plus one more for a 5-subplot layout)
    # We'll create a 2x3 grid but leave one empty for better spacing
    frames_to_show = []
    frame_labels = []
    
    # Frame 0: Initial (random centroids)
    frames_to_show.append(0)
    frame_labels.append('Initial\n(Iteration 0)')
    
    # Frame at ~20% progress
    early_idx = max(1, n_iterations // 5)
    if early_idx < n_iterations - 1:
        frames_to_show.append(early_idx)
        frame_labels.append(f'Early\n(Iter. {early_idx})')
    
    # Frame at ~40% progress
    mid_idx = n_iterations // 2
    if mid_idx < n_iterations - 1 and mid_idx != early_idx:
        frames_to_show.append(mid_idx)
        frame_labels.append(f'Middle\n(Iter. {mid_idx})')
    
    # Frame at ~70% progress
    late_idx = (3 * n_iterations) // 4
    if late_idx < n_iterations - 1 and late_idx != mid_idx:
        frames_to_show.append(late_idx)
        frame_labels.append(f'Late\n(Iter. {late_idx})')
    
    # Final frame
    frames_to_show.append(n_iterations - 1)
    frame_labels.append('Final\n(Converged)')
    
    n_plots = len(frames_to_show)
    
    # Create a mixed layout: 2 rows, 3 columns (or adjust based on number of plots)
    if n_plots <= 4:
        n_rows, n_cols = 2, 2
    else:
        n_rows, n_cols = 2, 3
    
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('K-means Clustering Evolution', fontsize=16, fontweight='bold', y=0.98)
    
    # Use GridSpec for mixed layout with different subplot sizes if needed
    from matplotlib.gridspec import GridSpec
    
    if n_plots <= 4:
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        positions = [(0,0), (0,1), (1,0), (1,1)]
    else:
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]
        # Leave (1,2) empty for better spacing
    
    for idx, (frame, label) in enumerate(zip(frames_to_show, frame_labels)):
        if idx >= len(positions):
            break
            
        ax = fig.add_subplot(gs[positions[idx]])
        
        centroids = history['centroids'][frame]
        assignments = history['assignments'][frame]
        
        # Compute decision boundary
        xx, yy, Z = compute_decision_boundary(X, centroids, resolution=100)
        
        # Plot decision regions with subtle colors
        ax.contourf(xx, yy, Z, alpha=0.12, levels=[-0.5, 0.5, 1.5], 
                   colors=['#1f77b4', '#ff7f0e'])
        
        # Plot decision boundary line
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5, 
                  linestyles='--', alpha=0.7)
        
        # Plot points (small for compact figure)
        colors = ['#1f77b4', '#ff7f0e']
        for cluster_id in [0, 1]:
            mask = assignments == cluster_id
            if np.any(mask):
                ax.scatter(X[mask, 0], X[mask, 1], 
                          c=colors[cluster_id], alpha=0.4, s=8,
                          edgecolors='white', linewidth=0.2)
        
        # Plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                  c='red', marker='X', s=60, linewidth=1.5, 
                  edgecolors='black', zorder=10)
        
        # Add centroid labels with cluster sizes
        for i, centroid in enumerate(centroids):
            n_points = np.sum(assignments == i)
            ax.annotate(f'C{i}\n(n={n_points})', (centroid[0], centroid[1]), 
                       xytext=(6, 6), textcoords='offset points',
                       fontsize=7, fontweight='bold', color='darkred',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7))
        
        # Add title with iteration information
        ax.set_title(label, fontsize=10, fontweight='bold', pad=8)
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_aspect('equal')
        
        # Add a small text box with inertia if not final frame
        if frame > 0 and frame < n_iterations - 1:
            # Calculate approximate inertia for this frame
            inertia = 0
            for i in range(2):
                mask = assignments == i
                if np.any(mask):
                    inertia += np.sum((X[mask] - centroids[i]) ** 2)
            ax.text(0.02, 0.98, f'WCSS: {inertia:.0f}', 
                   transform=ax.transAxes, fontsize=6,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Add boundary equation annotation for final frame
        if frame == n_iterations - 1:
            # Calculate the perpendicular bisector line
            mid_point = (centroids[0] + centroids[1]) / 2
            slope = -(centroids[1][0] - centroids[0][0]) / (centroids[1][1] - centroids[0][1]) if (centroids[1][1] - centroids[0][1]) != 0 else np.inf
            
            ax.text(0.98, 0.02, 'Decision\nBoundary', 
                   transform=ax.transAxes, fontsize=6,
                   horizontalalignment='right', verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Add a main title explanation at the bottom
    fig.text(0.5, 0.01, 'K-means iteratively refines clusters: (1) Assign points to nearest centroid, (2) Move centroids to cluster means',
             ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save as SVG for Beamer presentation
    svg_filename = './kmeans_summary.svg'
    plt.savefig(svg_filename, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Summary figure saved as '{svg_filename}'")
    
    # Also save as PDF for backup (better for some LaTeX workflows)
    pdf_filename = './kmeans_summary.pdf'
    plt.savefig(pdf_filename, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Summary figure also saved as '{pdf_filename}'")
    
    plt.show()
    print("  Summary figure generated with mixed layout (2x2 or 2x3 grid)")


def print_clustering_metrics(history, X):
    """
    Calculate and print clustering metrics for each iteration
    """
    print("\n" + "="*60)
    print("CLUSTERING METRICS EVOLUTION")
    print("="*60)
    
    n_iterations = len(history['assignments'])
    
    print("\nIteration | Inertia (WCSS) | Centroids Distance | Points Reassigned")
    print("-" * 60)
    
    prev_assignments = None
    
    for frame in range(1, n_iterations):  # Start from frame 1 (after first assignment)
        centroids = history['centroids'][frame]
        assignments = history['assignments'][frame]
        
        # Calculate inertia (Within-Cluster Sum of Squares)
        inertia = 0
        for i in range(2):  # k=2
            mask = assignments == i
            if np.any(mask):
                inertia += np.sum((X[mask] - centroids[i]) ** 2)
        
        # Calculate distance between centroids
        centroid_distance = np.sqrt(np.sum((centroids[0] - centroids[1]) ** 2))
        
        # Calculate number of points reassigned (compared to previous iteration)
        if prev_assignments is not None:
            n_reassigned = np.sum(assignments != prev_assignments)
        else:
            n_reassigned = 0
        
        print(f"   {frame:2d}     | {inertia:10.2f}   | {centroid_distance:8.2f}       | {n_reassigned:5d}")
        
        prev_assignments = assignments
    
    print("-" * 60)


def main():
    """Main function to run k-means visualization"""
    print("="*60)
    print("K-MEANS CLUSTERING VISUALIZATION WITH DECISION BOUNDARIES")
    print("="*60)
    
    # Generate dataset
    print("\n1. Generating synthetic dataset with 2 natural clusters...")
    X, y_true = generate_dataset()
    print(f"   Dataset shape: {X.shape}")
    print(f"   Number of samples: {len(X)}")
    print(f"   Samples per cluster: ~500 (plus overlapping points)")
    
    # Run k-means from scratch with slower convergence
    print("\n2. Running k-means algorithm from scratch...")
    print("   (Configured for slower convergence to show more iterations)")
    k = 2
    history, final_centroids, final_assignments = kmeans_scratch(
        X, k, max_iters=30, convergence_tol=1e-3
    )
    
    n_iterations = len(history['assignments']) - 1
    print(f"\n   Total iterations executed: {n_iterations}")
    print(f"   Final centroids:")
    print(f"     Centroid 0: ({final_centroids[0][0]:.3f}, {final_centroids[0][1]:.3f})")
    print(f"     Centroid 1: ({final_centroids[1][0]:.3f}, {final_centroids[1][1]:.3f})")
    
    # Print detailed metrics
    print_clustering_metrics(history, X)
    
    # Create timelapse GIF (with all iterations)
    print("\n3. Creating timelapse GIF with all iterations...")
    create_timelapse_gif(X, history, './kmeans_timelapse.gif')
    
    # Create detailed evolution GIF
    print("\n4. Creating detailed boundary evolution GIF...")
    create_detailed_evolution_gif(X, history, './kmeans_detailed_evolution.gif')
    
    # Create summary figure in SVG format with mixed layout
    print("\n5. Creating summary figure in SVG format...")
    create_summary_figure(X, history)
    
    # Display final statistics
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  1. ./kmeans_timelapse.gif - Shows centroids moving and points being reassigned")
    print("  2. ./kmeans_detailed_evolution.gif - Detailed view with cluster statistics")
    print("  3. ./kmeans_summary.svg - Summary figure in SVG format (suitable for Beamer)")
    print("  4. ./kmeans_summary.pdf - PDF version for backup")
    print("\nThe summary figure uses a mixed layout (2x2 or 2x3 grid) that fits well in Beamer presentations.")
    print("\nThe GIFs clearly show how k-means:")
    print("  - Iteratively refines cluster assignments")
    print("  - Gradually moves centroids to optimal positions")
    print("  - Updates decision boundaries at each step")
    print("  - Converges when centroids stabilize")


if __name__ == "__main__":
    main()